"""
Encode an audio corpus into the paired (text, audio_codes) binary format
the Odin trainer consumes. Also tokenizes optional sidecar captions with
the SmolLM2 BPE tokenizer.

Layout:

  audio_data/
    *.wav                # any audio readable by torchaudio
    *.txt                # optional sidecar caption (UTF-8); same stem as the .wav

Outputs (--out PREFIX writes PREFIX.bin and PREFIX.idx):

  PREFIX.bin
    [header 64B]
    [record 0]
    [record 1]
    ...
  PREFIX.idx
    [u32 example_count][u32 reserved=0]
    [int64 offset_0]
    [int64 offset_1]
    ...

Header (64 bytes, little-endian):

  u32 magic            = 0xC0DECDAA
  u32 version          = 1
  u32 example_count
  u32 num_codebooks
  u32 sample_rate
  u32 codec_frame_rate
  u32 vocab_per_codebook
  u32 text_vocab_size
  u8  reserved[32]

Per-record:

  u32 text_len
  u32 audio_frames
  int32 text_ids[text_len]
  int32 audio_codes[audio_frames * num_codebooks]   # row-major: frame, codebook

Run from the repo root:

  python tools/audio_dump.py --input-dir my_audio --out audio_data/sample [--bandwidth 3.0]
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np
import torch
import torchaudio
from tokenizers import Tokenizer
from transformers import EncodecModel

MAGIC               = 0xC0DECDAA
VERSION             = 1
HEADER_BYTES        = 64
ENCODEC_24K_ID      = "facebook/encodec_24khz"
ENCODEC_SAMPLE_RATE = 24000
ENCODEC_FRAME_RATE  = 75      # the 24kHz model emits 75 frames/sec.
ENCODEC_VOCAB       = 1024    # per-codebook codebook size.

DEFAULT_TOKENIZER   = Path("smollm_data/tokenizer.json")
DEFAULT_BANDWIDTH   = 3.0     # kbps. 1.5/3.0/6.0/12.0/24.0 -> 2/4/8/16/32 codebooks.
DEFAULT_MAX_SECONDS = 30.0    # clip audio longer than this.

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir",   required=True, type=Path)
    p.add_argument("--out",         required=True, type=Path, help="output prefix; writes <out>.bin and <out>.idx")
    p.add_argument("--tokenizer",   type=Path, default=DEFAULT_TOKENIZER)
    p.add_argument("--bandwidth",   type=float, default=DEFAULT_BANDWIDTH)
    p.add_argument("--max-seconds", type=float, default=DEFAULT_MAX_SECONDS)
    p.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--limit",       type=int, default=0, help="cap the number of files; 0 = no cap")
    return p.parse_args()


def discover_inputs(root: Path) -> list[Path]:
    paths = sorted(p for p in root.rglob("*") if p.suffix.lower() in AUDIO_EXTENSIONS and p.is_file())
    return paths


def load_audio_24k_mono(path: Path) -> torch.Tensor:
    waveform, sr = torchaudio.load(str(path))
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != ENCODEC_SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, ENCODEC_SAMPLE_RATE)
    return waveform


@torch.no_grad()
def encode_audio(model: EncodecModel, waveform: torch.Tensor, bandwidth: float, device: str) -> np.ndarray:
    """Return [audio_frames, num_codebooks] int32 codes."""
    waveform = waveform.unsqueeze(0).to(device)  # [1, 1, samples]
    out = model.encode(waveform, bandwidth=bandwidth)
    # encoder returns audio_codes shape [chunks, batch, num_codebooks, frames];
    # for inputs shorter than the codec chunk_length there's only one chunk.
    codes = out.audio_codes[0, 0]                     # [num_codebooks, frames]
    return codes.transpose(0, 1).contiguous().cpu().numpy().astype(np.int32)


def write_header(f, example_count: int, num_codebooks: int, text_vocab_size: int) -> None:
    payload = struct.pack(
        "<IIIIIIII",
        MAGIC, VERSION, example_count, num_codebooks,
        ENCODEC_SAMPLE_RATE, ENCODEC_FRAME_RATE, ENCODEC_VOCAB, text_vocab_size,
    )
    f.write(payload)
    f.write(b"\x00" * (HEADER_BYTES - len(payload)))


def patch_example_count(bin_path: Path, count: int) -> None:
    with open(bin_path, "r+b") as f:
        f.seek(8)  # past magic + version
        f.write(struct.pack("<I", count))


def main() -> None:
    args = parse_args()

    if not args.tokenizer.exists():
        raise SystemExit(f"missing tokenizer at {args.tokenizer}; run tools/smollm_dump.py first")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    bin_path = args.out.with_suffix(".bin")
    idx_path = args.out.with_suffix(".idx")

    print(f"Loading EnCodec ({ENCODEC_24K_ID}) on {args.device} ...")
    codec = EncodecModel.from_pretrained(ENCODEC_24K_ID).to(args.device).eval()
    num_codebooks = bandwidth_to_codebooks(args.bandwidth, codec)
    print(f"  bandwidth={args.bandwidth} kbps -> {num_codebooks} codebooks @ {ENCODEC_FRAME_RATE} Hz")

    print(f"Loading tokenizer {args.tokenizer} ...")
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    text_vocab = tokenizer.get_vocab_size()
    print(f"  text vocab: {text_vocab:,}")

    inputs = discover_inputs(args.input_dir)
    if args.limit > 0:
        inputs = inputs[: args.limit]
    print(f"Found {len(inputs)} audio files under {args.input_dir}")
    if not inputs:
        raise SystemExit("nothing to encode")

    max_samples = int(args.max_seconds * ENCODEC_SAMPLE_RATE)

    offsets: list[int] = []
    skipped = 0
    total_audio_frames = 0
    total_text_tokens  = 0

    with open(bin_path, "wb") as bin_f:
        write_header(bin_f, example_count=0, num_codebooks=num_codebooks, text_vocab_size=text_vocab)

        for i, audio_path in enumerate(inputs):
            try:
                waveform = load_audio_24k_mono(audio_path)
            except Exception as e:
                print(f"  [skip] {audio_path}: {e}")
                skipped += 1
                continue

            if waveform.shape[1] > max_samples:
                waveform = waveform[:, :max_samples]
            if waveform.shape[1] < ENCODEC_SAMPLE_RATE // 10:  # < 100ms
                skipped += 1
                continue

            codes = encode_audio(codec, waveform, args.bandwidth, args.device)
            audio_frames = codes.shape[0]
            assert codes.shape[1] == num_codebooks, codes.shape

            caption_path = audio_path.with_suffix(".txt")
            if caption_path.exists():
                caption = caption_path.read_text(encoding="utf-8").strip()
                text_ids = tokenizer.encode(caption, add_special_tokens=False).ids
            else:
                text_ids = []

            text_arr = np.asarray(text_ids, dtype=np.int32)

            offsets.append(bin_f.tell())
            bin_f.write(struct.pack("<II", len(text_arr), audio_frames))
            bin_f.write(text_arr.tobytes())
            bin_f.write(codes.tobytes())

            total_audio_frames += audio_frames
            total_text_tokens  += len(text_arr)

            if (i + 1) % 50 == 0 or (i + 1) == len(inputs):
                print(f"  [{i+1:>5}/{len(inputs)}] {audio_path.name}  audio={audio_frames}f  text={len(text_arr)}t")

    patch_example_count(bin_path, len(offsets))

    with open(idx_path, "wb") as idx_f:
        idx_f.write(struct.pack("<II", len(offsets), 0))
        idx_f.write(np.asarray(offsets, dtype=np.int64).tobytes())

    print(f"Done. {len(offsets)} examples, {skipped} skipped.")
    print(f"  audio: {total_audio_frames:,} frames ({total_audio_frames / ENCODEC_FRAME_RATE / 60:.1f} minutes total)")
    print(f"  text:  {total_text_tokens:,} tokens")
    print(f"  wrote {bin_path}  ({bin_path.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"  wrote {idx_path}  ({idx_path.stat().st_size / 1024:.1f} KB)")


def bandwidth_to_codebooks(bandwidth: float, codec: EncodecModel) -> int:
    available = codec.config.target_bandwidths
    if bandwidth not in available:
        raise SystemExit(f"bandwidth {bandwidth} not in {available}")
    # 24kHz EnCodec uses 75 Hz frame rate, 10-bit codes -> bandwidth_kbps = 0.75 * num_codebooks.
    return int(round(bandwidth / 0.75))


if __name__ == "__main__":
    main()
