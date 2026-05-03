"""
Read one example from a paired (text, audio_codes) binary written by
`audio_dump.py`, decode the audio codes back to a waveform with EnCodec,
and write a .wav. Prints the decoded caption (if any).

Use this to sanity-check that:
  - the binary format round-trips correctly,
  - EnCodec at the chosen bandwidth produces audible reconstructions,
  - the SmolLM2 tokenizer round-trips the captions you intend to train on.

Run from the repo root:

  python tools/audio_decode.py --in audio_data/sample --index 0 --out reconstructed.wav
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

MAGIC          = 0xC0DECDAA
HEADER_BYTES   = 64
ENCODEC_24K_ID = "facebook/encodec_24khz"

DEFAULT_TOKENIZER = Path("smollm_data/tokenizer.json")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--in",        dest="prefix", required=True, type=Path, help="prefix without extension; reads <prefix>.bin/.idx")
    p.add_argument("--index",     type=int, default=0)
    p.add_argument("--out",       type=Path, default=Path("reconstructed.wav"))
    p.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    p.add_argument("--device",    default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def read_header(buf: bytes) -> dict:
    magic, version, count, num_codebooks, sample_rate, frame_rate, vocab_per_cb, text_vocab = \
        struct.unpack_from("<IIIIIIII", buf, 0)
    if magic != MAGIC:
        raise SystemExit(f"bad magic 0x{magic:08x}")
    return {
        "version":        version,
        "count":          count,
        "num_codebooks":  num_codebooks,
        "sample_rate":    sample_rate,
        "frame_rate":     frame_rate,
        "vocab_per_cb":   vocab_per_cb,
        "text_vocab":     text_vocab,
    }


def load_index(idx_path: Path) -> np.ndarray:
    data = idx_path.read_bytes()
    count, _reserved = struct.unpack_from("<II", data, 0)
    offsets = np.frombuffer(data, dtype=np.int64, count=count, offset=8)
    return offsets.copy()


def read_record(bin_data: bytes, offset: int, num_codebooks: int) -> tuple[np.ndarray, np.ndarray]:
    text_len, audio_frames = struct.unpack_from("<II", bin_data, offset)
    cursor = offset + 8
    text_ids = np.frombuffer(bin_data, dtype=np.int32, count=text_len, offset=cursor).copy()
    cursor += text_len * 4
    codes = np.frombuffer(bin_data, dtype=np.int32, count=audio_frames * num_codebooks, offset=cursor).copy()
    return text_ids, codes.reshape(audio_frames, num_codebooks)


@torch.no_grad()
def decode_audio(codec: EncodecModel, codes: np.ndarray, device: str) -> torch.Tensor:
    """codes: [audio_frames, num_codebooks] int32 -> waveform tensor [samples]."""
    arr = torch.from_numpy(codes).t().contiguous()                                # [num_codebooks, frames]
    arr = arr.unsqueeze(0).unsqueeze(0).long().to(device)                         # [chunks=1, batch=1, num_codebooks, frames]
    audio_scales = [None]
    out = codec.decode(arr, audio_scales=audio_scales)
    return out.audio_values[0, 0].cpu()


def main() -> None:
    args = parse_args()

    bin_path = args.prefix.with_suffix(".bin")
    idx_path = args.prefix.with_suffix(".idx")

    bin_data = bin_path.read_bytes()
    header = read_header(bin_data)
    print(f"Header: {header}")

    offsets = load_index(idx_path)
    if args.index >= len(offsets):
        raise SystemExit(f"index {args.index} out of range (count={len(offsets)})")

    text_ids, codes = read_record(bin_data, int(offsets[args.index]), header["num_codebooks"])
    print(f"Example {args.index}: text_len={len(text_ids)}, audio_frames={codes.shape[0]}, codebooks={codes.shape[1]}")

    if len(text_ids) > 0:
        tokenizer = Tokenizer.from_file(str(args.tokenizer))
        caption = tokenizer.decode(text_ids.tolist(), skip_special_tokens=False)
        print(f"Caption: {caption!r}")

    print(f"Loading EnCodec on {args.device} ...")
    codec = EncodecModel.from_pretrained(ENCODEC_24K_ID).to(args.device).eval()

    waveform = decode_audio(codec, codes, args.device)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(args.out), waveform.unsqueeze(0), header["sample_rate"])

    seconds = waveform.shape[0] / header["sample_rate"]
    print(f"Wrote {args.out} ({seconds:.2f} s @ {header['sample_rate']} Hz)")


if __name__ == "__main__":
    main()
