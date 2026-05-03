"""
Encode an audio file into a flat int32 token stream for autoregressive LM
training. EnCodec 24 kHz mono at 6.0 kbps gives 8 codebooks @ 75 Hz, each
with vocab=1024. We interleave the codebooks (cb0_t, cb1_t, ..., cb7_t,
cb0_{t+1}, ...) and offset codebook k by k*1024 so the flat stream has
vocab=8192 at 600 tokens/s.

Output format matches `tools/tinystories_dump.py` so the existing trainer
can read it unchanged:

  audio_train.bin / audio_valid.bin
    [u32 count][i32 token_0][i32 token_1]...

Run from the repo root:

  python tools/audio_lm_dump.py --input audio_sources/WeirdSongLow.wav --out-dir examples/data --prefix audio
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np
import torch
import torchaudio
from transformers import EncodecModel


ENCODEC_24K_ID      = "facebook/encodec_24khz"
ENCODEC_SAMPLE_RATE = 24000
ENCODEC_FRAME_RATE  = 75
CODEBOOK_VOCAB      = 1024
NUM_CODEBOOKS       = 8
BANDWIDTH_KBPS      = 6.0
FLAT_VOCAB          = CODEBOOK_VOCAB * NUM_CODEBOOKS  # = 8192


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input",       required=True, type=Path)
    p.add_argument("--out-dir",     required=True, type=Path)
    p.add_argument("--prefix",      default="audio")
    p.add_argument("--valid-frac",  type=float, default=0.1)
    p.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def load_audio_24k_mono(path: Path) -> torch.Tensor:
    waveform, sr = torchaudio.load(str(path))
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != ENCODEC_SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, ENCODEC_SAMPLE_RATE)
    return waveform


@torch.no_grad()
def encode_full(model: EncodecModel, waveform: torch.Tensor, device: str) -> np.ndarray:
    """Returns codes shape [num_codebooks, frames] int64."""
    waveform = waveform.unsqueeze(0).to(device)                                # [1, 1, samples]
    out = model.encode(waveform, bandwidth=BANDWIDTH_KBPS)
    chunked = [chunk[0] for chunk in out.audio_codes]                          # list of [num_codebooks, frames_chunk]
    codes = torch.cat(chunked, dim=-1)                                         # [num_codebooks, frames]
    return codes.cpu().numpy()


def interleave_with_offset(codes: np.ndarray) -> np.ndarray:
    """codes: [num_codebooks, frames] -> flat int32 [frames * num_codebooks].

    Codebook k is shifted into [k*1024, (k+1)*1024).
    """
    assert codes.shape[0] == NUM_CODEBOOKS, codes.shape
    flat = np.empty(codes.shape[1] * NUM_CODEBOOKS, dtype=np.int32)
    for k in range(NUM_CODEBOOKS):
        flat[k::NUM_CODEBOOKS] = codes[k].astype(np.int64) + k * CODEBOOK_VOCAB
    return flat


def write_token_bin(path: Path, tokens: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack("<I", len(tokens)))
        f.write(tokens.astype(np.int32).tobytes())


@torch.no_grad()
def main() -> None:
    args = parse_args()

    print(f"Loading EnCodec ({ENCODEC_24K_ID}) on {args.device} ...")
    codec = EncodecModel.from_pretrained(ENCODEC_24K_ID).to(args.device).eval()

    print(f"Loading {args.input} ...")
    waveform = load_audio_24k_mono(args.input)
    duration_seconds = waveform.shape[1] / ENCODEC_SAMPLE_RATE
    print(f"  {waveform.shape[1]} samples, {duration_seconds:.2f} s @ {ENCODEC_SAMPLE_RATE} Hz")

    print(f"Encoding @ {BANDWIDTH_KBPS} kbps -> {NUM_CODEBOOKS} codebooks @ {ENCODEC_FRAME_RATE} Hz ...")
    codes = encode_full(codec, waveform, args.device)                          # [num_codebooks, frames]
    print(f"  codes shape = {codes.shape}")

    flat = interleave_with_offset(codes)
    print(f"  flat tokens = {len(flat):,} (vocab = {FLAT_VOCAB})")

    pivot_frames = int(codes.shape[1] * (1.0 - args.valid_frac))
    pivot_tokens = pivot_frames * NUM_CODEBOOKS
    train_tokens = flat[:pivot_tokens]
    valid_tokens = flat[pivot_tokens:]

    train_path = args.out_dir / f"{args.prefix}_train.bin"
    valid_path = args.out_dir / f"{args.prefix}_valid.bin"
    write_token_bin(train_path, train_tokens)
    write_token_bin(valid_path, valid_tokens)

    train_seconds = len(train_tokens) / NUM_CODEBOOKS / ENCODEC_FRAME_RATE
    valid_seconds = len(valid_tokens) / NUM_CODEBOOKS / ENCODEC_FRAME_RATE
    print(f"Wrote {train_path}  ({len(train_tokens):,} tokens, {train_seconds:.1f} s)")
    print(f"Wrote {valid_path}  ({len(valid_tokens):,} tokens, {valid_seconds:.1f} s)")


if __name__ == "__main__":
    main()
