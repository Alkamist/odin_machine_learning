"""
Decode a flat int32 token stream (vocab=8192, interleaved 8-codebook
EnCodec) back to a waveform. Used to:

  - sanity-check the dumper round-trips (codec ceiling at 6.0 kbps),
  - listen to samples emitted by the Odin trainer.

Run from the repo root:

  python tools/audio_lm_decode.py --input examples/data/audio_train.bin --out roundtrip.wav
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
CODEBOOK_VOCAB      = 1024
NUM_CODEBOOKS       = 8


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input",  required=True, type=Path, help="flat int32 token .bin written by audio_lm_dump.py")
    p.add_argument("--out",    required=True, type=Path)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def read_token_bin(path: Path) -> np.ndarray:
    data = path.read_bytes()
    count = struct.unpack_from("<I", data, 0)[0]
    tokens = np.frombuffer(data, dtype=np.int32, count=count, offset=4).copy()
    return tokens


def deinterleave(tokens: np.ndarray) -> np.ndarray:
    """flat int32 [frames * num_codebooks] -> codes [num_codebooks, frames] int64.

    Drops a trailing partial frame if present, and clamps any out-of-range
    values back into [0, 1024) so model samples don't crash the decoder.
    """
    usable = (len(tokens) // NUM_CODEBOOKS) * NUM_CODEBOOKS
    tokens = tokens[:usable].astype(np.int64)

    cb_rows: list[np.ndarray] = []
    for k in range(NUM_CODEBOOKS):
        row = tokens[k::NUM_CODEBOOKS] - k * CODEBOOK_VOCAB
        cb_rows.append(np.clip(row, 0, CODEBOOK_VOCAB - 1))
    return np.stack(cb_rows, axis=0)


@torch.no_grad()
def decode_audio(codec: EncodecModel, codes: np.ndarray, device: str) -> torch.Tensor:
    arr = torch.from_numpy(codes).unsqueeze(0).unsqueeze(0).long().to(device)  # [chunks=1, batch=1, num_codebooks, frames]
    out = codec.decode(arr, audio_scales=[None])
    return out.audio_values[0, 0].cpu()


def main() -> None:
    args = parse_args()

    tokens = read_token_bin(args.input)
    print(f"Read {len(tokens):,} tokens from {args.input}")

    codes = deinterleave(tokens)
    print(f"  codes shape = {codes.shape}  ({codes.shape[1] / 75:.2f} s of audio)")

    print(f"Loading EnCodec on {args.device} ...")
    codec = EncodecModel.from_pretrained(ENCODEC_24K_ID).to(args.device).eval()

    waveform = decode_audio(codec, codes, args.device)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(args.out), waveform.unsqueeze(0), ENCODEC_SAMPLE_RATE)
    print(f"Wrote {args.out} ({waveform.shape[0] / ENCODEC_SAMPLE_RATE:.2f} s @ {ENCODEC_SAMPLE_RATE} Hz)")


if __name__ == "__main__":
    main()
