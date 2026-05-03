"""
Decode a flat int32 token stream (vocab = 9 * 2 * 1024 = 18432, frame-
interleaved 9-codebook stereo DAC) back to a waveform. Used to:

  - sanity-check the dumper round-trips,
  - listen to samples emitted by the Odin trainer.

Run from the repo root:

  python tools/dac_lm_decode.py --input examples/data/dac_lm_train.bin --out roundtrip.wav
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import dac
import numpy as np
import torch
import torchaudio


DAC_MODEL_TYPE = "44khz"
NUM_CHANNELS   = 2
CODEBOOK_VOCAB = 1024


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input",  required=True, type=Path, help="flat int32 token .bin written by dac_lm_dump.py")
    p.add_argument("--out",    required=True, type=Path)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def read_token_bin(path: Path) -> np.ndarray:
    data = path.read_bytes()
    count = struct.unpack_from("<I", data, 0)[0]
    return np.frombuffer(data, dtype=np.int32, count=count, offset=4).copy()


def deinterleave(tokens: np.ndarray, num_codebooks: int) -> list[np.ndarray]:
    """flat int32 [frames * slot_count] -> list of NUM_CHANNELS arrays [num_codebooks, frames] int64.

    Drops a trailing partial frame if present, and clamps any out-of-range
    values back into [0, 1024) so model samples don't crash the decoder.
    """
    slot_count = num_codebooks * NUM_CHANNELS
    usable     = (len(tokens) // slot_count) * slot_count
    tokens     = tokens[:usable].astype(np.int64)

    channels: list[np.ndarray] = []
    for channel_index in range(NUM_CHANNELS):
        rows: list[np.ndarray] = []
        for k in range(num_codebooks):
            slot = k * NUM_CHANNELS + channel_index
            row  = tokens[slot::slot_count] - slot * CODEBOOK_VOCAB
            rows.append(np.clip(row, 0, CODEBOOK_VOCAB - 1))
        channels.append(np.stack(rows, axis=0))
    return channels


@torch.no_grad()
def decode_channel(model: dac.DAC, codes: np.ndarray, device: str) -> torch.Tensor:
    """codes: [num_codebooks, frames] int64. Returns waveform [samples]."""
    arr = torch.from_numpy(codes).unsqueeze(0).long().to(device)   # [1, num_codebooks, frames]
    z_q, _, _ = model.quantizer.from_codes(arr)
    decoded = model.decode(z_q)                                     # [1, 1, samples]
    return decoded[0, 0].cpu()


def main() -> None:
    args = parse_args()

    print(f"Loading DAC {DAC_MODEL_TYPE} on {args.device} ...")
    weights       = dac.utils.download(model_type=DAC_MODEL_TYPE)
    model         = dac.DAC.load(weights).to(args.device).eval()
    sample_rate   = model.sample_rate
    num_codebooks = model.n_codebooks
    frame_rate    = sample_rate / model.hop_length

    tokens = read_token_bin(args.input)
    print(f"Read {len(tokens):,} tokens from {args.input}")

    codes_per_channel = deinterleave(tokens, num_codebooks)
    frames = codes_per_channel[0].shape[1]
    print(f"  per-channel codes shape = {codes_per_channel[0].shape}  ({frames / frame_rate:.2f} s of audio)")

    decoded_channels: list[torch.Tensor] = []
    for channel_index in range(NUM_CHANNELS):
        decoded_channels.append(decode_channel(model, codes_per_channel[channel_index], args.device))

    common_length = min(channel.shape[0] for channel in decoded_channels)
    stereo_output = torch.stack([channel[:common_length] for channel in decoded_channels], dim=0)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(args.out), stereo_output, sample_rate)
    print(f"Wrote {args.out} ({stereo_output.shape[1] / sample_rate:.2f} s @ {sample_rate} Hz, {stereo_output.shape[0]} ch)")


if __name__ == "__main__":
    main()
