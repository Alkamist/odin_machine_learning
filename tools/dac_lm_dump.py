"""
Encode an audio file into a flat int32 token stream for autoregressive LM
training over Descript Audio Codec at 44.1 kHz stereo.

DAC is mono, so each channel is encoded independently. We interleave the
two channels at every codebook level within each frame, coarse-to-fine:

    cb0_L_t0, cb0_R_t0, cb1_L_t0, cb1_R_t0, ..., cb8_L_t0, cb8_R_t0,
    cb0_L_t1, cb0_R_t1, ..., cb8_R_t1,
    cb0_L_t2, ...

There are 18 "slots" per frame (9 codebooks * 2 channels). Slot k uses the
codebook value range [k*1024, (k+1)*1024), so the flat stream has vocab
9 * 2 * 1024 = 18432 at ~1548 tokens/sec.

Output format matches `tools/audio_lm_dump.py`:

  dac_lm_train.bin / dac_lm_valid.bin
    [u32 count][i32 token_0][i32 token_1]...

Run from the repo root:

  python tools/dac_lm_dump.py --input audio_sources/WeirdSongLow.wav --out-dir examples/data --prefix dac_lm
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import dac
import numpy as np
import torch
import torchaudio


DAC_MODEL_TYPE  = "44khz"
NUM_CHANNELS    = 2
CODEBOOK_VOCAB  = 1024


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input",      required=True, type=Path)
    p.add_argument("--out-dir",    required=True, type=Path)
    p.add_argument("--prefix",     default="dac_lm")
    p.add_argument("--valid-frac", type=float, default=0.1)
    p.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def load_audio_stereo(path: Path, target_sr: int) -> torch.Tensor:
    waveform, sr = torchaudio.load(str(path))
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    if waveform.shape[0] == 1:
        waveform = waveform.repeat(2, 1)
    elif waveform.shape[0] > 2:
        waveform = waveform[:2]
    return waveform


@torch.no_grad()
def encode_channel(model: dac.DAC, channel_waveform: torch.Tensor, device: str) -> np.ndarray:
    """channel_waveform: [1, samples]. Returns codes [num_codebooks, frames] int64."""
    x = channel_waveform.unsqueeze(0).to(device)            # [1, 1, samples]
    x = model.preprocess(x, model.sample_rate)
    _, codes, _, _, _ = model.encode(x)                      # [1, num_codebooks, frames]
    return codes[0].cpu().numpy().astype(np.int64)


def interleave_with_offset(codes_per_channel: list[np.ndarray]) -> np.ndarray:
    """codes_per_channel: list of 2 arrays [num_codebooks, frames] -> flat int32 [frames * 2 * num_codebooks].

    Slot s = k*2 + channel is shifted into [s*1024, (s+1)*1024).
    """
    assert len(codes_per_channel) == NUM_CHANNELS
    num_codebooks = codes_per_channel[0].shape[0]
    frames        = codes_per_channel[0].shape[1]
    for c in codes_per_channel:
        assert c.shape == (num_codebooks, frames), c.shape

    slot_count = num_codebooks * NUM_CHANNELS
    flat       = np.empty(frames * slot_count, dtype=np.int32)
    for k in range(num_codebooks):
        for channel_index in range(NUM_CHANNELS):
            slot = k * NUM_CHANNELS + channel_index
            flat[slot::slot_count] = codes_per_channel[channel_index][k].astype(np.int64) + slot * CODEBOOK_VOCAB
    return flat


def write_token_bin(path: Path, tokens: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack("<I", len(tokens)))
        f.write(tokens.astype(np.int32).tobytes())


@torch.no_grad()
def main() -> None:
    args = parse_args()

    print(f"Loading DAC {DAC_MODEL_TYPE} on {args.device} ...")
    weights = dac.utils.download(model_type=DAC_MODEL_TYPE)
    model   = dac.DAC.load(weights).to(args.device).eval()
    sample_rate   = model.sample_rate
    num_codebooks = model.n_codebooks
    frame_rate    = sample_rate / model.hop_length
    flat_vocab    = num_codebooks * NUM_CHANNELS * CODEBOOK_VOCAB

    print(f"Loading {args.input} ...")
    waveform = load_audio_stereo(args.input, sample_rate)
    duration_seconds = waveform.shape[1] / sample_rate
    print(f"  {waveform.shape[1]} samples, {duration_seconds:.2f} s @ {sample_rate} Hz, stereo")

    print(f"Encoding -> {num_codebooks} codebooks per channel @ ~{frame_rate:.2f} Hz ...")
    codes_per_channel: list[np.ndarray] = []
    for channel_index in range(NUM_CHANNELS):
        single_channel = waveform[channel_index : channel_index + 1]
        codes = encode_channel(model, single_channel, args.device)
        codes_per_channel.append(codes)
        if channel_index == 0:
            print(f"  per-channel codes shape = {codes.shape}")

    flat = interleave_with_offset(codes_per_channel)
    print(f"  flat tokens = {len(flat):,} (vocab = {flat_vocab})")

    pivot_frames = int(codes_per_channel[0].shape[1] * (1.0 - args.valid_frac))
    pivot_tokens = pivot_frames * num_codebooks * NUM_CHANNELS
    train_tokens = flat[:pivot_tokens]
    valid_tokens = flat[pivot_tokens:]

    train_path = args.out_dir / f"{args.prefix}_train.bin"
    valid_path = args.out_dir / f"{args.prefix}_valid.bin"
    write_token_bin(train_path, train_tokens)
    write_token_bin(valid_path, valid_tokens)

    train_seconds = len(train_tokens) / (num_codebooks * NUM_CHANNELS * frame_rate)
    valid_seconds = len(valid_tokens) / (num_codebooks * NUM_CHANNELS * frame_rate)
    print(f"Wrote {train_path}  ({len(train_tokens):,} tokens, {train_seconds:.1f} s)")
    print(f"Wrote {valid_path}  ({len(valid_tokens):,} tokens, {valid_seconds:.1f} s)")


if __name__ == "__main__":
    main()
