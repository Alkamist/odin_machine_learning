"""
Round-trip a stereo audio file through Descript Audio Codec at 44.1 kHz and
write the reconstructed waveform back out. DAC is a mono codec, so the two
channels are encoded and decoded independently and then restacked.

Run from the repo root:

  python tools/dac_roundtrip.py
  python tools/dac_roundtrip.py --input audio_sources/WeirdSongLow.wav --out recon.wav
  python tools/dac_roundtrip.py --n-quantizers 4    # lower bitrate, lower quality
"""

from __future__ import annotations

import argparse
from pathlib import Path

import dac
import torch
import torchaudio


DAC_MODEL_TYPE      = "44khz"
DEFAULT_INPUT_PATH  = Path("audio_sources/WeirdSongLow.wav")
DEFAULT_OUTPUT_PATH = Path("audio_sources/WeirdSongLow_dac_roundtrip.wav")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input",        type=Path, default=DEFAULT_INPUT_PATH)
    p.add_argument("--out",          type=Path, default=DEFAULT_OUTPUT_PATH)
    p.add_argument("--n-quantizers", type=int,  default=None, help="codebooks to use; default = all (highest quality)")
    p.add_argument("--device",       default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


@torch.no_grad()
def roundtrip_channel(model: dac.DAC, channel_waveform: torch.Tensor, n_quantizers: int | None, device: str) -> tuple[torch.Tensor, int]:
    """channel_waveform: [1, samples] at model.sample_rate. Returns (decoded [samples_out], code_count)."""
    x = channel_waveform.unsqueeze(0).to(device)            # [1, 1, samples]
    x = model.preprocess(x, model.sample_rate)
    _, codes, _, _, _ = model.encode(x, n_quantizers=n_quantizers)
    z_q, _, _ = model.quantizer.from_codes(codes)
    decoded = model.decode(z_q)                              # [1, 1, samples_out]
    return decoded[0, 0].cpu(), int(codes.numel())


def to_stereo(waveform: torch.Tensor) -> torch.Tensor:
    if waveform.shape[0] == 1:
        print("(mono input duplicated to stereo for the round-trip)")
        return waveform.repeat(2, 1)
    if waveform.shape[0] > 2:
        print(f"(downmixing {waveform.shape[0]} channels to stereo by taking the first 2)")
        return waveform[:2]
    return waveform


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise SystemExit(f"missing input audio: {args.input}")

    print(f"Loading DAC {DAC_MODEL_TYPE} on {args.device} ...")
    weights = dac.utils.download(model_type=DAC_MODEL_TYPE)
    model   = dac.DAC.load(weights).to(args.device).eval()
    target_sr = model.sample_rate

    waveform, sr = torchaudio.load(str(args.input))
    print(f"Input: {waveform.shape[0]} ch, {sr} Hz, {waveform.shape[1] / sr:.2f} s ({args.input})")

    if sr != target_sr:
        print(f"Resampling {sr} -> {target_sr} ...")
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
        sr = target_sr

    waveform = to_stereo(waveform)

    n_quantizers = args.n_quantizers if args.n_quantizers is not None else model.n_codebooks
    print(f"Using {n_quantizers}/{model.n_codebooks} quantizers")

    decoded_channels: list[torch.Tensor] = []
    total_codes = 0
    for channel_index in range(2):
        single_channel = waveform[channel_index : channel_index + 1]
        decoded_waveform, code_count = roundtrip_channel(model, single_channel, n_quantizers, args.device)
        total_codes += code_count
        decoded_channels.append(decoded_waveform)

    common_length  = min(channel.shape[0] for channel in decoded_channels)
    stereo_output  = torch.stack([channel[:common_length] for channel in decoded_channels], dim=0)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(args.out), stereo_output, sr)

    duration_seconds = stereo_output.shape[1] / sr
    effective_kbps   = total_codes * 10 / duration_seconds / 1000
    frame_rate       = model.sample_rate / model.hop_length
    print(f"Wrote {args.out} ({duration_seconds:.2f} s @ {sr} Hz)")
    print(f"  frame_rate={frame_rate:.2f} Hz, both-channel bitrate ~{effective_kbps:.1f} kbps")


if __name__ == "__main__":
    main()
