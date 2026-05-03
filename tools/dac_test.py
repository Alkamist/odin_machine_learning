"""
Round-trip a stereo audio file through Descript Audio Codec to subjectively
evaluate fidelity at a chosen quantizer count and model rate. Each channel
is encoded independently (DAC is mono).

  python tools/dac_test.py --input audio_sources/WeirdSongLow.wav --out recon_dac.wav --model-type 44khz --n-quantizers 9
  python tools/dac_test.py --input audio_sources/WeirdSongLow.wav --out recon_dac24.wav --model-type 24khz --n-quantizers 32
"""

from __future__ import annotations

import argparse
from pathlib import Path

import dac
import torch
import torchaudio


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input",        required=True, type=Path)
    p.add_argument("--out",          required=True, type=Path)
    p.add_argument("--model-type",   default="44khz", choices=["44khz", "24khz", "16khz"])
    p.add_argument("--n-quantizers", type=int, default=None, help="codebooks to use; default = all available")
    p.add_argument("--device",       default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


@torch.no_grad()
def roundtrip_channel(model: dac.DAC, channel_waveform: torch.Tensor, n_quantizers: int | None, device: str) -> tuple[torch.Tensor, int]:
    """channel_waveform: [1, T] at model.sample_rate. Returns (decoded [T_out], code_count)."""
    x = channel_waveform.unsqueeze(0).to(device)            # [1, 1, T]
    x = model.preprocess(x, model.sample_rate)
    _, codes, _, _, _ = model.encode(x, n_quantizers=n_quantizers)
    z_q, _, _ = model.quantizer.from_codes(codes)
    decoded = model.decode(z_q)                              # [1, 1, T_out]
    return decoded[0, 0].cpu(), int(codes.numel())


def main() -> None:
    args = parse_args()

    print(f"Loading DAC {args.model_type} on {args.device} ...")
    weights = dac.utils.download(model_type=args.model_type)
    model = dac.DAC.load(weights).to(args.device).eval()
    target_sr = model.sample_rate

    waveform, sr = torchaudio.load(str(args.input))
    print(f"Input: {waveform.shape[0]} ch, {sr} Hz, {waveform.shape[1] / sr:.2f} s")

    if sr != target_sr:
        print(f"Resampling {sr} -> {target_sr} ...")
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
        sr = target_sr

    if waveform.shape[0] == 1:
        print("(mono input duplicated to stereo for the round-trip)")
        waveform = waveform.repeat(2, 1)
    elif waveform.shape[0] > 2:
        print(f"(downmixing {waveform.shape[0]} channels to stereo by taking first 2)")
        waveform = waveform[:2]

    n_quantizers = args.n_quantizers if args.n_quantizers is not None else model.n_codebooks
    print(f"Using {n_quantizers} quantizers (max {model.n_codebooks})")

    decoded_channels = []
    total_codes = 0
    for channel_index in range(2):
        single_channel = waveform[channel_index : channel_index + 1]
        decoded_waveform, code_count = roundtrip_channel(model, single_channel, n_quantizers, args.device)
        total_codes += code_count
        decoded_channels.append(decoded_waveform)

    common_length = min(channel_waveform.shape[0] for channel_waveform in decoded_channels)
    stereo_output = torch.stack([channel_waveform[:common_length] for channel_waveform in decoded_channels], dim=0)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(args.out), stereo_output, sr)

    duration_seconds = stereo_output.shape[1] / sr
    effective_kbps = total_codes * 10 / duration_seconds / 1000
    frame_rate = model.sample_rate / model.hop_length
    print(f"Wrote {args.out} ({duration_seconds:.2f} s @ {sr} Hz)")
    print(f"  frame_rate={frame_rate:.2f} Hz, both-channel bitrate ~{effective_kbps:.1f} kbps")


if __name__ == "__main__":
    main()
