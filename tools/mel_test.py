"""
Round-trip a stereo audio file through a mel spectrogram + neural vocoder
to subjectively evaluate the upper-bound quality of the mel + vocoder
architecture (option 3) before any LM training.

  audio -> resample 44.1kHz -> per-channel mel -> BigVGAN -> stereo wav

  python tools/mel_test.py --input audio_sources/WeirdSongLow.wav --out recon_mel.wav
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torchaudio
from huggingface_hub import hf_hub_download

from bigvgan import BigVGAN
from bigvgan.env import AttrDict
from bigvgan.meldataset import mel_spectrogram


REPO_ID = "nvidia/bigvgan_v2_44khz_128band_512x"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input",  required=True, type=Path)
    p.add_argument("--out",    required=True, type=Path)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def load_bigvgan(device: str) -> tuple[BigVGAN, AttrDict]:
    """Sidestep BigVGAN.from_pretrained's huggingface_hub-version mismatch."""
    config_path  = hf_hub_download(repo_id=REPO_ID, filename="config.json")
    weights_path = hf_hub_download(repo_id=REPO_ID, filename="bigvgan_generator.pt")

    with open(config_path) as config_file:
        config = AttrDict(json.load(config_file))

    model = BigVGAN(config, use_cuda_kernel=False)
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
    try:
        model.load_state_dict(checkpoint["generator"])
    except RuntimeError:
        model.remove_weight_norm()
        model.load_state_dict(checkpoint["generator"])

    model.remove_weight_norm()
    model.eval().to(device)
    return model, config


@torch.no_grad()
def roundtrip_channel(model: BigVGAN, config: AttrDict, channel_waveform: torch.Tensor, device: str) -> torch.Tensor:
    """channel_waveform: [1, T] at config.sampling_rate. Returns decoded [T_out]."""
    mel = mel_spectrogram(
        channel_waveform.to(device),
        n_fft=config.n_fft,
        num_mels=config.num_mels,
        sampling_rate=config.sampling_rate,
        hop_size=config.hop_size,
        win_size=config.win_size,
        fmin=config.fmin,
        fmax=config.fmax,
    )                                    # [1, num_mels, T_frames]
    decoded = model(mel)                 # [1, 1, T_out]
    return decoded[0, 0].cpu()


def main() -> None:
    args = parse_args()

    print(f"Loading BigVGAN v2 (44 kHz / 128-band) on {args.device} ...")
    model, config = load_bigvgan(args.device)
    target_sr = config.sampling_rate
    print(f"  sampling_rate={target_sr} num_mels={config.num_mels} hop={config.hop_size} ({target_sr / config.hop_size:.2f} frames/s)")

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

    decoded_channels = []
    for channel_index in range(2):
        single_channel = waveform[channel_index : channel_index + 1]
        decoded_waveform = roundtrip_channel(model, config, single_channel, args.device)
        decoded_channels.append(decoded_waveform)

    common_length = min(channel_waveform.shape[0] for channel_waveform in decoded_channels)
    stereo_output = torch.stack([channel_waveform[:common_length] for channel_waveform in decoded_channels], dim=0)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(args.out), stereo_output, sr)
    print(f"Wrote {args.out} ({stereo_output.shape[1] / sr:.2f} s @ {sr} Hz)")


if __name__ == "__main__":
    main()
