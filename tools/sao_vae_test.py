"""
Round-trip a stereo audio file through Stable Audio Open's Oobleck VAE
to evaluate the upper-bound quality of option 1 (continuous-latent
diffusion) before any model training.

  audio -> resample 44.1kHz stereo -> VAE encode -> latent -> VAE decode -> stereo wav

The VAE compresses 44.1 kHz stereo to ~21.5 Hz, 64-dim continuous latents.
No diffusion is run; this only measures the autoencoder bottleneck.

Requires huggingface auth: accept the license at
https://huggingface.co/stabilityai/stable-audio-open-1.0 and
`huggingface-cli login` first.

  python tools/sao_vae_test.py --input audio_sources/WeirdSongLow.wav --out recon_sao.wav
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torchaudio
from diffusers import AutoencoderOobleck


REPO_ID = "stabilityai/stable-audio-open-1.0"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input",    required=True, type=Path)
    p.add_argument("--out",      required=True, type=Path)
    p.add_argument("--vae-path", type=Path, default=None,
                   help="local folder with config.json + diffusion_pytorch_model.safetensors; "
                        "if omitted, downloads from HuggingFace (requires `huggingface-cli login`)")
    p.add_argument("--device",   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()

    print(f"Loading Stable Audio Open VAE on {args.device} ...")
    if args.vae_path is not None:
        vae = AutoencoderOobleck.from_pretrained(str(args.vae_path)).to(args.device).eval()
    else:
        vae = AutoencoderOobleck.from_pretrained(REPO_ID, subfolder="vae").to(args.device).eval()
    target_sr = vae.config.sampling_rate
    print(f"  sampling_rate={target_sr}  audio_channels={vae.config.audio_channels}  latent_channels={vae.config.decoder_input_channels}")

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

    audio_input = waveform.unsqueeze(0).to(args.device)                  # [1, 2, T]

    encoder_output = vae.encode(audio_input).latent_dist
    latent = encoder_output.mode()                                        # [1, 64, T_latent]
    latent_frame_rate = sr * latent.shape[2] / audio_input.shape[2]
    print(f"Latent: {tuple(latent.shape)}  (~{latent_frame_rate:.2f} Hz, {latent.shape[1]} dims)")

    decoded = vae.decode(latent).sample                                   # [1, 2, T_out]
    stereo_output = decoded[0].cpu()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(args.out), stereo_output, sr)
    duration_seconds = stereo_output.shape[1] / sr
    print(f"Wrote {args.out} ({duration_seconds:.2f} s @ {sr} Hz)")


if __name__ == "__main__":
    main()
