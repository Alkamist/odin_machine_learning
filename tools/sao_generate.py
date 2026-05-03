"""
Generate stereo audio from a text prompt using the full Stable Audio Open
1.0 pipeline (T5 text encoder -> DiT denoiser -> Oobleck VAE decoder).

This is the Python reference for the eventual Odin port. The output here
is the ground truth that the Odin implementation has to reproduce.

Requires huggingface auth: accept the license at
https://huggingface.co/stabilityai/stable-audio-open-1.0 and
`huggingface-cli login` first.

  python tools/sao_generate.py --prompt "tight 808 kick, dry, no reverb" --out gen_kick.wav --seconds 4
  python tools/sao_generate.py --prompt "lush analog synth pad, slow attack" --out gen_pad.wav --seconds 8 --steps 100 --cfg 7.0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torchaudio
from diffusers import StableAudioPipeline


REPO_ID = "stabilityai/stable-audio-open-1.0"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--prompt",         required=True, type=str)
    p.add_argument("--out",            required=True, type=Path)
    p.add_argument("--negative-prompt", type=str, default="low quality, distorted")
    p.add_argument("--seconds",        type=float, default=10.0)
    p.add_argument("--steps",          type=int,   default=50)
    p.add_argument("--cfg",            type=float, default=7.0, help="classifier-free guidance scale")
    p.add_argument("--seed",           type=int,   default=0)
    p.add_argument("--device",         default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype",          default="float16", choices=["float16", "bfloat16", "float32"])
    return p.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()

    torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    if args.device == "cpu" and torch_dtype == torch.float16:
        print("(cpu device: forcing float32; fp16 cpu is unsupported)")
        torch_dtype = torch.float32

    print(f"Loading Stable Audio Open 1.0 on {args.device} ({args.dtype}) ...")
    pipe = StableAudioPipeline.from_pretrained(REPO_ID, torch_dtype=torch_dtype).to(args.device)

    target_sr = pipe.vae.config.sampling_rate
    print(f"  sampling_rate={target_sr}  steps={args.steps}  cfg={args.cfg}  seed={args.seed}")
    print(f"  prompt: {args.prompt!r}")

    generator = torch.Generator(device=args.device).manual_seed(args.seed)

    result = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        num_inference_steps=args.steps,
        audio_end_in_s=args.seconds,
        guidance_scale=args.cfg,
        generator=generator,
    )

    audio_output = result.audios[0].to(torch.float32).cpu()                 # [2, T]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(args.out), audio_output, target_sr)
    duration_seconds = audio_output.shape[1] / target_sr
    print(f"Wrote {args.out} ({duration_seconds:.2f} s @ {target_sr} Hz)")


if __name__ == "__main__":
    main()
