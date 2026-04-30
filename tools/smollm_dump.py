"""
Download SmolLM2-135M from HuggingFace, run a fixed prompt through the
reference implementation, and dump:

  - smollm_data/model.safetensors      (the HF checkpoint, copied through huggingface_hub)
  - smollm_data/prompt_tokens.bin      (int32 array, T entries)
  - smollm_data/expected_logits.bin    (float32 array, T * vocab entries, F32-cast from HF's bf16)

The Odin example then loads the safetensors, runs forward on the same
tokens, and asserts logits match within a bf16-friendly tolerance.

Run from the repo root:

    python tools/smollm_dump.py
"""

from __future__ import annotations

import argparse
import shutil
import struct
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID    = "HuggingFaceTB/SmolLM2-135M"
PROMPT      = "The capital of France is"
OUT_DIR     = Path("smollm_data")
TOKENS_PATH = OUT_DIR / "prompt_tokens.bin"
LOGITS_PATH = OUT_DIR / "expected_logits.bin"
MODEL_PATH  = OUT_DIR / "model.safetensors"


def write_int_array(path: Path, values: np.ndarray) -> None:
    payload = values.astype(np.int32).tobytes()
    with open(path, "wb") as f:
        f.write(struct.pack("<I", len(values)))
        f.write(payload)


def write_tensor(path: Path, t: np.ndarray) -> None:
    t = t.astype(np.float32, copy=False)
    with open(path, "wb") as f:
        f.write(b"TNSR")
        f.write(struct.pack("<I", t.ndim))
        for axis in t.shape:
            f.write(struct.pack("<I", axis))
        f.write(t.tobytes())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default=PROMPT)
    args = parser.parse_args()

    OUT_DIR.mkdir(exist_ok=True)

    print(f"Downloading {MODEL_ID} weights ...")
    hf_safetensors = Path(hf_hub_download(MODEL_ID, "model.safetensors"))
    if not MODEL_PATH.exists() or MODEL_PATH.stat().st_size != hf_safetensors.stat().st_size:
        shutil.copyfile(hf_safetensors, MODEL_PATH)
    print(f"  -> {MODEL_PATH} ({MODEL_PATH.stat().st_size / 1e6:.1f} MB)")

    print("Tokenizing prompt ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    token_ids = tokenizer(args.prompt, return_tensors="pt", add_special_tokens=False).input_ids[0]
    print(f"  prompt = {args.prompt!r}")
    print(f"  tokens = {token_ids.tolist()}")
    write_int_array(TOKENS_PATH, token_ids.numpy())

    print("Running HF reference forward ...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        logits = model(token_ids.unsqueeze(0)).logits[0]   # [T, vocab]
    write_tensor(LOGITS_PATH, logits.numpy())
    print(f"  -> {LOGITS_PATH}, shape={tuple(logits.shape)}")

    print("Top-5 next-token predictions per position:")
    for position in range(logits.shape[0]):
        top = torch.topk(logits[position], k=5)
        decoded = [tokenizer.decode([int(t)]) for t in top.indices]
        print(f"  pos {position}: {decoded}")


if __name__ == "__main__":
    main()
