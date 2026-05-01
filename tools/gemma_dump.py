"""
Download Gemma 4 E4B from HuggingFace, run a fixed prompt through the
reference implementation, and dump:

  - gemma_data/model.safetensors       (HF checkpoint, copied from the cache)
  - gemma_data/tokenizer.model         (SentencePiece model file)
  - gemma_data/prompt_tokens.bin       (int32 array, T entries)
  - gemma_data/expected_logits.bin     (float32 array, T * vocab entries)
  - gemma_data/config.json             (full HF config, verbatim)
  - gemma_data/layer_config.json       (distilled per-layer info we care about:
                                        attention_type, rope base, rope_fraction,
                                        num_kv_shared_layers, PLE dims, RoPE
                                        rotation convention sniffed from the
                                        reference impl)

The Odin example then loads the safetensors, runs forward on the same
tokens, and asserts logits match within a bf16-friendly tolerance.

Run from the repo root:

    python tools/gemma_dump.py
"""

from __future__ import annotations

import argparse
import inspect
import json
import shutil
import struct
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

MODEL_ID       = "google/gemma-4-e4b-it"
PROMPT         = "The capital of France is"
OUT_DIR        = Path("gemma_data")
TOKENS_PATH    = OUT_DIR / "prompt_tokens.bin"
LOGITS_PATH    = OUT_DIR / "expected_logits.bin"
MODEL_PATH     = OUT_DIR / "model.safetensors"
TOKENIZER_PATH = OUT_DIR / "tokenizer.model"
CONFIG_PATH    = OUT_DIR / "config.json"
LAYER_PATH     = OUT_DIR / "layer_config.json"
PARITY_PATH    = OUT_DIR / "tokenizer_parity.json"

PARITY_PROMPTS = [
    "The capital of France is",
    "Hello, world!",
    "   leading spaces and  internal  doubles ",
    "Numbers like 42, 3.14, and 1000000.",
    "Don't worry, it's fine — we'll see.",
    "Mix of tabs\tand\nnewlines.",
    "日本語のテスト。",
    "Mixed: 한국어 and العربية and русский.",
    "def fibonacci(n):\n    if n < 2: return n\n    return fibonacci(n-1) + fibonacci(n-2)",
    "",
    " ",
    "  ",
    "\n",
    "\t\t",
    "a",
    "A",
    "AB",
    "abc",
    "The quick brown fox jumps over the lazy dog.",
    "He said, \"Hello!\" and walked away.",
    "Curly “quotes” and ‘apostrophes’.",
    "Em—dash and en–dash and minus-hyphen.",
    "Ellipsis… and full-stop.",
    "Math: 1 + 2 = 3, 10 × 5 = 50, π ≈ 3.14159.",
    "Greek alphabet: αβγδεζηθ.",
    "Hebrew: שלום עולם.",
    "Thai: สวัสดีครับ.",
    "Hindi: नमस्ते.",
    "Emoji ride: 🚀 🌍 🤖 ✨️.",
    "Skin tones: 👋🏽 and 👩🏾.",
    "Family ZWJ: 👨‍👩‍👧‍👦.",
    "Combining marks: café vs café.",
    "URL: https://example.com/path?q=hello%20world#section.",
    "Email: someone+filter@example-host.co.uk.",
    "Path: C:\\Users\\corey\\Documents\\OdinStuff",
    "Path: /usr/local/bin/python3",
    "JSON: {\"name\": \"alice\", \"age\": 30, \"items\": [1, 2, 3]}",
    "XML: <root><child attr=\"value\"/></root>",
    "Markdown: **bold**, *italic*, `code`, [link](http://x.y).",
    "Triple backticks ```\ncode\n``` end.",
    "Long word: pneumonoultramicroscopicsilicovolcanoconiosis.",
    "Repeated:           many spaces.",
    "Repeated tabs:\t\t\t\t\tend.",
    "Repeated newlines:\n\n\n\n\nend.",
    "Mixed whitespace: a \t\nb \tc.",
    "Numbers in words: one hundred twenty-three thousand four hundred fifty-six.",
    "Scientific: 6.022e23, 1.6e-19, 2.998 × 10⁸ m/s.",
    "Hex: 0xDEADBEEF and 0xff00ff00.",
    "Roman: MCMLXXVIII and MMXXVI.",
    "Smileys :) :-D ;-) :P >_< T_T.",
    "C: int main(void) { return 0; }",
    "Rust: fn add(a: i32, b: i32) -> i32 { a + b }",
    "SQL: SELECT * FROM users WHERE id = 42 ORDER BY name DESC;",
]


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


def copy_hub_file(repo_id: str, filename: str, dest: Path) -> Path | None:
    try:
        src = Path(hf_hub_download(repo_id, filename))
    except Exception as err:
        print(f"  skipped {filename}: {err}")
        return None
    # Always overwrite. Two Gemma 4 variants (base vs IT) have identical
    # safetensors sizes, so a size-only check could leave stale weights
    # from a previous run in place after switching MODEL_ID.
    shutil.copyfile(src, dest)
    print(f"  -> {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
    return dest


def sniff_rope_convention(model) -> str:
    """
    Inspect the reference model's RoPE application to record whether it uses
    `rotate_half` (HF default: split halves, swap, negate) or interleaved
    pairs (our convention). Captured by reading the source of the rotary
    helper used by the first attention layer.
    """
    try:
        text_model = getattr(model, "language_model", None) or getattr(model, "model", model)
        layers = getattr(text_model, "layers", None) or getattr(getattr(text_model, "language_model", text_model), "layers")
        first_layer = layers[0]
        attn = first_layer.self_attn
        candidates = []
        for name in ("rotary_emb", "_rope", "rope"):
            obj = getattr(attn, name, None)
            if obj is not None:
                candidates.append((name, obj))
        # Also fish out the apply function from the module the layer lives in.
        module = inspect.getmodule(attn)
        if module is not None:
            for name, obj in inspect.getmembers(module):
                if callable(obj) and "rotary_pos_emb" in name.lower():
                    candidates.append((name, obj))
        sources = []
        for name, obj in candidates:
            try:
                sources.append(f"# {name}\n" + inspect.getsource(obj))
            except (OSError, TypeError):
                continue
        joined = "\n\n".join(sources)
        if "rotate_half" in joined:
            return "rotate_half"
        if "x[..., ::2]" in joined or "x[..., 1::2]" in joined:
            return "interleaved"
        return "unknown"
    except Exception as err:
        return f"error: {err}"


def extract_layer_config(config, model) -> dict:
    """
    Distil the bits the TODO calls out as open questions. Gemma 4 nests the
    text-stack config under `text_config`, so resolve names against that.
    """
    text_cfg = getattr(config, "text_config", config)

    def get(*names, default=None):
        for name in names:
            if hasattr(text_cfg, name):
                return getattr(text_cfg, name)
        return default

    rope_params = get("rope_parameters", default={}) or {}
    full_rope     = rope_params.get("full_attention", {})    if isinstance(rope_params, dict) else {}
    sliding_rope  = rope_params.get("sliding_attention", {}) if isinstance(rope_params, dict) else {}

    return {
        "model_id":                 MODEL_ID,
        "num_hidden_layers":        get("num_hidden_layers"),
        "hidden_size":              get("hidden_size"),
        "intermediate_size":        get("intermediate_size"),
        "num_attention_heads":      get("num_attention_heads"),
        "num_key_value_heads":      get("num_key_value_heads"),
        "head_dim":                 get("head_dim"),
        "global_head_dim":          get("global_head_dim"),
        "vocab_size":               get("vocab_size"),
        "max_position_embeddings":  get("max_position_embeddings"),
        "sliding_window":           get("sliding_window"),
        "layer_types":              get("layer_types"),
        "hidden_activation":        get("hidden_activation"),
        "final_logit_softcapping":  get("final_logit_softcapping"),
        "rope_full":                full_rope,
        "rope_sliding":             sliding_rope,
        "rms_norm_eps":             get("rms_norm_eps"),
        "num_kv_shared_layers":     get("num_kv_shared_layers"),
        "vocab_size_per_layer_input":   get("vocab_size_per_layer_input"),
        "hidden_size_per_layer_input":  get("hidden_size_per_layer_input"),
        "tie_word_embeddings":      get("tie_word_embeddings"),
        "rope_convention":          sniff_rope_convention(model),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt",   default=PROMPT)
    parser.add_argument("--model-id", default=MODEL_ID)
    args = parser.parse_args()

    OUT_DIR.mkdir(exist_ok=True)

    print(f"Downloading {args.model_id} weights ...")
    copy_hub_file(args.model_id, "model.safetensors", MODEL_PATH)
    for candidate in ("tokenizer.model", "spiece.model", "tokenizer.json"):
        if copy_hub_file(args.model_id, candidate, OUT_DIR / candidate) is not None:
            break

    print("Loading config ...")
    hf_config = AutoConfig.from_pretrained(args.model_id)
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(hf_config.to_dict(), f, indent=2, default=str)
    print(f"  -> {CONFIG_PATH}")

    print("Tokenizing prompt ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    token_ids = tokenizer(args.prompt, return_tensors="pt", add_special_tokens=False).input_ids[0]
    print(f"  prompt = {args.prompt!r}")
    print(f"  tokens = {token_ids.tolist()}")
    write_int_array(TOKENS_PATH, token_ids.numpy())

    parity = {
        prompt: tokenizer(prompt, add_special_tokens=False).input_ids
        for prompt in PARITY_PROMPTS
    }
    with open(PARITY_PATH, "w", encoding="utf-8") as f:
        json.dump(parity, f, indent=2, ensure_ascii=False)
    print(f"  -> {PARITY_PATH} ({len(parity)} prompts)")

    print("Running HF reference forward ...")
    model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        out = model(token_ids.unsqueeze(0), output_hidden_states=True)
        logits = out.logits[0]
        # Pre-lm_head hidden state = last entry of `hidden_states` after the final
        # `model.norm` is applied. transformers returns `hidden_states[-1]` as the
        # final-norm output, then `lm_head(hidden_states[-1])` gives the logits.
        final_hidden = out.hidden_states[-1][0]
    write_tensor(LOGITS_PATH, logits.numpy())
    print(f"  -> {LOGITS_PATH}, shape={tuple(logits.shape)}")
    write_tensor(OUT_DIR / "expected_final_hidden.bin", final_hidden.numpy())
    print(f"  -> {OUT_DIR / 'expected_final_hidden.bin'}, shape={tuple(final_hidden.shape)}")

    layer_cfg = extract_layer_config(hf_config, model)
    with open(LAYER_PATH, "w", encoding="utf-8") as f:
        json.dump(layer_cfg, f, indent=2, default=str)
    print(f"  -> {LAYER_PATH}")
    print(json.dumps(layer_cfg, indent=2, default=str))

    print("Top-5 next-token predictions per position:")
    for position in range(logits.shape[0]):
        top = torch.topk(logits[position], k=5)
        decoded = [tokenizer.decode([int(t)]) for t in top.indices]
        print(f"  pos {position}: {decoded}")


if __name__ == "__main__":
    main()
