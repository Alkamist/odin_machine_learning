"""
Download the TinyStories corpus from HuggingFace, tokenize it with SmolLM2's
GPT-2-style BPE tokenizer, and write the token IDs to flat int32 binaries
the Odin trainer can mmap.

  examples/data/tinystories_train.bin   [count: u32 LE][int32 ids ...]
  examples/data/tinystories_valid.bin   [count: u32 LE][int32 ids ...]

Run from the repo root:

    python tools/tinystories_dump.py
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

REPO_ID        = "roneneldan/TinyStories"
TRAIN_FILE     = "TinyStories-train.txt"
VALID_FILE     = "TinyStories-valid.txt"
TOKENIZER_PATH = Path("smollm_data/tokenizer.json")
OUT_DIR        = Path("examples/data")
OUT_TRAIN      = OUT_DIR / "tinystories_train.bin"
OUT_VALID      = OUT_DIR / "tinystories_valid.bin"


CHUNK_LINES = 4096


def encode_file_streaming(tokenizer: Tokenizer, src_path: Path, dst_path: Path) -> int:
    """Stream-encode `src_path` into `dst_path`. Reserves a 4-byte length
    header at the start, writes int32 token IDs as they're produced (in
    parallel batches), and patches the header at the end."""
    total_tokens = 0
    src_bytes = src_path.stat().st_size
    print(f"  text: {src_bytes:,} bytes")

    with open(dst_path, "wb") as out:
        out.write(struct.pack("<I", 0)) # length placeholder

        with open(src_path, "r", encoding="utf-8") as f:
            buffer: list[str] = []
            last_report = 0
            consumed_bytes = 0
            for line in f:
                buffer.append(line)
                consumed_bytes += len(line.encode("utf-8"))
                if len(buffer) >= CHUNK_LINES:
                    encs = tokenizer.encode_batch(buffer, add_special_tokens=False)
                    for e in encs:
                        ids = np.asarray(e.ids, dtype=np.int32)
                        out.write(ids.tobytes())
                        total_tokens += len(ids)
                    buffer.clear()
                    if consumed_bytes - last_report >= 100 * 1024 * 1024:
                        pct = 100 * consumed_bytes / src_bytes
                        print(f"  ... {pct:5.1f}%  {total_tokens:,} tokens")
                        last_report = consumed_bytes

            if buffer:
                encs = tokenizer.encode_batch(buffer, add_special_tokens=False)
                for e in encs:
                    ids = np.asarray(e.ids, dtype=np.int32)
                    out.write(ids.tobytes())
                    total_tokens += len(ids)

        out.seek(0)
        out.write(struct.pack("<I", total_tokens))

    mb = (4 + total_tokens * 4) / (1024 * 1024)
    print(f"  wrote {dst_path}  ({total_tokens:,} tokens, {mb:.1f} MB)")
    return total_tokens


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not TOKENIZER_PATH.exists():
        raise SystemExit(f"missing {TOKENIZER_PATH}; run tools/smollm_dump.py first")

    print("Loading tokenizer ...")
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    print(f"  vocab size: {tokenizer.get_vocab_size():,}")

    print(f"Downloading {TRAIN_FILE} ...")
    train_path = Path(hf_hub_download(repo_id=REPO_ID, filename=TRAIN_FILE, repo_type="dataset"))
    print(f"  -> {train_path}")
    print(f"Downloading {VALID_FILE} ...")
    valid_path = Path(hf_hub_download(repo_id=REPO_ID, filename=VALID_FILE, repo_type="dataset"))
    print(f"  -> {valid_path}")

    print(f"Encoding {VALID_FILE} ...")
    valid_count = encode_file_streaming(tokenizer, valid_path, OUT_VALID)

    print(f"Encoding {TRAIN_FILE} ...")
    train_count = encode_file_streaming(tokenizer, train_path, OUT_TRAIN)

    total = train_count + valid_count
    print(f"Done. Total tokens: {total:,} ({train_count:,} train + {valid_count:,} valid)")


if __name__ == "__main__":
    main()
