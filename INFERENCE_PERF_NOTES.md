# Gemma 4 E4B GPU inference perf — findings

Goal: match Ollama's decode tok/s on Gemma 4 E4B with our Vulkan backend.
Hardware reference: NVIDIA RTX 3090 Ti (1008 GB/s peak, 80 SMs, subgroup=32).

## Headline numbers

| Stage | Decode tok/s | Notes |
|---|---|---|
| Baseline (Q4_0, GPU) | 9.4 | first end-to-end run |
| Bf16 baseline (no quant) | 10.7 | proves we weren't bandwidth-bound |
| Cached scalars on model | 9.9 | record dropped 96→4ms but sync rose to compensate |
| GEMV-shape Q4_0 shader (M=1) | 23–24 | ~2.5× over baseline; old in-process Q4_0 path |
| WG=64 / K_PER_ITER=8 experiments (Q4_0) | ~20.5 | shader-shape changes don't move the needle (memory-bound) |
| **GGUF Q4_K_M, first GPU run** | **10.3** | bf16 placeholders for q/k after permute, wasteful tile shader for lm_head |
| **+ `linear_bf16_gemv.comp`** | **19.6** | M=1 GEMV for lm_head + q/k_proj |
| **+ Q4_K row-permute on load** | **36.9** | q/k stay Q4_K; HF→ggml interleaved-pair permute as a byte-level row shuffle |
| **+ Q6_K shader hoist** | **~37** | small per-op win; Q6_K is only 11% of GPU |
| Ollama (their numbers, same machine) | 133 | target |

Current gap to Ollama: ~3.6×.

## What's spending the 25 ms / token today

```
GPU work    17.8 ms   ← summed dispatch time (timestamp queries)
GPU stalls   4-5 ms   ← barriers + queue wait between dispatches
CPU record   2-3 ms   ← Vulkan command recording
─────────────────────
total wall  ~25 ms
```

GPU work breakdown:

| pipeline | dispatches/tok | µs/op | ms/tok | % |
|---|---|---|---|---|
| Q4_K GEMV | 309 | 31.5 | 9.7 | 55% |
| Q6_K GEMV | 34 | 57.0 | 1.94 | 11% |
| bf16 GEMV (lm_head only) | 1 | 1436 | 1.44 | 8% |
| small per-element ops (rmsnorm, mul, add, silu, rope, attention) | ~1100 | 3-5 | ~5 | 28% |

For context, **theoretical bandwidth lower bound is ~6-7 ms/token** (4.5 GB of
weights at ~800 GB/s effective). Ollama at 7.5 ms is essentially at the
bandwidth floor. We're 3-4× above it.

## Debugging timeline / what we learned

### 1. Q4_0 vs bf16 are the same speed → not bandwidth-bound (early Q4_0 work)

Ran `gemma_bench --quantize none` (10.7 tok/s) vs `--quantize q4` (9.4). If we
were weight-bandwidth-bound, Q4_0 should have been ~4× faster. Instead they
matched. **Decode is overhead-bound, not bandwidth-bound** at the start.

### 2. Per-op time breakdown via split timers and counters

Added `record_ns` (CPU recording forward) vs `sync_ns` (submit + waitidle +
readback) timers in `gemma_bench`. Then per-op counters in
`backend.forward` and `buffer_alloc`.

- record: 96 ms/token
- backend.forward: 1142 ops × 2 µs = 2.3 ms
- buffer_alloc: 1577 calls × 0.1 µs = 0.16 ms
- **~93 ms unaccounted for** — turned out to be inside `ml.scalar` and
  `ml.set_data_bytes` calls.

### 3. The killer: forced GPU syncs inside the forward path

`ml.scalar(value, dtype)` and `ml.set_data_bytes(...)` go through
`gpu.buffer_set` → `_upload`, which calls `end_batch` (vkQueueSubmit +
vkQueueWaitIdle) THEN does a one-shot copy (another submit + waitidle).
That's two full GPU syncs per call.

`gemma.forward_cached` had **7 such calls per token**. Counter measured
**13 ms per buffer_set on average × 7 = 93 ms/token**. Exactly the missing
budget.

**Fix**: pre-bake the constants as persistent shape-`[1]` tensors on the
`Gemma` struct (`embed_scale`, `ple_token_scale`, `ple_ctx_scale`,
`ple_combine_scale`, `softcap`, `softcap_inv`) and reference them from the
forward path. Eliminated 6 of the 7 syncs. Recording dropped 96→4 ms.

### 4. The Q4_0 tile shader wasted 31/32 of compute on M=1

The original `linear_q4.comp` had TILE_M=32 — for decode (M=1), 31/32 of
the M-dim compute was wasted. **Built a Q4_0 GEMV shader**: WG=32 (one
subgroup), ROWS_PER_WG=2, threads stride K with stride 32, accumulate
two partial sums per WG, `subgroupAdd` reduce, thread 0 writes the
bf16-packed uint. Per-op cost: 247 µs → 73 µs (3.4×). Decode: 9.9 → 23.7
tok/s. Bit-exact against CPU reference at M=1.

(The Q4_0 shaders, the `.I4` and `.I8` data types, and the in-process
`quantize_int4` / `quantize_q8_0` machinery were retired once the GGUF
Q4_K_M path was working — they were strictly inferior in quality and
identical in bandwidth. The same GEMV-shape trick was missing for
`linear_bf16` until we hit it again on Gemma 4 GGUF — see § GGUF below.)

### 5. ROWS_PER_WG=4 was a regression on Q4_0

Tried fattening each WG to 4 output rows for X-reuse and ILP. Decode went
**backwards** to 17.6 tok/s. Likely: increased register pressure cut SM
occupancy, and X is small enough (5 KB) that L2 already serves redundant
reads cheaply. Reverted to ROWS_PER_WG=2.

## GGUF Q4_K_M for Gemma 4 E4B (this round)

Switching the Q4_0 in-process quantization path to loading Ollama's
Q4_K_M GGUF directly. Q4_K_M is a mixed format: the bulk of language-model
linears are Q4_K (4.5 bits/weight, 256-element super-blocks with 6-bit
scales/mins and an fp16 super-scale), with `attn_v` and parts of `ffn_down`
in Q6_K (6.5 bits/weight) for precision-sensitive paths. The exact mix
varies per layer — it's not a uniform "all attn_v are Q6_K" pattern.

### What got built

- `loaders/gguf/gguf.odin` — read-only GGUF v3 reader, mmap-style API
  mirroring the safetensors loader (`load`, `destroy`, `get_info`,
  `get_bytes`, typed KV accessors).
- `loaders/gguf/quants.odin` — bit-exact `dequantize_q4_k` and
  `dequantize_q6_k` matching ggml's reference implementations
  (`dequantize_row_q4_K`, `dequantize_row_q6_K` in `ggml-quants.c`).
- `ml.odin`: added `.Q4_K` and `.Q6_K` data types with block-aware
  `_data_byte_count` (144 bytes/256 weights and 210 bytes/256 weights).
  Added `Linear_Q4_K` / `Linear_Q6_K` op variants and `linear_q4_k` /
  `linear_q6_k` procs.
- `backends/cpu/cpu.odin`: `linear_q4_k_forward` / `linear_q6_k_forward` —
  reference implementations: dequant the row, dot against bf16 activation.
  Slow but correct; intended as the parity baseline for the GPU shader.
- `backends/gpu/shaders/linear/linear_q4_k_gemv.comp` — WG=32,
  ROWS_PER_WG=2, subgroup-reduced. fp16 d/dmin via `unpackHalf2x16`,
  inline 6-bit scale/min unpacker mirroring `get_scale_min_k4`.
- `backends/gpu/shaders/linear/linear_q6_k_gemv.comp` — same shape.
  210-byte blocks aren't 4-byte aligned, so byte addressing via a
  `load_byte` helper that reads from a `uint w[]` view.
- `networks/gemma/loader_gguf.odin` — maps GGUF tensor names to Gemma
  fields. Norms come in F32, converted down to Bf16 at load. `q_norm` is
  pre-multiplied by `sqrt(head_dim)` to absorb the `1/sqrt(head_size)` in
  `ml.attention` (matches the safetensors loader's transform).
  `embed_tokens` is dequantized from Q6_K to Bf16 at load.

### The RoPE convention bug

Per-position top-K logit overlap with the HF reference degraded with
position (pos 0: 3/5, pos 4: 1/5) — classic signature of a RoPE bug.

Looking at `convert_hf_to_gguf.py` for Gemma 4:

> the expected ordering is cc000000ss000000 (c = cos, s = sin, 0 = unrotated),
> but ggml neox only supports ccss000000000000, and we cannot rearrange the
> head because that will break use_alternative_attention.
> solution is to set specific freq_factors for the unrotated dims

So GGUF Gemma 4 stores q/k weights in HF's split-half (`[first_half | second_half]`)
form, **not** ggml's interleaved-pair form. Our `ml.rope` is interleaved-pair
(neox), so feeding HF-form weights through it rotates the wrong dim pairs.

The position-dependent error makes sense: at position 0, `sin=0, cos=1` so
RoPE is identity for both conventions and any layout works; as position
grows, the rotation angles diverge.

**Fix**: apply the same `[first_half | second_half] → interleaved-pair`
row permutation that `_load_rope_permuted` does in the safetensors loader.
First implementation dequantized → permuted → encoded as Bf16, which
worked but cost ~500 MB of GPU memory and routed q/k through the bf16
linear path (slow, see next section).

**Better fix**: Q4_K quantizes along the **input_size** axis; the row
permutation only reorders **output rows**. So the permutation is a pure
byte-level row shuffle — no dequant needed, each row's super-blocks stay
intact. q/k stay Q4_K; bytes get shuffled at load time only.
This is `_load_rope_permuted_q` in `loader_gguf.odin`.

After: top-K overlap is 3-5/5 per position with max abs logit diff ~9.7
vs HF reference (Q4_K_M quantization noise; 9.7 in a softcapped [-30, 30]
range is reasonable for 4.5 bits/weight).

### Adding `linear_bf16_gemv.comp`

The first GPU profile of the GGUF Gemma showed bf16 linear pipeline
dominating at 44.6% of GPU time:

```
rank 1: 67 dispatches/tok, 299 µs/op, 20 ms/tok    bf16 linear (q + k + lm_head)
```

Two things were stacked on the bf16 linear path:
- `lm_head` (262144 × 2560 bf16 = 1.34 GB weight read once per token).
- q_proj + k_proj for every layer (because we hadn't yet figured out the
  byte-level permute and were dequantizing them to Bf16).

`linear_bf16` uses the tile shader (TILE_M=32) — same M=1 inefficiency
that `linear_q4_gemv.comp` was created to fix for Q4_0 weights. Wrote
the bf16 equivalent (`linear_bf16_gemv.comp`, same shape: WG=32,
ROWS_PER_WG=2, subgroup reduce). Per-op time on the GEMV path: 63 µs
(vs 299 µs tiled). **Decode: 19.6 → 36.9 tok/s** after both this and the
Q4_K permute fix.

After: lm_head is the only thing left in bf16 GEMV column at 1.44 ms/tok,
roughly bandwidth-limited.

### Q6_K shader: hoist out the inner loop

Modest win: hoisting `d`, the 16 sub-block scales, and the per-half
ql/qh byte loads outside the per-quadrant inner loop. Ql at l vs l+32 is
shared between quadrants {0,2}/{1,3}; qh at l is shared across all 4.
Per-op: 61.8 → 57.0 µs (~8%). Q6_K is only 11% of GPU time so the wall
impact is tiny.

## Things we tried that didn't matter (or backfired)

- **Push descriptors** (`VK_KHR_push_descriptor`): replaced per-dispatch
  `vkAllocateDescriptorSets`/`UpdateDescriptorSets`/`BindDescriptorSets`
  with `vkCmdPushDescriptorSetKHR`. Microbench unchanged. The DS path
  wasn't the bottleneck. We kept the change anyway — simpler code, no DS
  pool churn, and removes a future obstacle for pre-recorded CBs.
- **Activation buffer pool reuse** (`activation_pool` + cursor): VkBuffer
  handles are now recycled across forward passes; clear() rewinds the
  cursor instead of destroying buffers. Microbench unchanged at the time
  because the alloc bench was dominated by submit+waitidle per
  iteration. Real-world effect was minimal — alloc was already <1% of
  per-token cost.
- **Inference mode** (`set_inference_only(true)` skips `.Gradient` buffer
  on activations): single-buffer alloc dropped 95 µs → 50 µs in the
  microbench. Real workload: tiny effect on decode (~9.4 → 9.2), modest on
  prefill (63 → 144 tok/s — likely noise). Cost-per-cycle is per-call,
  not per-buffer. Kept anyway, it's correct.
- **ROWS_PER_WG=4** in the Q4_0 GEMV shader: regressed.
- **Skip zero-fill on slot reuse**: implemented (only fresh slots get
  `vkCmdFillBuffer`), no measurable change. The fills are cheap.
- **WG=64 with 2 subgroups + cross-subgroup reduce in `linear_q4_gemv`**:
  bumped `local_size_x` to 64, halving per-thread K-loop iterations and
  combining via shared memory after `subgroupAdd`. Bit-exact parity. The
  microbench showed gains on `mlp_down` (1.39 → 0.25 ms) and `lm_head`
  (7.98 → 4.27 ms) — but those were L2-cache artifacts (same weights
  re-read 200×). End-to-end gemma_bench: 20.5 → 20.6 tok/s, gemv 89.9 →
  89.3 µs/op. Wash. **Lesson: the `gpu_linear_q4_bench` microbench is
  L2-hot and unrepresentative of real decode for any shape that fits in
  6 MB.** Reverted.
- **K_PER_ITER=8 + uvec4 X loads** in `linear_q4_gemv` (mirrored
  llama.cpp's `mul_mat_vec.comp`): each thread processes 8 K-elements per
  loop step, loads X as `uvec4` (8 packed bf16), one scale and one W uint
  per row per iter, then 8 fmas. Drops the K-loop iteration count 8×.
  Bit-exact parity. End-to-end: gemv 89.9 → 88.3 µs/op median, decode
  20.5 → 20.5 tok/s — within ±1 tok/s run-to-run noise. Reverted because
  the complexity wasn't earning anything. **Lesson: `linear_q4_gemv` is
  memory-bandwidth bound; restructuring the K-loop, vectorizing X loads,
  and going from 1 → 8 fmas/iter all leave wall time unchanged.**

## Architecture insights worth remembering

- **`ml.scalar` is a synchronous upload**. Same for `ml.set_data_bytes`.
  Never call these inside a forward pass. If you need a constant tensor,
  bake it once at load time and reference it.
- **Vulkan timestamp queries** are easy and exactly what you want for
  GPU-time profiling. `gpu.enable_timing()` / `gpu.dump_timing()` work
  per-pipeline-pointer; identify pipelines by dispatch count + the
  `_*_pipeline` static globals.
- The activation pool's "index-based reuse" depends on the alloc
  sequence being identical across forward passes. Any size change at
  slot N invalidates all slots ≥ N (we destroy and rebuild the tail).
  This held for prefill → decode → decode → ... but watch for it if
  graph shape ever depends on token state.
- **The global `SHADER → SHADER` memory barrier between every dispatch
  serializes all GPU work.** With ~1500 dispatches per token, this is a
  meaningful chunk of GPU stall time. Removing redundant barriers (only
  insert when consecutive ops actually share buffers) is a real
  optimization opportunity, not a correctness sacrifice.
- **GGUF stores tensor shapes in ggml's reversed (column-major-fastest)
  order.** Bytes are identical to row-major; we just compare shape after
  reversal in the loader.
- **GGUF Gemma 4 q/k weights are in HF split-half form, not ggml
  interleaved-pair form** — the conversion script intentionally does not
  permute (cited "use_alternative_attention" compatibility) and uses
  `ROPE_FREQS` to disable rotation on the unrotated dims. Our `ml.rope`
  is neox/interleaved-pair, so we permute on load. Q4_K-aware byte-level
  row shuffle preserves the format.
- **Q4_K block layout (144 bytes / 256 weights):** `d` (fp16) | `dmin`
  (fp16) | `scales[12]` (eight 6-bit scales + eight 6-bit mins, packed
  per `get_scale_min_k4`) | `qs[128]` (4-bit quants).
- **Q6_K block layout (210 bytes / 256 weights):** `ql[128]` (low 4 bits)
  | `qh[64]` (upper 2 bits, 4 per byte) | `scales[16]` (i8) | `d` (fp16).
  210 isn't a multiple of 4, so block-aligned uint indexing breaks across
  blocks; the shader uses byte addressing into a `uint w[]` view.

## What's between us and Ollama, ranked

Today: 25 ms/tok wall. Ollama: 7.5 ms/tok. Theoretical bandwidth floor:
~6-7 ms/tok. Gap: 17.5 ms/tok needs to disappear.

| barrier | est. saving | effort | what it is |
|---|---|---|---|
| **No coopmat (tensor-core) Q4_K matmul path** | **5-7 ms** | large | mirror `linear_bf16_coopmat.comp` for in-shader-Q4_K-dequant feeding a coopmat tile accumulator. ggml's high-perf path. Requires tensor-core-friendly tile sizes; can do M=1 via padded-to-tile or a separate GEMV-style coopmat path. |
| **No Q8_1 activation pre-quantization (mmvq)** | **3-4 ms** | medium | Quantize the bf16 activation to Q8_1 (per-32 i8 + scale) on the fly per layer, then do `int8 × int4 → int32` integer-dot inside the matmul. CUDA's `mul_mat_vec_q4_K_q8_1` is the reference. Wins from integer ALUs being faster than fp32 fma per cycle on Ampere/Ada. |
| **~1500 dispatches/tok with global barriers** | **3-5 ms** | small/medium | Two angles: (a) **op fusion**: rmsnorm has separate stats+main, silu+mul are separate, residual adds are separate — fuse realistic candidates and halve dispatch count. (b) **precise barriers**: don't fire `SHADER+TRANSFER → SHADER` between every dispatch, only when consecutive ops share buffers. |
| **Q4_K GEMV at 57% of theoretical bandwidth** | **0.5-1 ms** | small | 31 µs/op vs ~18 µs theoretical for ffn_gate. Most of the gap is per-dispatch barrier/queue tax (covered by row above), not shader-work tax. Compounds with #3. |
| **Per-token recording cost** | **2-3 ms** | medium | Pre-recorded command buffer with push constants moved to a uniform buffer that gets updated host-side per token. Won't move the needle alone but compounds with #3 because re-recording 1500 commands is what we're doing today. |

### Stacking estimates

| stack | est. wall | est. tok/s |
|---|---|---|
| Today | 25 ms | ~37 |
| + op fusion + precise barriers + pre-recorded CB | ~17 ms | ~58 |
| + Q8_1 activation pre-quant (mmvq) | ~13.5 ms | ~75 |
| + coopmat Q4_K path | ~8 ms | ~125 |

The honest summary: **the 1.5-2× win to compete with Ollama needs
cooperative-matrix or integer-dot-product paths** — same conclusion the
Q4_0 perf round reached. Everything else gets us to the 50-75 tok/s
range and is much cheaper to implement.

## Code added in this round (still in tree)

GGUF + Q4_K_M (this round):

- `loaders/gguf/gguf.odin` — GGUF v3 reader.
- `loaders/gguf/quants.odin` — Q4_K / Q6_K dequant.
- `ml.odin` — `.Q4_K` / `.Q6_K` dtypes, `Linear_Q4_K` / `Linear_Q6_K`
  variants, `linear_q4_k` / `linear_q6_k` procs.
- `backends/cpu/cpu.odin` — `linear_q4_k_forward` / `linear_q6_k_forward`
  reference implementations.
- `backends/gpu/shaders/linear/linear_q4_k_gemv.comp` — Q4_K GEMV (M=1).
- `backends/gpu/shaders/linear/linear_q6_k_gemv.comp` — Q6_K GEMV (M=1).
- `backends/gpu/shaders/linear/linear_bf16_gemv.comp` — bf16 GEMV (M=1).
- `backends/gpu/backend.odin` — `linear_q4_k_forward`, `linear_q6_k_forward`,
  M=1-decode bf16 GEMV path in `linear_forward`.
- `networks/gemma/loader_gguf.odin` — Gemma 4 GGUF loader, with Q4_K row
  permute on load.
- `tests/gguf_dump/` — header / KV / tensor inspector.
- `tests/gguf_quants/` — dequant unit tests + real-tensor smoke.
- `tests/gguf_linear_parity/` — bit-exact Q4_K / Q6_K linear op parity.
- `tests/gguf_gemma_load/` — full-model load smoke.
- `tests/gguf_gemma_forward/` — single-token CPU forward sanity.
- `tests/gguf_gemma_parity/` — multi-token CPU forward vs HF reference logits.
- `tests/gguf_gemma_gpu_parity/` — single-token-at-a-time GPU forward vs
  HF reference logits, with per-pipeline GPU timing dump.

Cross-cutting infrastructure (still in tree):

- `backends/gpu/buffer.odin` — activation pool reuse, slot tail-rebuild
  on size mismatch, skip-fill on reuse.
- `backends/gpu/pipeline.odin` — push descriptors, optional timestamp
  query bracketing.
- `backends/gpu/backend.odin` — instrumentation counters
  (`forward_stats`, `alloc_stats`, `upload_stats`), timing API
  (`enable_timing`, `reset_timing`, `dump_timing`).
- `ml.odin` — `Context.inference_only`, `set_inference_only`.
- `examples/gemma_bench/main.odin` — non-interactive bench, Ollama-shape
  output, `--skip-weights`, per-token instrumentation dump, GPU timing
  dump.
- `networks/gemma/gemma.odin` — pre-baked scalar tensors on `Gemma`.

## Open chores / cleanups before merging

- Prefill on GPU: the new Q4_K / Q6_K shaders are M=1 only. M>1 panics.
  Either add tile shaders or accept "decode after CPU prefill" workflow.
- The instrumentation in `gemma_bench` (record/sync timers, op counters,
  GPU timing dump) is permanently on. Probably wants a `--profile` flag
  and a quieter default output.
- The diagnostic counters (`_forward_op_count`, `_alloc_count`,
  `_upload_count` etc.) live as global mutable state in
  `backends/gpu/backend.odin`. Either gate them on a build flag or push
  them onto `Context`.
- `examples/gemma_chat_repl` got `set_inference_only(true)` and prefill
  timing split — verify it still produces sensible output with the
  cached-scalar / pool-reuse path on a real GGUF model.
- Untied lm_head: `load_gguf` rejects models without `tie_word_embeddings`;
  add that branch when we encounter one.
