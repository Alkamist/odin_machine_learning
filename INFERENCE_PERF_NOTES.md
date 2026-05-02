# Gemma 4 E4B GPU inference perf — findings

Goal: match Ollama's decode tok/s on Gemma 4 E4B with our Vulkan backend.
Hardware reference: NVIDIA RTX 3090 Ti (1008 GB/s peak, 80 SMs, subgroup=32).

## Headline numbers

| Stage | Decode tok/s | Notes |
|---|---|---|
| Baseline (Q4_0, GPU) | 9.4 | first end-to-end run |
| Bf16 baseline (no quant) | 10.7 | proves we weren't bandwidth-bound |
| Cached scalars on model | 9.9 | record dropped 96→4ms but sync rose to compensate |
| GEMV-shape Q4 shader (M=1) | **23–24** | current state, ~2.5× over baseline |
| Ollama (their numbers, same machine) | 133 | target |

Current gap to Ollama: ~5.6×.

## Debugging timeline / what we learned

### 1. Q4 vs bf16 are the same speed → not bandwidth-bound

Ran `gemma_bench --quantize none` (10.7 tok/s) vs `--quantize q4` (9.4). If we
were weight-bandwidth-bound, Q4 should have been ~4× faster. Instead they
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

`gemma.forward_cached` had **7 such calls per token**:

| Site | Constant |
|---|---|
| `forward_cached:529` | `sqrt(hidden_size)` |
| `_per_layer_inputs:426` | `set_data_bytes` for token-id embedding lookup |
| `_per_layer_inputs:427` | `sqrt(ple_dim)` |
| `_per_layer_inputs:431` | `1/sqrt(hidden_size)` |
| `_per_layer_inputs:441` | `1/sqrt(2)` |
| `forward_cached:607` | `1/softcap` (if softcapping) |
| `forward_cached:609` | `softcap` (if softcapping) |

Counter measured **13 ms per buffer_set on average × 7 = 93 ms/token**.
Exactly the missing budget.

**Fix**: pre-bake the constants as persistent shape-`[1]` tensors on the
`Gemma` struct (`embed_scale`, `ple_token_scale`, `ple_ctx_scale`,
`ple_combine_scale`, `softcap`, `softcap_inv`) and reference them from the
forward path. Eliminated 6 of the 7 syncs. Recording dropped 96→4 ms.

The remaining `set_data_bytes` is the per-layer-input embedding lookup
(`embed_tokens_per_layer_bytes` is host-side, 5.6 GB at bf16 — exceeds
Vulkan's `maxStorageBufferRange`). Fixable but needs splitting the table
into per-layer chunks.

### 4. After eliminating syncs, GPU work itself was 96 ms

With recording at 4 ms, total per-token was still ~100 ms — the work just
moved into one big submit at the end. Confirmed via Vulkan timestamp
queries (`VK_QUERY_TYPE_TIMESTAMP`, written before/after every
`CmdDispatch`, results aggregated per-pipeline in `dump_timing`).

The dump showed **one pipeline = 93% of GPU time**: `linear_q4`, 354
dispatches per token at 247 µs each = 87 ms. Identified it by dispatch
count matching the number of linear projections in the model (q+k+v+o +
gate+up+down + ple_gate+ple_proj per layer × 42 + lm_head + per_layer_proj).

### 5. The Q4 tile shader wastes 31/32 of compute on M=1

`linear_q4.comp` has TILE_M=32 — for decode (M=1), 31/32 of the M-dim
compute is wasted. **Built `linear_q4_gemv.comp`**:

- WG = 32 (one NVIDIA subgroup)
- Each WG handles 2 adjacent output rows (so the bf16-packed
  `(hi << 16) | lo` write of 2 cells per uint never races neighbors)
- Threads stride K with stride 32, accumulate two partial sums
- `subgroupAdd` reduce at the end
- Thread 0 writes the packed uint
- Falls through to the original tile shader when M > 1

Per-op cost: 247 µs → 73 µs (3.4×). Decode: 9.9 → 23.7 tok/s.
**Bit-exact against CPU reference** at M=1.

### 6. ROWS_PER_WG=4 was a regression

Tried fattening each WG to 4 output rows for X-reuse and ILP. Decode went
**backwards** to 17.6 tok/s. Likely: increased register pressure cut SM
occupancy, and X is small enough (5 KB) that L2 already serves redundant
reads cheaply. Reverted to ROWS_PER_WG=2.

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
  microbench. In real workload: tiny effect on decode (~9.4 → 9.2),
  modest on prefill (63 → 144 tok/s — likely noise). Cost-per-cycle is
  per-call, not per-buffer. Kept anyway, it's correct.
- **ROWS_PER_WG=4** in the GEMV shader: see above, regressed.
- **Skip zero-fill on slot reuse**: implemented (only fresh slots get
  `vkCmdFillBuffer`), no measurable change. The fills are cheap.

## Architecture insights worth remembering

- **`ml.scalar` is a synchronous upload**. Same for `ml.set_data_bytes`.
  Never call these inside a forward pass. If you need a constant tensor,
  bake it once at load time and reference it.
- **Vulkan timestamp queries** are easy and exactly what you want for
  GPU-time profiling. `gpu.enable_timing()` / `gpu.dump_timing()` work
  per-pipeline-pointer; identify pipelines by dispatch count.
- The activation pool's "index-based reuse" depends on the alloc
  sequence being identical across forward passes. Any size change at
  slot N invalidates all slots ≥ N (we destroy and rebuild the tail).
  This held for prefill → decode → decode → ... but watch for it if
  graph shape ever depends on token state.
- The global `SHADER → SHADER` memory barrier between every dispatch
  serializes all GPU work. We didn't touch it — it's correctness
  insurance — but it's why the GPU can't pipeline independent dispatches.
  Future optimization opportunity.

## Remaining attack surface (rough ROI estimates)

The decode bottleneck is now `linear_q4_gemv` at ~22 ms/token (about 65%
of decode time). Per-op average 73 µs, theoretical floor for the weight
bandwidth ~3 ms total (10× headroom).

| Move | Est. multiplier | Effort | Notes |
|---|---|---|---|
| Larger WG (64 = 2 subgroups), inter-subgroup reduction | 1.1–1.3× | Small | Helps small-N ops underutilizing SMs |
| Vectorized weight loads (`uvec4`) | 1.1–1.2× | Small | Better memory throughput |
| Tiled GEMV with shared-mem X | 1.2–1.5× | Medium | Biggest win on `mlp_*`, the heaviest weights |
| Fused RMSNorm + linear_q4_gemv | 1.05–1.1× | Medium | Cuts dispatch count, doesn't help GPU work much |
| QKV fusion (3 linears → 1 dispatch) | 1.1–1.2× | Medium | Saves dispatches + shares X |
| FlashAttention-style fused attention | 1.1–1.2× | Medium-large | Pipeline 2 was attention_with_cache (2 ms/token); modest in our profile |
| Cooperative-matrix Q4 path | 1.5–2× | Large | Needs subgroup matrix multiply on Q4-dequantized tiles |
| Pre-recorded command buffer | 1.05–1.1× | Small | Saves recording (~4 ms); useful if we can overlap CPU & GPU |
| Eliminate the last `set_data_bytes` (per-layer-input upload) | 1.01× | Medium | Needs splitting the 5.6 GB table |
| Q4_K_M format | accuracy, not speed | Medium | Match Ollama's quantization for fair quality |

Stacking 3–4 of the above could plausibly land 60–80 tok/s.
Closing the rest of the gap to 133 likely requires cooperative-matrix.

## Code added in this round (still in tree)

- `backends/gpu/shaders/linear/linear_q4.comp` — the original tile shader
  (used for M>1; prefill).
- `backends/gpu/shaders/linear/linear_q4_gemv.comp` — M=1 GEMV.
- `backends/gpu/buffer.odin` — activation pool reuse, slot tail-rebuild
  on size mismatch, skip-fill on reuse.
- `backends/gpu/pipeline.odin` — push descriptors, optional timestamp
  query bracketing.
- `backends/gpu/backend.odin` — instrumentation counters
  (`forward_stats`, `alloc_stats`, `upload_stats`), timing API
  (`enable_timing`, `reset_timing`, `dump_timing`), `linear_q4_forward`
  M=1 fast path.
- `ml.odin` — `Context.inference_only`, `set_inference_only`.
- `networks/gemma/gemma.odin` — pre-baked scalar tensors on `Gemma`,
  `quantize_for_inference_fake` (for `--skip-weights` benches).
- `examples/gemma_bench/main.odin` — non-interactive bench, Ollama-shape
  output, `--skip-weights`, per-token instrumentation dump, GPU timing
  dump.
- `tests/gpu_linear_q4/main.odin` — CPU↔GPU parity for `linear_q4`.
- `tests/gpu_linear_q4_bench/main.odin` — per-shape isolated GEMV bench.
- `tests/gpu_alloc_bench/main.odin` — alloc/clear-cycle microbench.

## Open chores / cleanups before merging

- The instrumentation in `gemma_bench` (record/sync timers, op counters,
  GPU timing dump) is permanently on. Probably wants a `--profile` flag
  and a quieter default output.
- The diagnostic counters (`_forward_op_count`, `_alloc_count`,
  `_upload_count` etc.) live as global mutable state in
  `backends/gpu/backend.odin`. Either gate them on a build flag or push
  them onto `Context`.
- `examples/gemma_chat_repl` got `set_inference_only(true)` and prefill
  timing split — verify it still produces sensible output with the
  cached-scalar / pool-reuse path on a real model.
- Consider whether to commit `gemma.quantize_for_inference_fake` as a
  permanent helper or leave it as a test-only path.
