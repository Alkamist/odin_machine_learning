# CUDA backend decode-perf notes

Snapshot of where decode throughput stands and what to try next on
`gemma_chat_repl`. Pick this up by re-reading this file, then the diff
in the files listed under "What's in tree".

## Numbers

Hardware: NVIDIA GeForce RTX 3090 Ti, cc=8.6, 84 SMs.
Model: Gemma 4 E4B, Q4_K_M GGUF.
Test: `gemma_chat_repl --gguf gemma_data/model.gguf --max-tokens 128`,
"write me a story" prompt, 12-token prefill.

Headline (no `--timing`):

| Stage                                            | tok/s |
|--------------------------------------------------|-------|
| Initial baseline (start of project session)      | 27.0  |
| + CUDA graph capture (auto, transparent)         | 38.8  |
| + Q4_K mmvq 4-warp                               | 41.2  |
| + Q4_K q8_1-input cache (`q8_1_cache`)           | 43.0  |
| + Q6_K gemv 4-warp + load hoist                  | (rolled in) |
| + attention K-vector load (uint4)                | 44.2  |
| + fused q4_k gate+up+GEGLU mmvq                  | 45.8  |

Reference target (ollama / llama.cpp on the same model+hw, per user):
~130 tok/s.

`--timing` itself slows the run because `cuEventRecord` fires twice
per dispatch. Quote tok/s headlines from the no-`--timing` run; use
`--timing` only for the kernel breakdown.

## Where the time goes now

`--timing` snapshot at the 45.8 tok/s headline (timing-on figures —
absolute ms shifts vs no-timing, but shares are representative):

| Kernel                              | share | notes |
|-------------------------------------|-------|-------|
| linear_q4_k_mmvq                    | ~18%  | q/k/v/o + down_proj |
| linear_q4_k_gate_up_geglu_bf16      | ~14%  | fused FFN front-half (gate+up+GEGLU) |
| linear_q6_k_gemv                    | ~10%  | (subset of weights are Q6_K) |
| rmsnorm_bf16                        | ~17%  | rmsnorm + weight mul (already fused) |
| attention_cache_bf16                | ~13%  | inc. K/V memcpy + flash-attn-2 inner loop |
| quantize_q8_1_bf16                  | ~9%   | bf16 → q8_1, served from `q8_1_cache` on reuse |
| gelu_mul_bf16                       | ~6%   | PLE residual block only (FFN path is fused) |
| rmsnorm_rope_bf16                   | ~5%   | q/k norm + rope, fused |
| add_bf16 / mul_bf16 / add_rmsnorm   | ~6%   | per-layer residuals + scalar mul + pre_ff fused |
| (rest)                              | ~2%   | slice_trailing, select |

`gpu/wall ≈ 38%` at decode with `--timing` on; without timing the ratio
shifts toward GPU but wall is still well above GPU. **CPU is the
binding constraint** — see Key Findings.

## What's in tree

All under `backends/cuda/` plus the per-call sites in
`networks/gemma/gemma.odin`, with the new fused-op machinery in
`ml.odin`.

### Auto CUDA-graph capture
- `bindings/cuda/cuda.odin` — `cuGraphExecUpdate_v2` binding +
  `GraphExecUpdateResult{Info}` types.
- `cuda.odin` — auto-graph fields on `Context`
  (`auto_graph_enabled`, `auto_capturing`, `auto_warmup_done`,
  `auto_exec`). `clear()` finishes any in-flight capture and starts
  the next one. First forward after enable runs direct so cuBLAS can
  do its first-call algo selection (not capturable cold). Public
  `enable_decode_graph(bool)`, mutually exclusive with
  `enable_timing`.
- `graph.odin` — `_auto_graph_finish`: end stream capture, attempt
  `cuGraphExecUpdate`; if topology changed, re-instantiate. Topology
  change is expected on prefill-chunk → decode-chunk transition.
- `buffer.odin` — `buffer_get` calls `_auto_graph_finish` before
  `StreamSynchronize` + `MemcpyDtoH`, so logits are valid.
  `buffer_set` skips its post-copy `StreamSynchronize` while
  capturing (illegal during capture); in-tree callers all use
  `temp_allocator` buffers that outlive the forward.

### q8_1 input reuse
- `Context.q8_1_cache: map[DevicePtr]DevicePtr`, cleared in
  `clear()`, deleted in `context_destroy`. `_linear_q4_k_forward` (and
  the fused gate+up kernel) consult it keyed by input device pointer
  and emit `quantize_q8_1` only on cache miss. Halves quantize
  dispatches when q/k/v share the input_norm output and gate/up share
  the pre_ff_norm output.

### Quant matmul kernels (M=1 decode)
- `kernels/linear/linear_q4_k_mmvq.cu` — 4 warps × 32 lanes,
  `ROWS_PER_WG=2`. Each warp covers 1/NWARPS of the K-blocks;
  cross-warp reduction via shared memory. 26.4 → 20.5 µs / call.
- `kernels/linear/linear_q4_k_gate_up_geglu_bf16.cu` (this session) —
  fused gate + up + GEGLU. One mmvq block accumulates two partial
  sums (gate, up) using the same q8_1 input loads, and the final
  store writes `gelu_tanh(gate) * up`. Mirrors ggml's
  `ffn_up + ffn_gate + glu` fusion (`ggml_cuda_should_fuse_mul_mat`
  in `ggml-cuda.cu`). Per-call ≈ 53 µs (vs 2 × 20.5 µs ≈ 41 µs for
  the unfused pair, but eliminates the gate/up bf16 writeback and
  the `gelu_mul_bf16` dispatch — net win).
- `kernels/linear/linear_q6_k_gemv.cu` — same NWARPS=4 treatment +
  hoisting bf16 input loads out of the row × quadrant inner loops
  (was reloaded per row × 4 quadrants). 74.4 → 49.5 µs / call.

### Attention
- `kernels/attention/attention_cache_bf16.cu` — vectorized inner K
  dot loop to `uint4` loads (8 bf16 / load). 77.9 → 55.0 µs.

### Fused-op opt-in mechanism (new this session)
- `ml.odin` — added `Backend_Capability` enum +
  `Backend_Capabilities` bit_set on the `Backend` struct. Procs that
  emit fused ops check the capability and decompose into primitives
  when absent, so backends without a kernel for the fusion still work
  unchanged. `Linear_Q4_K_Gate_Up_Geglu` is the first such cap.
- `backends/cuda/cuda.odin` — sets `capabilities = { .Linear_Q4_K_Gate_Up_Geglu }`.
- `backends/cpu/cpu.odin` and (transitively) Vulkan — no capability
  set, so `linear_q4_k_gate_up_geglu` decomposes into the legacy
  `gate / up / gelu_mul` sequence at op-emission time.
- `networks/gemma/gemma.odin` — added `_gate_up_geglu(x, w_gate, w_up)`
  helper; both the prefill and decode forward paths route through it.

### Timing instrumentation
- `Context.timing_enabled / timing_totals / timing_pool /
  timing_cursor` + `_acquire_timing_slot` in `pipeline.odin`. Per-
  dispatch start/end events folded into per-pipeline totals at
  `clear()` time. Public `enable_timing(bool)`, `timing_snapshot()`,
  `reset_timing()`.
- `examples/gemma_chat_repl/main.odin` — `--timing` flag prints the
  per-kernel breakdown after each generation. The `gpu.enable_decode_graph(true)`
  is set when not in timing mode (mutually exclusive paths).

## ggml verification (already done — don't redo)

Cross-checked the proposed fusion menu against
`llama.cpp/ggml/src/ggml-cuda/`:

- **ggml DOES** fuse `rms_norm + mul (+ optional add)` —
  `ggml_cuda_op_rms_norm_fused` / `_fused_add` in `norm.cu`. The mul
  is the rmsnorm weight; the add is the residual. One kernel.
- **ggml DOES** fuse `mul_mat (ffn_up) + mul_mat (ffn_gate) + GLU`
  (swiglu/geglu) into a single mmvq —
  `ggml_cuda_should_fuse_mul_mat` in `ggml-cuda.cu` (~line 2227),
  dispatched in the fusion block (~line 3471). Both matmuls +
  activation in one kernel; gate/up intermediates never written.
  *We have this for Q4_K decode now.*
- **ggml DOES** fuse `rope + set_rows` (K cache write).
- **ggml DOES NOT** fuse `rms_norm + quantize_q8_1`. Their MMQ
  pipeline keeps `quantize_mmq_q8_1` as its own kernel.
- **ggml DOES NOT** fuse `rms_norm + mul_mat`.
- **CUDA graph**: ggml uses the same `cuGraphExecUpdate`-on-
  unchanged-topology pattern we already implement. There is no
  separate "compiled decoder" API on their side.

## Key findings

1. **Auto-graph capture (`cuGraphExecUpdate` per-step) was the single
   biggest win** — 27 → 38.8 tok/s. Confirmed launch-overhead-bound
   diagnosis; gpu/wall was ~52% pre-graph.

2. **Per-kernel polish has diminishing returns at M=1.** The Q4_K /
   Q6_K mmvq kernels are bandwidth-bound, not compute-bound, so
   adding threads (4 warps × 1 vs 1 warp × 2) yielded ~25–30%, not
   the 2–3× ggml gets from the same architectural change. Reductions
   (rmsnorm, gelu_mul, add) are *single-block-per-call* at decode and
   already at the small-kernel-launch floor — no per-kernel polish
   gets them faster.

3. **Wall ≫ GPU at decode — we're CPU-bound.** Without timing
   instrumentation, GPU work is ~14–17 ms/tok while wall is ~22 ms/tok
   at the 45.8 tok/s headline. The host-side cost is dominated by
   - `gemma.forward_cached` rebuilding the ml.Operation list every step,
   - `cuGraphExecUpdate` on each step (we re-capture every step),
   - per-step buffer pool walking.
   Kernel-side fusion is now mostly converting to lower gpu/wall ratio,
   not lower wall.

4. **Structural ceiling without ml-system rework: ~50–55 tok/s.**
   Even with infinitely fast kernels, the per-step host traversal in
   `forward_cached` (calling ~70 ml ops per layer × 42 layers ≈ 3000
   ml-op dispatches, then the cuGraph re-capture / update) puts a
   floor on wall time. ggml/llama.cpp escape this by replaying a
   pre-built compute graph with parameter updates.

5. **`--timing` count column was misleading; us avg is correct.**
   Odin's `fmt_write_padding` defaults to `'0'` unless the explicit
   `' '` flag is set, so `%-5d` right-pads with `'0'` and a 4-digit
   count like `1054` printed as `10540`. Fixed in
   `examples/gemma_chat_repl/main.odin` by switching to `% 5.1f /
   % 7.1f / x% -7d`. The "10× display anomaly" recorded in earlier
   notes was this; per-call avg was always trustworthy.

## Things to NOT redo

- Don't combine `auto_graph` mode with `enable_timing` — asserted off
  in code; `cuEventRecord` is not stream-capturable.
- Don't add a periodic graph re-instantiation cycle;
  `cuGraphExecUpdate` already handles steady-state arg patching.
  Re-instantiation only fires on topology change (prefill-chunk →
  decode-chunk transition).
- Don't fold K/V cache writes into `attention_cache_bf16` without
  cooperative-groups grid sync. The kernel launches many blocks per
  head; a write at the start of one block is not guaranteed visible
  to another block's read within the same launch. Either add
  cooperative launch or keep the explicit memcpy and patch its dst
  if Path B lands.
- Don't skip the warmup forward before enabling capture. cuBLAS does
  first-call algo selection that isn't capturable cold; the
  `auto_warmup_done` flag in `Context` exists for exactly this.
- Don't pursue `rmsnorm + quantize_q8_1` or `rmsnorm + mul_mat`
  fusion. Verified: ggml does neither, despite being faster than us.
  The wins lie elsewhere.

## Next steps (ggml-aligned, ordered)

1. **Path B: compile-once / replay decoder.** This is now the
   biggest remaining lever — the GPU savings from kernel fusion
   already aren't moving wall, so reducing host-side per-step cost
   is what matters. Genuine multi-session work:
   - New cuda bindings: `cuStreamGetCaptureInfo_v2`,
     `cuGraphExecKernelNodeSetParams` (+ `CUDA_KERNEL_NODE_PARAMS`),
     `cuGraphExecMemcpyNodeSetParams` (+ `CUDA_MEMCPY3D`,
     ~24 size_t fields).
   - Cuda backend "compile-mode capture": every `_dispatch` of a
     position-bearing pipeline (rmsnorm_rope, rope, attention_cache)
     records the inserted graph node + a stable copy of its
     kernel-params buffer + the byte offset of `cache_position`
     inside it. Same for dynamic-source HtoD memcpys (select index
     upload, per-layer-input row upload) — captured graph must
     reference pinned host buffers.
   - Public `gemma.compile_decoder` runs (a) one warmup
     `forward_cached` direct-launch, (b) one capture forward in
     compile-mode, saves graph_exec + node-handle list +
     pinned-buffer pointers. `decode_step` writes new dynamic state
     to pinned buffers, calls `cuGraphExecKernelNodeSetParams` on
     every position node (~125 nodes) and
     `cuGraphExecMemcpyNodeSetParams` on K/V cache write nodes
     (~150 nodes), launches, syncs, copies logits.
   - Combined with the ggml-aligned kernel fusions below, this is
     what gets us into ggml's range. Alone, expected end point is
     ~50–55 tok/s.

2. **Fuse `rms_norm + add (residual)`.** ggml-aligned, mirrors
   `ggml_cuda_op_rms_norm_fused_add` (the trailing add of a residual
   onto the rmsnorm-with-weight output). In gemma decode, this folds
   the `residual = ml.add(residual, mlp)` after the post_ff rmsnorm
   and the equivalent after the post_per_layer rmsnorm. Saves ~half
   the `add_bf16` dispatches per layer — modest GPU win, but ggml
   does it. **Requires a new ml op + new capability bit + cpu/vulkan
   decompose paths**, same shape as `Linear_Q4_K_Gate_Up_Geglu`.
   We already added the capability mechanism, so the marginal cost
   is just the new op + kernel.

3. **Fuse `rope + K-cache write`** (`rope + set_rows` in ggml). Our
   `attention_cache_bf16` does the K/V memcpy + flash-attn-2 inner
   loop; the K rope is currently a separate `rmsnorm_rope_bf16`
   call that writes a temporary bf16 K, which then gets memcopied
   into the cache. Fusing the rope write directly into the cache
   slot avoids the temp buffer + one memcpy. Needs careful handling
   of cache-position wrap and head/lane indexing — review ggml's
   `rope.cu` + `set_rows` interaction first.

## Skipped (verified non-ggml)

- `rmsnorm + quantize_q8_1` fusion — ggml keeps quantize separate.
- `rmsnorm + mul_mat` fusion — ggml keeps these separate.

## Open questions

- Confirm the gemma E4B `q4_k`/`q6_k` weight assignment in the GGUF
  loader. The `linear_q6_k_gemv` call count is higher than expected
  for a standard Q4_K_M layout, suggesting some weights we treated
  as Q4_K are actually Q6_K (or vice versa). Worth checking
  `loader_gguf.odin` against the model's actual tensor types — and
  if some current Q6_K calls should be Q4_K, the fused gate+up+GEGLU
  path may apply to more layers than it does today.
