# CUDA backend decode-perf notes

Where decode throughput stands and what to try next on `gemma_chat_repl`.
Pick this up by re-reading this file, then the diff in the files listed
under "What's in tree."

## Numbers

Hardware: NVIDIA GeForce RTX 3090 Ti, cc=8.6, 84 SMs.
Model: Gemma 4 E4B, Q4_K_M GGUF.
Test: `gemma_chat_repl --gguf gemma_data/model.gguf --max-tokens 128`,
"write me a story" prompt, 13-token prefill.

Headline (no `--timing`):

| Stage                                            | tok/s |
|--------------------------------------------------|-------|
| Pre-session baseline                             | 45.8  |
| + skip MemsetD8Async on activation Data buffers  | 54.3  |
| + warp-shuffle reduction in rmsnorm-family       | 55.1  |

Reference target (ollama / llama.cpp on the same model+hw): ~130 tok/s.

`--timing` slows the run because `cuEventRecord` fires twice per dispatch.
Quote tok/s headlines from the no-`--timing` run; use `--timing` only
for the kernel breakdown.

## Where the time goes now

`--timing` snapshot at the 55.1 tok/s headline (timing-on absolute ms is
inflated by `cuEventRecord` overhead; treat shares, not absolutes, as
representative — the no-timing run drops to ~32 tok/s under timing):

| Kernel                              | share | notes |
|-------------------------------------|-------|-------|
| linear_q4_k_mmvq                    | ~24%  | q/k/v/o + down_proj |
| quantize_q8_1_bf16                  | ~14%  | bf16 → q8_1 (q8_1_cache halves *some* dispatches) |
| attention_cache_bf16                | ~13%  | inc. K/V memcpy + flash-attn-2 inner loop |
| linear_q4_k_gate_up_geglu_bf16      | ~12%  | fused FFN front-half (gate + up + GEGLU) |
| rmsnorm_bf16                        | ~12%  | rmsnorm + weight mul, fused |
| linear_q6_k_gemv                    | ~9%   | a subset of weights are Q6_K |
| add_rmsnorm_bf16                    | ~7%   | residual + pre_ff norm fused |
| add_bf16 / mul_bf16 / others        | ~9%   | per-layer residuals + scalar mul |

`gpu/wall ≈ 59%` with `--timing` on; without timing the host gap shrinks
to ~3-5 ms/tok.

**Q4_K matmul + quantize + gate_up_geglu = ~50% of GPU. This is the
largest concentration and the most promising target.**

## What's in tree

All under `backends/cuda/` plus per-call sites in
`networks/gemma/gemma.odin` and the fused-op machinery in `ml.odin`.

### Auto CUDA-graph capture
- `cuda.odin` / `graph.odin` / `buffer.odin` — `enable_decode_graph(true)`
  flips on transparent stream capture. `clear()` begins capture for the
  upcoming forward; `buffer_get` ends + `cuGraphExecUpdate`-or-
  reinstantiates + launches. Mirrors the same pattern as ggml's
  `ggml_cuda_graph_evaluate_and_capture`. Mutually exclusive with
  `enable_timing`.

### Activation memset skip
- `ml.odin` — `Backend.buffer_alloc` takes a `Buffer_Kind`.
- `backends/cuda/buffer.odin` (and Vulkan equivalent) skips the per-alloc
  zero-fill when `kind == .Data && !persist`. Forward kernels fully
  overwrite their output, so the zero was wasted work AND each skipped
  memset removes a node from the captured graph (~halves graph node
  count, lowers `cuGraphExecUpdate` cost). **The single biggest win
  this session: 45.8 → 54.3 tok/s.**
- `backends/cpu/cpu.odin` accepts the new param (no behavior change;
  `make` already zero-fills).

### Warp-shuffle reductions in rmsnorm-family
- `kernels/rmsnorm/rmsnorm_bf16.cu`, `add_rmsnorm_bf16.cu`,
  `rmsnorm_rope_bf16.cu` — replaced shared-mem tree reductions with
  intra-warp `__shfl_xor_sync` butterfly + lane-0 publish + a single
  `__syncthreads` + local fold across `warp_sums[NWARPS]`. One sync
  vs. ~log2(WG) before. Mirrors ggml's `norm.cu`.

### q8_1 input reuse
- `Context.q8_1_cache: map[DevicePtr]DevicePtr`, cleared in `clear()`.
  Q4_K matmul (and the fused gate+up+GEGLU) consult by input device
  pointer; emit `quantize_q8_1` only on cache miss. Saves ~35 quantizes/
  token via q/k/v dedup × 24 non-shared layers.

### Quant matmul kernels (M=1 decode), ggml-aligned body
- `linear_q4_k_mmvq.cu` — 4 warps × 32 lanes, `ROWS_PER_WG=2`, but the
  inner work mirrors ggml's `vec_dot_q4_K_q8_1_impl_vmmq`: 16 threads
  per K-block stride (`blocks_per_iter = 8`), each thread loads 2 q4
  ints + 4 q8 ints once and runs 4 dp4a calls covering 16 weights via
  low/high nibble passes. Min term uses our precomputed `q8_s` (one fp
  mul per sub-block-pair, only one thread credits it) instead of ggml's
  per-thread dp4a-of-1s trick.
- `linear_q4_k_gate_up_geglu_bf16.cu` — fused gate + up + GEGLU. One
  mmvq block accumulates two partial sums using shared q8_1 input loads.
  (Hasn't been ported to the new mmvq pattern; still uses the older
  body. Likely an additional small gain available there.)
- `linear_q6_k_gemv.cu` — 4 warps × 32 lanes, ROWS_PER_WG=2, hoisted
  bf16 input loads out of the row × quadrant inner loops.

### Position state in a device tensor
- `Context.position_pinned` (4-byte pinned host buffer) +
  `Context.position_dev` (4-byte device buffer). `_emit_position_upload`
  in `ops.odin` lazily writes the new `cache_position` to pinned and
  emits one HtoDAsync per forward. `rmsnorm_rope_bf16`,
  `rope_bf16`/`rope.cu`, and `attention_cache_bf16` take `const int*
  position_offset_dev` and read once at kernel entry. Mirrors how ggml
  keeps position offsets in tensors (their `pos` tensor src).

### K/V cache write via kernel
- `kernels/attention/cache_write_bf16.cu` — copies a `[n_rows, kv_size]`
  bf16 source into the `[t_capacity, kv_size]` cache slot at offset
  `(*pos_dev + row) % t_capacity`. Replaces the host-issued
  `cuMemcpyDtoDAsync` (one per K, one per V, plus optional wrap pair)
  inside `_attention_cache_forward`. Similar in spirit to ggml's
  `set-rows.cu`, but standalone rather than fused with rope.

### Attention
- `attention_cache_bf16.cu` — vectorized inner K dot loop to `uint4`
  loads (8 bf16 / load).

### Fused-op opt-in mechanism
- `ml.odin` — `Backend_Capability` enum + `Backend_Capabilities` bit_set
  on `Backend`. Procs that emit fused ops decompose to primitives when
  the cap is absent. CUDA sets `.Linear_Q4_K_Gate_Up_Geglu`; CPU/Vulkan
  do not, so they get the legacy gate/up/gelu_mul sequence.

## Findings on ggml (verified `ggml/src/ggml-cuda/ggml-cuda.cu`)

- **ggml uses `cudaGraphExecUpdate`**, same as us. Their fast path is a
  uid-keyed skip: when llama.cpp passes a cgraph with the same `uid` as
  the previous compute, ggml short-circuits the node walk + update +
  just relaunches the existing exec (`ggml_cuda_graph_update_required`
  returns false on uid match, line 3141). We don't have an equivalent
  uid mechanism — rebuilding stable HtoD source pointers + a heuristic
  "stability detection" in our setup did not measurably move tok/s and
  was reverted; leaving cuGraphExecUpdate to handle it on each step is
  fine.
- **Why the skip works for them**: all step-varying state lives in
  tensors (= stable device buffers). Our cache_position-as-tensor and
  cache_write-via-kernel changes adopt this pattern.
- **Their Q4_K mmvq is 2-3× our throughput at the same nwarps/rows
  shape** (own observation; matches `mmvq.cu:293-345` using
  `nwarps=4, rows_per_block=1` on Ampere). A direct port of their inner
  body to our kernel only narrowed the gap to ~4%; the remaining gap
  is in something more subtle (likely register tiling, instruction
  scheduling, or a difference in how their template specialization
  generates code). **This is the largest remaining lever and the next
  thing to investigate.**

## Things to NOT redo

- Don't combine `auto_graph` with `enable_timing` — asserted off in code;
  `cuEventRecord` is not stream-capturable.
- Don't pursue a uid-style "skip cuGraphExecUpdate" path. Tried both
  Step A (skip update on heuristic stable detection) and Step B (full
  bypass of `gemma.forward_cached` with stable-pinned host staging).
  Step A was a no-op on this hardware; Step B was a regression. ggml's
  uid skip works because their cgraph rebuild is cheaper than ours, and
  their uid is set externally — we can't easily replicate that.
- Don't pursue `rmsnorm + quantize_q8_1` or `rmsnorm + mul_mat` fusion.
  ggml does neither.
- Don't fold K/V cache writes into `attention_cache_bf16` without
  cooperative-groups grid sync.
- Don't skip the warmup forward before enabling capture — cuBLAS first-
  call algo selection isn't capturable cold (`auto_warmup_done` exists
  for this).

## Next steps (kernel-focused, ordered by leverage)

1. **Close the Q4_K mmvq gap.** Combined Q4_K mmvq + quantize_q8_1 +
   gate_up_geglu = ~50% of GPU, and ggml's body is materially faster
   than ours at the same shape. The first port (`linear_q4_k_mmvq.cu`)
   adopted ggml's outer loop and inner vec_dot but only got 4%, far
   below the apparent 2-3× headroom. Next angles to investigate:
   - `__launch_bounds__` / register usage. Compare `cuobjdump` of our
     kernel vs ggml's for register count and spilling. ggml uses 1
     block per SM; we may differ.
   - Try `ROWS_PER_WG=1` (matching ggml exactly) and use atomics for
     the bf16 pair-packed output write.
   - Inline `unpack_scale_min` differently — ggml's `aux[2]` packing
     may produce better PTX than our branch-on-sub.
   - Measure whether the bottleneck is bandwidth or compute. If
     bandwidth, check whether we're hitting the L1/L2 line-fill rate.

2. **Port mmvq pattern to `linear_q4_k_gate_up_geglu_bf16`.** It's 12%
   of GPU and structurally similar to mmvq. Whatever closes the gap on
   mmvq should apply here directly.

3. **Port mmvq pattern to `linear_q6_k_gemv`.** 9% of GPU. Q6_K is its
   own format but the per-thread-work / register-tiling lessons from
   step 1 carry over.

4. **Fuse `rope + K-cache write`** (= ggml's `rope + set_rows`). Our
   `cache_write_bf16` is a partial step toward this. Full fusion would
   eliminate the temp K bf16 writeback. Modest GPU bandwidth win.

## Open questions

- Confirm Gemma E4B `q4_k`/`q6_k` weight assignment in the GGUF loader.
  `linear_q6_k_gemv` call count is higher than expected for a standard
  Q4_K_M layout. If some Q6_K weights should be Q4_K, the fused gate+
  up+GEGLU path applies to more layers than today.
