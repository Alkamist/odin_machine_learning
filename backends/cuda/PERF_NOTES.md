# CUDA backend decode-perf notes

Where decode throughput stands on `gemma_chat_repl` and what to port from
ggml next. Pick this up by re-reading this file, then the diff in the
files listed under "What's in tree."

## Numbers

Hardware: NVIDIA GeForce RTX 3090 Ti, cc=8.6, 84 SMs.
Model: Gemma 4 E4B, Q4_K_M GGUF.
Test: `gemma_chat_repl --gguf gemma_data/model.gguf --max-tokens 128`,
"write me a story" prompt, 13-token prefill.

Headline (no `--timing`):

| Stage                                            | tok/s | Δ    |
|--------------------------------------------------|-------|------|
| Pre-session baseline                             | 45.8  | —    |
| + skip MemsetD8Async on activation Data buffers  | 54.3  | +8.5 |
| + warp-shuffle reduction in rmsnorm-family       | 55.1  | +0.8 |
| + Q4_K mmvq literal port from ggml               | 55.8  | +0.7 |
| + fattn-vec port for decode attention (D=256)    | 58.8  | +3.0 |
| + fattn-vec generalized to D=512                 | 60.7  | +1.9 |
| + Q4_K gate_up_geglu literal port                | 61.5  | +0.8 |
| + Q6_K literal port                              | 60.5  | wash |
| + linear KV cache layout (foundation for MMA F16)| 61.7\*| wash |

\* Pre-window decode (32 tokens). Post-window decode at 600 tokens runs
~57 tok/s — the host-issued shift cuMemcpys add ~140 graph nodes per
token of overhead. Acceptable for now; MMA F16 should swamp this.

Total: **45.8 → ~60.5 tok/s (+32%)**.

Reference target (ollama / llama.cpp on the same model+hw): ~130 tok/s.

`--timing` slows the run because `cuEventRecord` fires twice per dispatch.
Quote tok/s headlines from the no-`--timing` run; use `--timing` only
for the kernel breakdown.

## What's in tree

All under `backends/cuda/` plus per-call sites in
`networks/gemma/gemma.odin` and the fused-op machinery in `ml.odin`.

### Auto CUDA-graph capture
- `cuda.odin` / `graph.odin` / `buffer.odin` — `enable_decode_graph(true)`
  flips on transparent stream capture. `clear()` begins capture; `buffer_get`
  ends + `cuGraphExecUpdate`-or-reinstantiates + launches. Mutually
  exclusive with `enable_timing`.

### Activation memset skip (biggest single win)
- `ml.odin` — `Backend.buffer_alloc` takes a `Buffer_Kind`.
- `backends/cuda/buffer.odin` (and Vulkan equivalent) skip the per-alloc
  zero-fill when `kind == .Data && !persist`. Forward kernels fully
  overwrite their output, AND each skipped memset removes a node from
  the captured graph. **45.8 → 54.3 tok/s.**

### Warp-shuffle reductions in rmsnorm-family
- `kernels/rmsnorm/rmsnorm_bf16.cu`, `add_rmsnorm_bf16.cu`,
  `rmsnorm_rope_bf16.cu` — `__shfl_xor_sync` butterfly + lane-0 publish +
  one `__syncthreads` + local fold across `warp_sums[NWARPS]`.

### q8_1 input reuse
- `Context.q8_1_cache: map[DevicePtr]DevicePtr`, cleared in `clear()`.
  Q4_K matmul (and the fused gate+up+GEGLU) consult by input device
  pointer; emit `quantize_q8_1` only on cache miss.

### Quant matmul kernels — literal ports of ggml's `mul_mat_vec_q`
Ports of `ggml/src/ggml-cuda/mmvq.cu` + `vecdotq.cuh`, specialized for
ncols_dst=1, no fusion variants, no MoE channels/strides. All write fp32
output; `pack_f32_to_bf16_pairs.cu` converts to the bf16 pair-packed
format the rest of the pipeline reads.

- `linear_q4_k_mmvq.cu` — port of `vec_dot_q4_K_q8_1_impl_vmmq` body,
  `mul_mat_vec_q` outer loop with `rows_per_cuda_block=1`, `nwarps=4`,
  `blocks_per_iter=8`.
- `linear_q4_k_gate_up_geglu_bf16.cu` — same with a parallel `tmp_gate`
  alongside `tmp` (= `tmp_up`); shares q8_1 input loads between gate and
  up matmuls; final `gelu_tanh(gate)*up` combine before the fp32 store.
- `linear_q6_k_gemv.cu` — outer structure ported. Inner dot reads bf16
  input directly rather than going through ggml's q8_1 + dp4a Q6_K path
  (see "Gaps vs ggml #2").
- `pack_f32_to_bf16_pairs.cu` — trivial f32→bf16-pair converter.

### Flash attention — port of ggml's `fattn-vec.cuh`
- `kernels/attention/attention_cache_vec_bf16.cu` — port of
  `flash_attn_ext_vec`, stripped to bf16 K/V, ncols=1, causal + optional
  sliding window. Compiled twice via NVRTC `-DD_HEAD={256,512}`.
- NTHREADS_KQ = NTHREADS_V = 8 cooperate per K-row dot / V output element.
  Q in registers pre-scaled by 1/sqrt(D). 128 threads/block, BC=128.
- Note: ggml does NOT actually use fattn-vec for our case on this hw —
  see "Gaps vs ggml #1".

### Position state in a device tensor
- `Context.position_pinned` (4-byte pinned host buffer) +
  `Context.position_dev` (4-byte device buffer). `_emit_position_upload`
  in `ops.odin` lazily writes `cache_position` to pinned and emits one
  HtoDAsync per forward. `rmsnorm_rope_bf16`, `rope_bf16`/`rope.cu`,
  and `attention_cache*_bf16` take `const int* position_offset_dev` and
  read once at kernel entry.

### Linear K/V cache layout (ggml-compatible)
KV cache is stored linearly in seq order: slot 0 oldest, slot cap-1
newest. No ring-buffer modulo anywhere — every cache-reading kernel
iterates contiguous slot ranges, and ggml drop-in kernels (incl.
`flash_attn_ext_f16` MMA, when ported) work without modification.

- `kernels/attention/cache_write_bf16.cu` — writes new K/V rows at slot
  `min(*pos_dev, capacity - n_rows) + row`. For full layers the cache
  is sized to user max_context, so writes are pure linear append. For
  sliding layers in steady state writes go at slots [cap-n_rows, cap)
  after a host-emitted shift.
- `_attention_cache_forward` (ops.odin) — when sliding cache would
  overflow (`cache_position + n_rows > capacity`), emits two
  `cuMemcpyDtoDAsync` (per K and per V) through `Context.shift_scratch_dev`
  to shift contents back by `shift_amount = pos + n - cap` rows. Memcpy
  size and topology are stable across decode steps, so the captured
  graph is patchable via `cuGraphExecUpdate`.
- Per-Q-token live K range computed from a unified formula:
  `t_q_slot = min(cache_position + t_q, capacity - q_token_count + t_q)`.
  Pre-fill / pre-shift: equals `cache_position + t_q`. Post-shift:
  pinned to `capacity - q_token_count + t_q`. `t_k_max = t_q_slot + 1`
  (causal); `t_k_min` from window param (for sliding) or 0 (full).
- Gemma's KV-shared layers (18 of 42) re-pass the source layer's k_cache
  pointer to `attention_with_cache`. `Context.cache_written_this_forward`
  dedups the shift+write so only the source layer's call mutates the
  cache; shared layers run attention only.

### Fused-op opt-in mechanism
- `ml.odin` — `Backend_Capability` enum + `Backend_Capabilities` bit_set
  on `Backend`. CUDA sets `.Linear_Q4_K_Gate_Up_Geglu`; CPU/Vulkan do not.

## Gaps vs ggml — direct ports, no experiments

Things ggml does on the same model+hw that we don't, ordered by leverage.
References are to `ggml/src/ggml-cuda/`.

### 1. Flash attention via tensor cores (MMA F16) — biggest gap
ggml's dispatcher (`fattn.cu:307` `ggml_cuda_get_best_fattn_kernel`)
returns `BEST_FATTN_KERNEL_MMA_F16` for our case: bf16 K/V, gqa_ratio=4,
mask present, max_bias=0, padded KV, on cc 8.6. The vec kernel is only
chosen on Ada Lovelace (cc≥8.9) for non-quantized K/V. Both Gemma E4B
attention head sizes (D=256 sliding, D=512 full) hit this path.

MMA F16 uses 16×16 bf16 tensor-core fragments via `mma.cuh` plus
`cp.async` (`cp-async.cuh`) for global→shared staging. Port surface is
`fattn-mma-f16.cuh` (~600 LOC) plus 21 `fattn-mma-f16-instance-*.cu`
template instances under `template-instances/`. Substantial effort but
this is the dominant remaining decode gap.

### 2. Q6_K via q8_1 + dp4a (not bf16 + scalar fp32)
ggml's `vec_dot_q6_K_q8_1_impl_mmvq` (`vecdotq.cuh:624`) uses 2× `__dp4a`
per K-block (8 i8×i8 MACs as 2 instructions plus a `__vsubss4`). Our
`linear_q6_k_gemv.cu` does 8 scalar fp32 FMAs per K-block on bf16 input.
Reusing the existing `q8_1_cache` adds zero quantize overhead since Q6_K
matmuls in Gemma share inputs with Q4_K matmuls. Q6_K is ~7-9% of GPU.

### 3. Vectorized 16-byte loads in fattn-vec
ggml uses `ggml_cuda_memcpy_1<16>` (`common.cuh:756`, compiles to
LD.E.128 on Volta+) for Q/K/V loads. We use 4-byte
`__ldg(unsigned int*)` in 16-iteration loops at
`attention_cache_vec_bf16.cu:99-104, 142-148, 201-208`. **4× more LDG
transactions per K-row dot and per V-col load.** Applies even if we
keep vec as a fallback for D=512 after #1 lands.

### 4. q4_K mmvq small-K path
`mmvq.cu:743 should_use_small_k` triggers when
`K < nwarps × vdr × 32 / qi × QK_K = 2048` for Q4_K. The dispatcher
sets `rows_per_cuda_block = nwarps = 4`: each warp owns a row instead
of all warps cooperating on one. For Gemma E4B only `per_layer_projection`
(K=256, N=2560) hits this from Q4_K. Limited impact for this model.

### 5. `quantize_q8_1` block size 256, not 32
`CUDA_QUANTIZE_BLOCK_SIZE = 256` (`quantize.cuh:8`). Each CUDA block
handles 8 Q8_1 blocks via 8 warps; per-warp reduction with
`warp_reduce_max<QK8_1>`. Our `quantize_q8_1_bf16.cu` runs 32 threads
per block, one Q8_1 block per CUDA block. **8× fewer block launches
per quantize**. 30-minute change.

### 6. ROPE + K-cache-write fusion
`rope.cu:154-157` (rope_neox `set_rows_stride` branch) writes directly
into the K cache slot using `row_indices[i2]`. We have separate
`rope_bf16` then `cache_write_bf16` kernels. Saves one launch per
attention block.

## Things to NOT redo

- Don't combine `auto_graph` with `enable_timing` — asserted off in code;
  `cuEventRecord` is not stream-capturable.
- Don't pursue a "skip cuGraphExecUpdate" fast path. Tried both Step A
  (skip update on heuristic stable detection) and Step B (full bypass
  of `gemma.forward_cached` with stable-pinned host staging). Step A
  was a no-op; Step B regressed. ggml's uid skip works because their
  cgraph rebuild is cheap and uid is set externally — we can't easily
  replicate that. Reverted.
- Don't pursue `rmsnorm + quantize_q8_1` or `rmsnorm + mul_mat` fusion.
  ggml does neither.
- Don't fold K/V cache writes into `attention_cache_vec_bf16` without
  cooperative-groups grid sync.
- Don't skip the warmup forward before enabling capture — cuBLAS first-
  call algo selection isn't capturable cold (`auto_warmup_done` exists
  for this).

## Open questions

- Confirm Gemma E4B `q4_k`/`q6_k` weight assignment in the GGUF loader.
  `linear_q6_k_gemv` call count is higher than expected for a standard
  Q4_K_M layout. If some Q6_K weights should be Q4_K, the fused
  gate+up+GEGLU path applies to more layers than today.
