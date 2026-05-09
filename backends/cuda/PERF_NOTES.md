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
| MMA F16 attention (WMMA, in-tree but disabled)   | 46.2† | regr |
| ggml MMA F16 (offline-nvcc PTX, bf16→fp16 path)  | 55.3‡ | wash |
| + fattn-vec 16-byte LDG (gap #3)                 | 63.2§ | n/a  |
| + quantize_q8_1 256-thread block (gap #5)        | 63.2§ | wash |
| + Q6_K via q8_1+dp4a (gap #2)                    | 63.2§ | wash |
| + ROPE + K-cache-write fusion (gap #6)           | 63.1§ | wash |
| + fp32 activations end-to-end (drop pack)        | 63.0§ | wash |

\* Pre-window decode (32 tokens). Post-window decode at 600 tokens runs
~56 tok/s — the host-issued shift cuMemcpys add ~140 graph nodes per
token of overhead. Acceptable for now; MMA F16 should swamp this.

† 600-token decode with WMMA-based MMA kernel. Slower than vec because
it stages K/V through shared mem (vec reads direct from DRAM via
`__ldg`), uses m16n16k16 with 75% Q-row waste (vs ggml's m16n8k16),
and the FA2 scale_old rescale of the wmma accumulator does a
store→scale→load per tile. See gap #1 below for the path to make it
fast.

‡ MMA-only decode rate, computed from a 1000-token decode that crossed
the sliding-window boundary (~502 MMA tokens at 9.07 s). Wash vs vec
because the bf16→fp16 K/V conversion ggml's kernel requires
(`launch_fattn` pattern) eats the per-call MMA gain. The kernel itself
is fast; the host-side conversion is the overhead.

§ Greedy (`--temperature 0`) 256-token decode. The default sampling
path (`top-k=40` selection-sort over a 262 144-vocab logits row) adds
~15 ms/tok on the host that masks the GPU side, so the headline
collapses to ~31 tok/s with default temperature. Use `--temperature 0`
when comparing GPU work. Per-kernel `--timing` (256 tok greedy)
showed `attention_cache_vec_bf16` 26.5 → 22.8 µs avg after gap #3,
`linear_q6_k_gemv` 49.7 µs → `linear_q6_k_mmvq` 23.2 µs after gap #2,
and `cache_write_bf16` halved (~6 100 fewer K-side launches per
600-tok run) after gap #6. Total GPU time per token went 18.93 →
17.18 ms (~9% faster); wall time stayed at ~15.85 ms because host
work (forward-recording into the captured graph) is the bottleneck
in steady state.

Total: **45.8 → ~63.2 tok/s greedy (+38%)**.

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
  pointer to `attention_with_cache`. `Context.k_cache_written_this_forward`
  and `Context.v_cache_written_this_forward` dedup the shift+write
  separately for K and V so the fused `Rmsnorm_Rope_Write_Cache` op can
  mark only the K side as done.

### Fused-op opt-in mechanism
- `ml.odin` — `Backend_Capability` enum + `Backend_Capabilities` bit_set
  on `Backend`. CUDA sets `.Linear_Q4_K_Gate_Up_Geglu` and
  `.Rmsnorm_Rope_Write_Cache`; CPU/Vulkan do not.

### ROPE + K-cache-write fusion (gap #6 done)
- `kernels/rmsnorm/rmsnorm_rope_cache_bf16.cu` — same fused
  rmsnorm+rope as `rmsnorm_rope_bf16.cu` but writes the rotated K row
  directly to `cache[slot_base + row, ...]`, where
  `slot_base = min(*pos_dev, capacity - n_rows)` matching
  `cache_write_bf16`'s formula. Mirrors ggml's `rope_neox` `set_rows_stride`
  branch (`rope.cu:154-157`).
- `Rmsnorm_Rope_Write_Cache` op variant in `ml.odin` plus
  `rmsnorm_rope_write_cache` proc that falls back to `rmsnorm_rope` on
  backends without the capability.
- `_rmsnorm_rope_write_cache_forward` in `ops.odin` does the optional
  K-side cache shift (host-emitted memcpys before the kernel) and
  marks `k_cache_written_this_forward` so `_attention_cache_forward`
  skips the redundant K cache_write/shift but still runs the V side.

## Gaps vs ggml — direct ports, no experiments

Things ggml does on the same model+hw that we don't, ordered by leverage.
References are to `ggml/src/ggml-cuda/`.

### 1. Flash attention via tensor cores (MMA F16) — biggest gap
ggml's dispatcher (`fattn.cu:307` `ggml_cuda_get_best_fattn_kernel`)
returns `BEST_FATTN_KERNEL_MMA_F16` for our case: bf16 K/V, gqa_ratio=4,
mask present, max_bias=0, padded KV, on cc 8.6. The vec kernel is only
chosen on Ada Lovelace (cc≥8.9) for non-quantized K/V. Both Gemma E4B
attention head sizes (D=256 sliding, D=512 full) hit this path.

#### In-tree starting point
`kernels/attention/attention_cache_mma_bf16.cu` is a hand-rolled WMMA
flash-attention for D=256, ncols2=4 (gqa), bf16 K/V, ncols1=1 (decode).
Compiles via NVRTC with `--gpu-architecture=sm_86` and `#define
__CUDA_AMPERE_MMA__ 1` (the bf16 fragments are gated on this). Output
matches vec; correctness verified with 128-token decode and 600-token
decode (sliding window crossing).

**Currently slower than vec** (46.2 vs 56.2 tok/s on 600-token decode).
Three known sources of overhead vs ggml's optimized MMA path:

1. **Shared-mem K/V staging.** This kernel cooperatively loads K (then V)
   to shared per BC tile, then loads from shared into the WMMA fragment.
   Vec reads K/V directly from DRAM via `__ldg`. ~2× DRAM bandwidth on
   K/V because the data is read once into shared and a second time when
   the fragment is loaded. ggml uses `cp.async` to overlap the
   global→shared copy with prior compute, hiding this cost.
2. **m16n16k16 vs m16n8k16.** WMMA's bf16 fragment is 16×16; ggml uses
   `mma.sync.aligned.m16n8k16` raw PTX. With ncols2=4 valid Q rows, the
   16-row fragment wastes 75% of the tensor-core compute on zero rows.
   m16n8k16 wastes only 75% of half (still bad, but smaller fragment
   = more parallelism / better occupancy).
3. **scale_old rescale roundtrip.** FA2's per-tile alpha rescale needs
   to scale the VKQ accumulator fragment by a per-row factor. WMMA
   fragments are opaque (per-element layout is implementation-defined),
   so the rescale goes through shared mem (store→scale→load) per
   fragment per tile. Adds ~16 KB of shared-mem traffic per token per
   warp — small in absolute bandwidth but adds latency on each tile.

#### To make it faster than vec
- Switch from WMMA `<mma.h>` to raw `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`
  PTX inline asm. Smaller fragment, less Q-row waste, more parallelism.
- Add `cp.async.cg.shared.global` (Ampere `cp.async`) to pipeline the
  next tile's K/V load behind the current tile's mma.
- Avoid the scale_old roundtrip by directly manipulating the per-thread
  fragment register layout (Ampere mma.sync layout is documented and
  stable). Eliminates the shared-mem ping-pong.

These three changes are the bulk of `fattn-mma-f16.cuh` complexity.
Doing them brings us close to ggml's per-call cost.

#### Offline-nvcc literal port (in flight)
`kernels/attention/ggml_mma/` contains a wrapper that explicitly
instantiates `flash_attn_ext_f16<256, 256, 2, 4, false, false>` from
ggml's source. Compiled offline via `build_ptx.ps1`:

```
nvcc -arch=sm_86 -ptx -std=c++17 -O3 --extended-lambda \
     -Iggml/include -Iggml/src -Iggml/src/ggml-cuda \
     wrapper.cu -o attention_mma_d256_ncols2_4.ptx
```

Produces a 481 KB PTX with kernel symbol
`_Z18flash_attn_ext_f16ILi256ELi256ELi2ELi4ELb0ELb0EEv...` (mangled).

Template params:
- DKQ = DV = 256 (Gemma sliding head_dim).
- ncols1 = 2, ncols2 = 4. ggml's dispatcher pads decode (Q.ne[1]=1) up
  to ncols1=2 because the Ampere config table only has entries for
  ncols ∈ {8, 16, 32, 64} and gqa_ratio=4 → ncols2=4 → ncols1=2 is
  the minimum that fits. Kernel skips the dummy Q via `ic0 + j_VKQ
  >= ne01.z` checks.
- `use_logit_softcap = false`, `V_is_K_view = false`.

**Wired in and running.** Files added:
- `kernels/attention/ggml_mma/wrapper.cu` — explicit instantiation of
  `flash_attn_ext_f16<256, 256, 2, 4, false, false>`.
- `kernels/attention/ggml_mma/build_ptx.ps1` — nvcc invocation.
- `kernels/attention/ggml_mma/attention_mma_d256_ncols2_4.ptx` — output.
- `kernels/attention/ggml_mma/bf16_to_fp16.cu` — bf16→fp16 conversion
  for the K/V "shadow" buffers.

Plus in `pipeline.odin`: `_load_ptx_pipeline` (cuModuleLoadData +
cuModuleGetFunction with the mangled symbol).

Plus in `cuda.odin`: `Context.fp16_kv_cache` (per-cache-pointer dedup
map, cleared on `clear()`), `Context.fa_mask_dev` (static all-zero
mask, lazily allocated, freed on context_destroy).

Plus in `ops.odin`: `_dispatch_mma_attention_d256` does:
1. fp16 K and V dedup-allocate from activation pool, run
   `bf16_to_fp16_pairs` conversion (skipped for shared layers).
2. bf16 → fp32 Q conversion via existing `cast_bf16_to_f32` kernel.
3. fp32 dst + dst_meta scratch from activation pool.
4. ~30 stride/ne args computed inline.
5. Launch the PTX kernel with gridDim=(2,1,1) (= ntiles_dst, full-tile
   path, no fixup), block=(32,2,1), 99 KB dynamic shared mem.
6. Pack fp32 dst → bf16 output via existing `pack_f32_to_bf16_pairs`.

Wired into `_attention_cache_forward` conditional on `head_size==256
&& token_count==1 && gqa_ratio==4 && cache_position+1 >= sliding_window`
— the last condition ensures the all-zero mask is correct (sliding
cache fully filled). Pre-window decode and full-attention layers fall
through to vec.

#### Remaining optimization headroom on this path
1. **Eliminate the bf16→fp16 conversion overhead.** Currently we
   convert the entire K and V caches from bf16 to fp16 once per non-
   shared layer per forward (~256 KB per K + 256 KB per V × 20 unique
   sliding caches = ~10 MB of conversion bandwidth per token). Better:
   maintain a dual cache (bf16 + fp16) so cache_write writes both
   formats; the fp16 conversion happens once per new row (= 1 KB per
   sliding token), not per full cache.
2. **Stream-K parallelism.** Currently gridDim.x=2 (one block per
   output tile, full-tile no-fixup path). On a 3090 Ti with 84 SMs
   that's 0.6% block-occupancy. ggml's `launch_fattn` would pick
   gridDim.x=16 (= ntiles_KV × ntiles_dst = 8 × 2) — 8× more parallel,
   but requires the `flash_attn_combine_results` fixup kernel after.
   Worth the effort if compute (not memory) is the bottleneck.
3. **bf16-native kernel.** ggml's `mma.cuh` already has bf16 mma
   primitives (line 1065: `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`).
   Re-templating fattn-mma-f16.cuh from `half2` to `nv_bfloat162` would
   skip the conversion entirely. ~50 line patch to ggml's source we'd
   maintain ourselves.

### 2. Q6_K via q8_1 + dp4a (not bf16 + scalar fp32) — DONE
ggml's `vec_dot_q6_K_q8_1_impl_mmvq` (`vecdotq.cuh:624`) ported to
`kernels/linear/linear_q6_k_mmvq.cu`. Per-call kernel time 49.7 → 23.2 µs
(2.1× faster). Reuses `q8_1_cache` so most Q6_K calls share an existing
quantize result with co-located Q4_K matmuls. ~9% of GPU → ~4.5%.

### 3. Vectorized 16-byte loads in fattn-vec — DONE
`attention_cache_vec_bf16.cu` Q/K/V loops switched from 4-byte
`__ldg(uint*)` × 16 iters to 16-byte `__ldg(uint4*)` × 4 iters per
thread (4× fewer LDG instructions). Per-call kernel time 26.5 → 22.8 µs
on the 256-tok decode timing breakdown. Per-thread D-slice layout kept
the same so the output reduction needs no changes.

### 4. q4_K mmvq small-K path
`mmvq.cu:743 should_use_small_k` triggers when
`K < nwarps × vdr × 32 / qi × QK_K = 2048` for Q4_K. The dispatcher
sets `rows_per_cuda_block = nwarps = 4`: each warp owns a row instead
of all warps cooperating on one. For Gemma E4B only `per_layer_projection`
(K=256, N=2560) hits this from Q4_K. Limited impact for this model.

### 5. `quantize_q8_1` block size 256, not 32 — DONE
`CUDA_QUANTIZE_BLOCK_SIZE = 256`, 8 Q8_1 sub-blocks per CUDA block,
width-32 `warp_reduce_max/sum`. 8× fewer block launches per quantize.
Per-call wall is unchanged (the kernel is bandwidth-bound, not
launch-bound) but 8× fewer launches means 8× fewer captured graph
nodes for cuGraphExecUpdate.

### 6. ROPE + K-cache-write fusion — DONE
See "ROPE + K-cache-write fusion" under "What's in tree" above. The
fused `rmsnorm_rope_cache_bf16` kernel writes the rotated K row
directly to the cache slot, and the dedup split lets
`_attention_cache_forward` skip the K-side cache_write/shift while
still running the V side. Saves ~21 cache_write dispatches per token
(K side) on Gemma E4B and ~21 graph nodes from auto-graph capture.

## fp32 activations end-to-end — pipeline shape match (DONE)

The CUDA Gemma path now runs FP32 activations through the layer loop
(matching ggml's shape). Quantized weights stay Q4_K/Q6_K, normaliser
weights stay Bf16, KV cache stays Bf16, but all in-flight activations
are FP32. The `pack_f32_to_bf16_pairs` kernel is removed entirely —
mmvq writes directly to its FP32 output tensor.

Touched kernels (CUDA only; CPU/Vulkan are not maintained on this
phase):
- `quantize_q8_1_f32` (new, ggml signature) replaces the bf16 input
  path on the matmul→matmul boundary.
- `linear_q4_k_mmvq`, `linear_q4_k_gate_up_geglu_bf16`,
  `linear_q6_k_mmvq` write directly to FP32 dst (no scratch + pack).
- `rmsnorm` (existing kernel, updated to read Bf16 weight),
  `rmsnorm_rope_f32`, `rmsnorm_rope_cache_f32`, `add_rmsnorm_f32`,
  `gelu_mul_f32`, `attention_cache_vec_f32`, `cache_write_f32` —
  new FP32-activation paths that read Bf16 weight where applicable
  and write Bf16 to the KV cache where applicable.
- `_dispatch_mma_attention_d256` simplified: takes FP32 Q directly,
  writes FP32 output directly (no internal cast / pack).
- gemma.odin casts the embedded tokens to FP32 once after `select`,
  same for the per-layer-input lookup. lm_head's bf16 weight goes
  through cuBLAS via a pre-matmul cast back to Bf16 (then back to
  FP32) since `cublasGemmEx` doesn't support mixed Bf16/F32 inputs.
- Per-layer scalars (`embed_scale`, `ple_*`, `softcap*`,
  `layer_scalar`) stored as FP32 so `mul`/`add` compose without a
  cast.

### Per-kernel timings (256 tok greedy, `--timing`)

|                                | before refactor | after refactor |
|--------------------------------|-----------------|----------------|
| GPU total                      | 17.18 ms/tok    | **14.87 ms/tok (−13%)** |
| `linear_q4_k_mmvq`             | 12.5 µs avg     | 15.4 µs avg    |
| `quantize_q8_1_*`              | 8.4 µs (bf16)   | 9.4 µs (f32)   |
| `pack_f32_to_bf16_pairs`       | 7.5 µs × 296    | gone           |
| `attention_cache_vec_*`        | 22.8 µs (bf16)  | 21.0 µs (f32)  |

## Why headline tok/s isn't moving despite kernel wins

After gaps #2/#3/#5/#6 and the fp32 pipeline conversion, per-token GPU
compute went 18.93 → 14.87 ms (−21%) with `--timing` on. Headline
(no `--timing`, greedy) is unchanged at 63.0 tok/s = 15.87 ms/tok.

In auto-graph mode, GPU compute time and host work overlap; wall =
max(gpu, host). The wall has plateaued at ~15.85 ms/tok, which means
host work is on roughly the same order as GPU compute and shrinking
GPU alone won't move the wall further. The remaining host bottleneck
is the per-forward graph rebuild (Odin op-walk → `cuLaunchKernel`
recording → `cuGraphExecUpdate`) over ~250 captured nodes per token.

Levers from here:
1. Reduce per-forward op count further. ggml's MMQ-ish path elides
   the explicit `quantize_q8_1` by quantizing-on-the-fly inside the
   matmul kernel; that removes ~250 dispatches/token. Significant
   kernel-level work but a direct ggml port.
2. Pre-build the graph once per topology. Currently we capture each
   forward into a fresh Graph and `cuGraphExecUpdate` it; a stable
   "decode topology" graph cache would skip the re-record. Notes #1
   above ("Don't pursue a 'skip cuGraphExecUpdate' fast path") flagged
   prior attempts that regressed — would need a different design.
3. Make the host op-walk cheaper (Odin-side). Profiling-level work,
   not a ggml port.

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
  `linear_q6_k_mmvq` call count is higher than expected for a standard
  Q4_K_M layout. If some Q6_K weights should be Q4_K, the fused
  gate+up+GEGLU path applies to more layers than today.
