# CUDA backend decode-perf notes

Status of Gemma 4 E4B Q4_K_M decode on `gemma_chat_repl`. Re-read this
file before starting a new round of work.

## TL;DR

We are at the 130 tok/s reference target for the standard
"write me a short story about a cat" / 256-token decode test. The
remaining work below is speculative — none of it is needed to close a
known gap.

The fix that closed it was the **printer-thread split in
`examples/gemma_chat_repl/main.odin`** (session 2026-05-09). The
per-token `os.flush(os.stdout)` was stalling the host thread, and
because `clear()` synchronises on the CUDA stream at the start of each
forward, the GPU was idling on every flush — even when stdout was a
file. Decoupling the print into a `core:sync/chan` queue + worker
thread eliminated the stall.

## Numbers

Hardware: NVIDIA GeForce RTX 3090 Ti, cc=8.6, 84 SMs, ~1 TB/s memory.
Model: Gemma 4 E4B, Q4_K_M GGUF.
Test: `gemma_chat_repl --gguf gemma_data/model.gguf --max-tokens 256
--temperature 0` with the prompt "write me a short story about a cat"
(16-token prefill).

| stdout destination     | before printer thread | after  |
|------------------------|----------------------:|-------:|
| `>` file               |                  ~97  | ~131   |
| `\|` grep / pipe       |                   ~64 | ~129   |

After the fix the destination no longer matters (within noise). Both
runs produce ~130 tok/s (≈ 7.7 ms/tok wall).

The "before" file-redirect number was assumed in the prior PERF_NOTES
to be the GPU-bound floor; it was not. Even file-mode `os.flush` was
costing ~3 ms/tok of host-side stall on top of the captured graph
launch.

GPU compute (`--timing`, redirected output): ~14.9 ms/tok. `--timing`
adds ~9 ms/tok from `cuEventRecord` so the wall during a timing run is
misleading; the per-kernel µs values it reports are still trustworthy.

Reference target: ~130 tok/s, attributed to llama.cpp / ollama on the
same model+hw. **Still not independently verified** with `llama-bench`,
but we are matching it now so the urgency is gone.

### How ollama and llama.cpp avoid the flush stall (for reference)
Both run the model in a server context (HTTP server in ollama, in-proc
`server_context` in llama.cpp's `llama-cli`). The decode loop posts
tokens to a buffered queue/channel and returns immediately to the next
forward; a separate thread drains the queue and does the actual stdout
write + flush. The GPU never blocks on terminal I/O.

- ollama: `runner/ollamarunner/runner.go:200`
  (`responses: make(chan response, 100)`).
- llama.cpp: `tools/cli/cli.cpp:57` (`server_context` task queue;
  `server_response_reader` drains on a separate thread).

Our equivalent: `examples/gemma_chat_repl/main.odin` `Printer` struct +
`core:sync/chan` (cap 256) + `core:thread` worker. See "What's in
tree" below.

## Per-kernel breakdown (per token, --timing)

| kernel                            | per-call | calls/tok | total/tok | bw eff. |
|-----------------------------------|---------:|----------:|----------:|--------:|
| `linear_q4_k_mmvq` (K=N=2560)     | 16.1 µs  | 221       | 3.55 ms   | ~23%    |
| `quantize_q8_1_f32`               | 10.0 µs  | 249       | 2.49 ms   | low     |
| `rmsnorm_f32`                     | 11.4 µs  | 191       | 2.18 ms   | low     |
| `linear_q4_k_gate_up_geglu_bf16`  | 41.4 µs  | 41        | 1.70 ms   | ~73%    |
| `linear_q6_k_mmvq`                | 22.2 µs  | 33        | 0.73 ms   | ~30%    |
| `add_rmsnorm_f32`                 | 16.4 µs  | 41        | 0.67 ms   | ~50%    |
| `attention_cache_vec_f32`         | 18.2 µs  | 34        | 0.62 ms   | ~5%     |
| `add_f32`, `mul_f32`, etc.        |          |           | ~3.0 ms   |         |

~1000 dispatches per token total. Big kernels are reasonably efficient
(`gate_up_geglu` at 73% of peak DRAM bw). Small kernels are launch- and
overhead-bound — peak DRAM utilisation 5–30% is typical.

## Speculative wins beyond the reference target

We're at 130 tok/s. None of the items below are needed to close a known
gap; they're notes for if/when we want to push past the reference.
Listed in rough order of leverage. The leverage estimates are stale —
they were sized against the prior 97 tok/s baseline. Rerun the math
against the current ~7.7 ms/tok wall before treating any of them as
worthwhile.

1. **`attention_cache_vec_f32` only uses ~10% of the SMs on decode.**
   `ops.odin:1184` launches with `gridDim = (n_q_heads, q_token_count, 1)
   = (8, 1, 1)` — 8 blocks on 84 SMs. ggml's `launch_fattn`
   (`fattn-common.cuh:1091-1118`) computes `parallel_blocks` across the K
   dimension based on `cudaOccupancyMaxActiveBlocksPerMultiprocessor` and
   `ntiles_KV`, then runs a `flash_attn_combine_results` fixup. Modest
   win for 512-slot sliding, bigger on full-attention layers and long
   context.

2. **NVRTC vs offline nvcc compilation.** Only the ggml MMA wrapper is
   offline-compiled (`build_ptx.ps1`). Everything else goes through NVRTC
   at startup. NVRTC's optimizer passes are not identical to `nvcc -O3` —
   for register-heavy kernels at 5–10 µs you're paying every cycle.
   Worth trying offline cubin for `linear_q4_k_mmvq`, `linear_q6_k_mmvq`,
   `quantize_q8_1_f32`, the rmsnorm family, and comparing.

3. **Gemma E4B has more dispatches than typical models.** The PLE
   machinery (`per_layer_input_gate`, `gelu_mul`, `per_layer_projection`,
   `post_per_layer_input_norm`, `add`, `mul`) is ~6 ops per layer × 42 =
   ~250 extra dispatches/token vs vanilla Llama. With 1000 small kernels
   per token, you eat overhead floors regardless of how good each kernel
   is. Only fusion can reduce it.

4. **ggml-style stable-graph fast path** (skip the whole forward
   re-walk on stable steps and call `cudaGraphLaunch` directly). The
   captured-graph path already records once and patches with
   `cuGraphExecUpdate`; the host-side `forward_cached` re-walk is the
   remaining cost. With wall now ≈ gpu (printer-thread fix removed the
   serial host-print stall in front of the next forward), this is a
   smaller win than it looked from the prior baseline.

## How ggml's stable-graph fast path actually works

Reference: `ggml/src/ggml-cuda/ggml-cuda.cu:4188` (`ggml_backend_cuda_graph_compute`):

1. `ggml_cuda_graph_update_required(cgraph)` does a memcmp of node
   properties (op, dims, src ptrs) against the previously-stored snapshot.
   Cheap.
2. If properties unchanged AND warmup_complete: **skip stream capture,
   skip cgraph evaluation, skip `cuGraphExecUpdate`**. Just call
   `cudaGraphLaunch(graph->instance, stream)`. One driver call.
3. If properties changed: invalidate the warmup, evaluate cgraph
   directly (no graph). On next stable forward, re-capture and update.

The "Step A: skip update on heuristic stable detection" attempt
described in the previous PERF_NOTES failed because skipping just
`cuGraphExecUpdate` keeps the stream capture and forward_cached re-walk
— that's where most of the host work is. Skipping the WHOLE forward
(option (a) above) is what ggml does. To replicate it we'd need a
forward function that doesn't issue any backend dispatches when the
topology is provably unchanged — i.e. record the dispatch sequence
once (during warmup) and just cuGraphLaunch on subsequent calls. The
op list in `_current_ctx.operations` is already populated; it's the
backend dispatches inside `forward_cached` that need to be elided.

## What's in tree

All under `backends/cuda/` plus per-call sites in
`networks/gemma/gemma.odin` and the fused-op machinery in `ml.odin`.

### Chat REPL printer thread (closes the flush stall)
`examples/gemma_chat_repl/main.odin` runs a `core:thread` worker that
drains a `core:sync/chan.Chan(string)` (cap 256) of detokenised string
deltas. The decode loop clones the delta, increments a
`sync.Wait_Group`, and `chan.send`s; the worker `chan.recv`s, writes,
flushes, deletes, and `wait_group_done`s. Decode loop drains via
`wait_group_wait` before any direct `fmt.println` (timing line, etc.)
so output stays ordered. Channel is closed + thread joined on exit.

### Auto CUDA-graph capture
- `cuda.odin` / `graph.odin` / `buffer.odin` — `enable_decode_graph(true)`
  flips on transparent stream capture. `clear()` begins capture; `buffer_get`
  ends + `cuGraphExecUpdate`-or-reinstantiates + launches. Mutually
  exclusive with `enable_timing`. Always re-records every forward; does
  not yet implement the ggml-style stable-graph fast path described
  above.

### Activation memset skip
- `ml.odin` — `Backend.buffer_alloc` takes a `Buffer_Kind`.
- `backends/cuda/buffer.odin` (and Vulkan equivalent) skip the per-alloc
  zero-fill when `kind == .Data && !persist`. Forward kernels fully
  overwrite their output; each skipped memset removes one node from the
  captured graph.

### Warp-shuffle reductions in the rmsnorm family
- `kernels/rmsnorm/rmsnorm_bf16.cu`, `add_rmsnorm_bf16.cu`,
  `rmsnorm_rope_bf16.cu`, `add_rmsnorm_f32.cu`, `rmsnorm_rope_f32.cu`,
  `rmsnorm_rope_cache_*.cu` — `__shfl_xor_sync` butterfly + lane-0
  publish + one `__syncthreads` + local fold across `warp_sums[NWARPS]`.
- **`kernels/rmsnorm/rmsnorm.cu` (the f32 plain rmsnorm) still uses the
  old shared-memory tree.** Listed under "fix list" above.

### q8_1 input reuse
- `Context.q8_1_cache: map[DevicePtr]DevicePtr`, cleared in `clear()`.
  Q4_K matmul (and the fused gate+up+GEGLU) consult by input device
  pointer; emit `quantize_q8_1` only on cache miss. Hits q/k/v sharing
  the same rmsnorm output (3 mmvq → 1 quantize per layer).

### Quant matmul kernels — literal ports of ggml's `mul_mat_vec_q`
Specialized for ncols_dst=1, no fusion variants, no MoE.

- `linear_q4_k_mmvq.cu` — port of `vec_dot_q4_K_q8_1_impl_vmmq` body,
  `mul_mat_vec_q` outer loop with `rows_per_cuda_block=1`, `nwarps=4`,
  `blocks_per_iter=8`.
- `linear_q4_k_gate_up_geglu_bf16.cu` — same with parallel `tmp_gate`
  alongside `tmp` (= `tmp_up`); shares q8_1 input loads between gate and
  up matmuls; final `gelu_tanh(gate)*up` combine before fp32 store.
- `linear_q6_k_mmvq.cu` — port of `vec_dot_q6_K_q8_1_impl_mmvq`.

### Flash attention — port of ggml's `fattn-vec.cuh`
- `kernels/attention/attention_cache_vec_bf16.cu`, `attention_cache_vec_f32.cu`
  — port of `flash_attn_ext_vec`, stripped to bf16 K/V, ncols=1, causal +
  optional sliding window. Compiled twice via NVRTC `-DD_HEAD={256,512}`.
- 16-byte LDG (uint4) on K/V, NTHREADS_KQ = NTHREADS_V = 8 cooperate per
  K-row dot / V output element. Q in registers pre-scaled by 1/sqrt(D).
  128 threads/block, BC=128.
- **Missing parallel_blocks across the K dim** — see fix list.

### Position state in a device tensor
- `Context.position_pinned` (4-byte pinned host buffer) +
  `Context.position_dev` (4-byte device buffer). `_emit_position_upload`
  in `ops.odin` lazily writes `cache_position` to pinned and emits one
  HtoDAsync per forward. Position-bearing kernels read once at kernel
  entry. Keeps captured graph kernel-args bit-stable across decode steps.

### Linear K/V cache layout (ggml-compatible)
KV cache is stored linearly in seq order: slot 0 oldest, slot cap-1
newest. No ring-buffer modulo anywhere; ggml drop-in kernels work
without modification.

- `kernels/attention/cache_write_*.cu` — writes new K/V rows at slot
  `min(*pos_dev, capacity - n_rows) + row`.
- `_attention_cache_forward` (ops.odin) — when sliding cache would
  overflow, emits two `cuMemcpyDtoDAsync` (per K and per V) through
  `Context.shift_scratch_dev` to shift contents back. Memcpy size and
  topology are stable across decode steps, so the captured graph is
  patchable via `cuGraphExecUpdate`.
- Per-Q-token live K range from a unified formula:
  `t_q_slot = min(cache_position + t_q, capacity - q_token_count + t_q)`.
- Gemma's KV-shared layers (18 of 42) re-pass the source layer's k_cache
  pointer; `Context.{k,v}_cache_written_this_forward` dedup the
  shift+write so only one shift+write happens per cache.

### ROPE + K-cache-write fusion
- `kernels/rmsnorm/rmsnorm_rope_cache_*.cu` — same fused rmsnorm+rope
  as `rmsnorm_rope_*.cu` but writes the rotated K row directly to
  `cache[slot_base + row, ...]`. Mirrors ggml's `rope_neox`
  `set_rows_stride` branch.
- `Rmsnorm_Rope_Write_Cache` op variant in `ml.odin` plus
  `rmsnorm_rope_write_cache` proc that falls back to `rmsnorm_rope` on
  backends without the capability.

### Fused-op opt-in mechanism
- `ml.odin` — `Backend_Capability` enum + `Backend_Capabilities` bit_set
  on `Backend`. CUDA sets `.Linear_Q4_K_Gate_Up_Geglu` and
  `.Rmsnorm_Rope_Write_Cache`; CPU/Vulkan do not.

### fp32 activations end-to-end
The CUDA Gemma path runs FP32 activations through the layer loop
(matching ggml's shape). Quantized weights stay Q4_K/Q6_K, normaliser
weights stay Bf16, KV cache stays Bf16. Removed `pack_f32_to_bf16_pairs`
entirely — mmvq writes directly to its FP32 output tensor.

Reduced GPU/tok from 17.18 ms to 14.87 ms (-13%). Did NOT move headline
tok/s because wall is GPU-bound and the smaller savings were close to
host-overhead order. Worth periodically reconsidering reverting to bf16
activations now that MMA F16 was abandoned (the only reason for fp32
was MMA's fp16 K/V requirement) — would cut activation memory traffic in
half on the post-norm boundaries.

### MMA F16 attention (in-tree but disabled)
- `kernels/attention/attention_cache_mma_bf16.cu` — hand-rolled WMMA;
  46.2 tok/s on 600-token decode (regression vs vec).
- `kernels/attention/ggml_mma/wrapper.cu` + offline-built PTX —
  literal ggml `flash_attn_ext_f16<256, 256, 2, 4, false, false>`
  instantiation; 55.3 tok/s on a sliding-crossing decode (wash vs vec
  at 61.7). The bf16→fp16 K/V conversion ggml's kernel requires (~10 MB
  per token) eats the per-call MMA gain.
- Both gated off the live path. **Don't re-enable without solving the
  bf16↔fp16 K/V cache conversion problem first** — the kernel works,
  the surrounding glue is what kills it.

## Calibration: small per-row kernels are launch-bound, not reduction-bound

**Lesson from the rmsnorm warp-shuffle port:** the f32 rmsnorm kernel
went from a shared-mem reduction tree (8×`__syncthreads`) to a
warp-shuffle butterfly + 1×`__syncthreads`. Per-call time dropped
10.9 → 10.3 µs. With 191 calls/tok, total saved = ~0.1 ms/tok.
Headline tok/s unchanged within noise.

The kernel is launching with `gridDim=(count, 1, 1) = (1, 1, 1)` for
decode and 256 threads — 256 threads on 84 SMs is SM-occupancy noise.
Kernel body work is ~30 KB of memory traffic that finishes in <1 µs of
actual GPU work. The remaining ~9 µs is kernel-launch overhead through
the captured graph + driver scheduling. No amount of in-kernel
optimization can recover that floor.

**Implication for the rest of the fix list:** anything that reduces
per-call kernel time without reducing CALL COUNT or BATCHING ROWS will
hit the same wall. Real wins on this model need either:
  (a) fewer dispatches per token (fusion),
  (b) larger grids per dispatch (multiple rows per kernel, parallel_blocks
      across K dim),
  (c) skipping host-side dispatch entirely on stable forwards (the
      ggml-style fast path).

## Fix list (current priority order)

0. ✅ DONE — Printer-thread split in `examples/gemma_chat_repl/main.odin`
   (session 2026-05-09). Closed the gap to the reference target on its
   own. See "What's in tree".
1. ✅ DONE — `rmsnorm.cu` warp-shuffle port. Marginal; not load-bearing.
2. **Verify the 130 tok/s reference with `llama-bench`.** We're
   currently matching it but the reference number itself is still
   unverified. If llama.cpp on this model is actually faster than 130,
   there's headroom; if slower, we may already be ahead. One-time
   measurement.
3. **Speculative wins** — see "Speculative wins beyond the reference
   target" above. parallel_blocks for attention, offline-compiled
   cubin for hot kernels, ggml-style stable-graph fast path. None
   urgent.

## Things tried that didn't move the headline (don't redo)

- Combine `auto_graph` with `enable_timing`. Asserted off in code;
  `cuEventRecord` is not stream-capturable.
- "Skip cuGraphExecUpdate" fast path while still capturing every forward
  (Step A in prior notes). The capture/recording is most of the host
  work, not the update. Skipping just the update is a no-op. The right
  thing is to skip the whole forward dispatch — see fix #4.
- "Bypass forward_cached with stable-pinned host staging" (Step B in
  prior notes). Regressed because it didn't actually skip the dispatch
  loop in a clean way; needs the proper backend-capability for it.
- `rmsnorm + quantize_q8_1` or `rmsnorm + mul_mat` fusion. ggml does
  neither.
- Folding K/V cache writes into `attention_cache_vec_bf16` without
  cooperative-groups grid sync.
- Skipping the warmup forward before enabling capture — cuBLAS first-
  call algo selection isn't capturable cold (`auto_warmup_done` exists
  for this).

## Open questions

- Confirm Gemma E4B `q4_k`/`q6_k` weight assignment in the GGUF loader.
  `linear_q6_k_mmvq` call count is 33/tok which is more than expected
  for a standard Q4_K_M layout. If some Q6_K weights should be Q4_K,
  the fused gate+up+GEGLU path applies to more layers than today.
- Do we still need fp32 activations now that MMA F16 is abandoned? The
  whole pipeline could go back to bf16 activations and skip the
  cast_bf16_to_f32 / quantize_q8_1_f32 dtype rotations.

## Session notes (chronological, latest first)

Keep this short — only entries that change a future session's
priorities or correct a wrong assumption. Older context lives in git.

### Session 2026-05-09 (later)
- **Implemented the printer-thread split.** Added `Printer` struct +
  `core:sync/chan.Chan(string)` (cap 256) + `core:thread` worker in
  `examples/gemma_chat_repl/main.odin`. Decoder clones the delta,
  `wait_group_add(1)`, `chan.send`. Worker recv/print/flush/delete/
  `wait_group_done`. Decoder drains via `wait_group_wait` before any
  direct `fmt.println`. Channel closed + thread joined on exit.
- **Closed the gap to the reference target.** Three-run averages with
  `--temperature 0`:
  - File redirect: 97 → ~131 tok/s
  - Pipe (`| grep`): 64 → ~129 tok/s
  Destination no longer matters. Even file-mode `os.flush` was
  costing ~3 ms/tok, not just terminal/pipe. The "97 tok/s GPU-bound
  baseline" assumption from earlier in this session was wrong.
- **Replaced selection-sort top-K in `sample_next` with a min-heap.**
  After the printer fix, default-temperature decode (`--top-k 40` on
  vocab=262144) was capped at 67 tok/s by the O(n*k) sampling sort —
  10.5M comparisons + a 2 MB `indices` allocation per token. Min-heap
  top-K + heap-sort to descending order is O(n log k) and only
  allocates `k` ints. Default-temperature decode now matches
  `--temperature 0` at ~131 tok/s. The PERF_NOTES test was hiding
  this because it always used `--temperature 0`, which goes through a
  fast O(n) argmax path.
- **Demoted the rest of the fix list to speculative.** With us at the
  reference target, parallel_blocks / offline cubin / stable-graph
  fast path are no longer closing a known gap. Their leverage
  estimates in the old "Root-cause analysis" section were sized
  against the prior 97 tok/s baseline and are stale.

### Session 2026-05-09 (earlier)
- **Discovered the chat-REPL flush bottleneck.** Same binary giving 64
  vs 97 tok/s depending only on stdout target. Per-token flush stall
  through `os.flush(os.stdout)` while `clear()` synchronises the
  stream at the start of each forward.
- **Ported warp-shuffle to `kernels/rmsnorm/rmsnorm.cu`.** Verified
  the kernel is correct (deterministic across runs); fp32 reduction
  reorder shifts the greedy decode after ~10 tokens but kernel math
  is sound. Per-call -0.6 µs, ~0.1 ms/tok, no headline movement.
  Lesson: small per-row kernels are launch-bound, not compute-bound.
  In-kernel optimisations are mostly noise; need fusion / bigger grids
  / skip-dispatch for real wins.
- **Identified ggml's actual stable-graph fast path** in detail. ggml
  skips stream capture, cgraph evaluation, AND `cuGraphExecUpdate`
  when properties unchanged — they call `cudaGraphLaunch` directly.
  Our prior "Step A" only skipped the update; the recording itself was
  what cost.
