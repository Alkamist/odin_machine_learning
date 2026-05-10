# CUDA training notes

Status of training on the CUDA backend. Re-read this before picking up
new training work.

## TL;DR

Both F32 and bf16 training work end-to-end at the same speed.
Shakespeare convergence matches:

- **F32 tiny Gemma**: loss 4.93 → 1.91 over 1500 steps at ~20k tok/s
- **bf16 tiny Gemma**: loss 4.93 → 2.22 at step 1000, ~21k tok/s.
  Slightly slower convergence per step than F32 (bf16 precision cost),
  but actual wall-clock similar.
- **F32 Llama**: loss 4.67 → 2.13 over 1400 steps at ~24k tok/s

The recipe is bf16 weights + bf16 activations + **F32 gradients** +
F32 Adam moments. The "F32 grads" piece is what catches F32 speed:
native `atomicAdd<float>` instead of CAS-loop emulation for bf16.
`ml.buffer_dtype` hardcodes everything except `.Data` to F32.

Decomposition trigger for fused ops (`gelu_mul`, `add_rmsnorm`,
`rmsnorm_rope`) fires whenever `_current_ctx.clear_flags` lacks
`.No_Gradients` (i.e. training mode), so bf16 training uses unfused
ops with backward kernels, while bf16 inference still uses the fused
path.

## What's in tree

### Optimizer
- `kernels/optimizer/adam_f32.cu` — one thread per parameter element.
  Reads f32 weight + f32 grad + f32 m/v, applies the standard Adam
  update, zeroes the gradient. Mirrors `backends/cpu/cpu.odin:update`.

### Forward + backward kernels (all f32)
- `silu/silu_f32.cu`, `silu_back_f32.cu`
- `cross_entropy/cross_entropy_f32.cu`, `cross_entropy_back_f32.cu`
- `mul/mul_back_a_f32.cu`, `mul_back_b_f32.cu` (atomicAdd on the b-side
  for the broadcast case)
- `select/select_back_f32.cu` (atomicAdd scatter)
- `rope/rope_back_f32.cu` (rotation transpose, recomputes cos/sin
  on the fly)
- `rmsnorm/rmsnorm.cu` (rewritten: takes f32 weight, writes per-row
  `rstd` for backward) + `rmsnorm_back_f32.cu`
- `attention/attention_train_f32.cu` — non-flash, materialises
  softmax_outputs to `Attention.softmax_outputs`. Caps T at 1024 due
  to `d_p_row[1024]` shared memory in backward.
- `attention/attention_train_back_f32.cu` — uses materialised softmax;
  one block per (kv_head, q_head_in_group, t_q).

### Backward dispatcher (`_backward` in `ops.odin`)
Currently routes: Add, Mul, Linear, Silu, Select, Rmsnorm, Rope,
Attention, Cross_Entropy. Anything else panics.

### Forward dispatcher
`Cross_Entropy` and `Silu` were missing entirely on CUDA before this
work — they're added too.

### Examples
- `examples/shakespeare/main.odin` — runs end-to-end on CUDA.
  `sample()` was rewritten to use non-cached `llama.forward` because
  `attention_with_cache` is bf16-only after the inference cleanup.
  Slower than KV-cached generation but fine for the small sample
  budget.
- `examples/tinystories/main.odin` — same fix applied to its
  `sample()`. Imports switched from `vulkan` to `cuda`.
- `examples/dac_lm/main.odin` — imports switched from `vulkan` to
  `cuda`, but its sampling code (`forward_cached` calls at
  `main.odin:278`, `:287`, `:417`, `:433`) still hits the bf16-only
  `attention_with_cache` and will crash. Not tested end-to-end.
  Either apply the same `forward` rewrite or wait until Phase 5
  restores f32 `attention_with_cache`.

## What's left

### Phase 2 — bf16 mixed-precision (DONE)

End-to-end bf16 training works on tiny Gemma. Loss tracks F32 with
~5% slower per-step convergence (bf16 precision cost), but wall-clock
is comparable to F32 because bf16 GEMMs hit Tensor Cores natively.

The architecture is **bf16 data + F32 grads + F32 optimizer**. The
F32-grad piece is the perf unlock:

- Native `atomicAdd<float>` instead of CAS-emulated bf16 atomics
- Both fast and numerically stable (gradient accumulation is the
  precision-sensitive step in Adam)
- 2x gradient buffer memory cost is fine for the QLoRA target where
  base weights are quantized & frozen so most params have no grad
  buffer at all

What landed:
- **`ml.buffer_dtype`**: returns F32 for every kind except `.Data`.
  Adam_M, Adam_V, Gradient all F32.
- **bf16 Adam kernel** (`kernels/optimizer/adam_bf16.cu`): reads bf16
  W + f32 grad + f32 m/v; writes bf16 W + f32 m/v; zeroes f32 grad.
- **Mixed-precision backward kernels** (read bf16 forward inputs +
  read/write F32 grads, internal fp32 accumulators):
  - `silu_back_bf16`, `gelu_bf16` + `gelu_back_bf16`, `tanh_back_bf16`
  - `mul_back_a_bf16` + `mul_back_b_bf16` (native `atomicAdd<float>`)
  - `select_back_bf16` (effectively identical to f32 — pure grad path)
  - `slice_trailing_back_bf16` (same)
  - `rmsnorm_back_bf16` (forward `rmsnorm_bf16` now writes rstd too)
  - `rope_back_bf16` (no forward data needed in backward)
  - `attention_train_bf16` + `attention_train_back_bf16`
  - `_cross_entropy_backward` stays F32 (loss is always F32)
- **`_linear_backward`** in bf16: cast f32 dy down to bf16 in a
  scratch buffer, then run two `cublasGemmEx` with A=bf16, B=bf16,
  Tensor-Core compute_type=F32, output F32. Single cast per backward,
  Tensor Cores handle the heavy lifting.
- **`_cast_backward`**: with F32 grads, the backward of any
  `cast_to(x, target)` is just `dx += dy` in f32 (forward cast
  direction is irrelevant for the gradient). Uses a single
  `cast_back_f32` kernel for all directions.
- **Frozen-scalar skip in `mul_backward`**: when b has no gradient
  buffer (const scalars like `embed_scale`), the b-side accumulation
  is skipped. Saves the redundant atomic.
- **Decomposition trigger** for `gelu_mul` / `add_rmsnorm` /
  `rmsnorm_rope` fires when training is active (clear_flags lacks
  `.No_Gradients`). bf16 inference still uses fused.
- **Attention forward dispatch in bf16** routes to
  `attention_train_bf16` (materialises softmax) when training, or
  `attention_bf16` (flash) for inference.

What's still optional:
1. **`llama.make` dtype param.** Today llama.make hardcodes `.F32`
   for every weight tensor. Add a `dtype: ml.Data_Type = .F32`
   parameter like `gemma.make` has, so Llama training examples can
   pick their precision.

### Phase 3 — Gemma F32 training (DONE)

Shipped. F32 Gemma training works end-to-end on tiny config
(4 layers, hidden=256, head_dim=64, vocab=256, seq_len=128). See
`examples/gemma_shakespeare/main.odin`. Converges 4.93 → 1.91 over
1500 steps at ~20k tok/s.

What landed:

- **Tanh backward.** `kernels/tanh/tanh_back_f32.cu`. Uses cached
  forward output (`y = tanh(x)`); `dx += dy * (1 - y^2)`.
- **Slice_Trailing backward.** `kernels/slice_trailing/slice_trailing_back_f32.cu`.
  Each output element maps to one input element so no atomics needed.
- **Gelu forward + backward.** `kernels/gelu/gelu_f32.cu` and
  `gelu_back_f32.cu`. Matches the CPU tanh-approximation
  formulation in `gelu_forward` / `gelu_backward`.
- **Decomposition for fused ops in F32.** `ml.gelu_mul`,
  `ml.add_rmsnorm`, and `ml.rmsnorm_rope` now check the input dtype
  at the call site and decompose to unfused ops when input is F32.
  Bf16 still uses the fused path (no backward needed for inference).
  This avoids writing fused-op backward kernels.
- **Per-layer-input frozen lookup.** Gemma's
  `embed_tokens_per_layer_bytes` is host-side bytes, not a Tensor.
  In training-from-scratch this is initialised to small normal
  values via `_fill_per_layer_bytes_normal` and not updated by
  Adam. Effectively a frozen learned embedding. Restructuring it as
  a trainable Tensor is its own follow-up; the model still learns
  fine without it being trainable.
- **Const-scalar gradient buffers.** `_make_const_scalar` (embed_scale,
  ple_token_scale, softcap, etc.) takes a `buffers: ml.Buffer_Set`
  param now. In training mode `make` passes `.Data + .Gradient` so
  `mul(x, scalar)` backward has somewhere to write. The scalars
  themselves are excluded from `gemma.update`, so they stay constant.
- **Gemma `randomize`, `update`, `copy` procs.** Mirror llama's,
  iterate the right tensors, skip kv-shared layers correctly.

### Phase 4 — QLoRA on Gemma E4B (DONE)

QLoRA training works: Q4_K-quantized base (frozen) + bf16 LoRA
adapters on attention projections, all gradients F32.

What landed:
- **`networks/lora/lora.odin`** — generic LoRA Adapter struct (A,
  B matrices + alpha/rank scale). `lora.apply(input, base_output,
  adapter)` augments any base linear's output with the adapter
  contribution. Standard QLoRA init: A ~ N(0, 0.02), B = 0.
- **Gemma + LoRA integration** — `LoRA_Config` with rank/alpha and
  per-target bit_set ({Q, K, V, O, Gate, Up, Down}). `gemma.make`
  with `lora_cfg=...` allocates the base with `.Data`-only buffers
  (frozen) and the adapters with full Adam state. `update_lora`
  steps only adapter params; `randomize_lora` initialises them.
- **Frozen-weight skip in linear/select/rmsnorm backwards** — when a
  weight has no Gradient buffer the dW path is skipped (no crash,
  no wasted compute). Same `if (dw) atomicAdd(...)` guard inside
  the rmsnorm CUDA kernel.
- **Q4_K / Q6_K linear forward for arbitrary M** — the existing M=1
  mmvq paths are preserved for inference. M>1 dequantizes the Q4_K
  / Q6_K weight to bf16 scratch (`dequantize_q4_k_to_bf16.cu`,
  `dequantize_q6_k_to_bf16.cu`) and runs a Tensor-Core bf16 GEMM
  with F32 accumulation/output.
- **Q4_K / Q6_K linear backward** — computes dx only (W is frozen
  by design): reuse cached dequantized W, cast f32 dy down to bf16,
  Tensor-Core GEMM. No dW.
- **Dequant cache** (`gctx.dequant_cache`) — keyed by source weight
  pointer, value is the bf16 scratch from the activation pool.
  Forward populates it; backward reads from it. Cleared in
  `clear()` along with the activation pool. Speedup: ~4x measured
  on E4B QLoRA (158 → 642 tok/s at step 30, still ramping when
  measured).
- **`examples/gemma_qlora`** — load E4B from GGUF, attach LoRA
  adapters on attention projections, train on a tokenized text
  corpus. Saves adapter weights as a simple binary at the end
  (LORA0001 magic, per-layer (rank, in, out, A bytes, B bytes)).

Memory budget on a 3090 Ti for E4B QLoRA:
- Q4_K base weights: ~2 GB (kept in 4-bit)
- bf16 embeddings (lm_head tied): ~1.34 GB
- LoRA adapters + F32 grad + F32 Adam: ~30-50 MB
- Activations + dequant scratch: ~3-5 GB at seq_len=256
- Total: ~7-9 GB. Comfortable headroom on 24 GB.

### Phase 5 — fast cached sampling (optional)

`shakespeare.sample()` uses non-cached `llama.forward` because the
inference cleanup made `attention_with_cache` bf16-only. To get fast
KV-cached generation back, restore f32 variants of:
- `attention/attention_cache_vec_f32.cu`
- `attention/cache_write_f32.cu`

These were deleted in the bf16 inference cleanup. Easy to bring
back from git history if desired. Not on the critical path.

## Calibration notes

### Loss curve (shakespeare, llama, f32 path)
- step 50:  4.67
- step 500: 2.66 (val 2.56)
- step 1000: 2.30 (val 2.31)
- step 1400: 2.13

This is the post-bug-fix curve (the temp-allocator bug in
`sample()` — see "Things tried that didn't move the headline" — was
sampling-only, training itself was correct from the start).

### Loss curve (shakespeare, tiny gemma, f32 path)
- step 50:   4.93
- step 500:  2.39 (val 2.35)
- step 1000: 2.12 (val 2.07)
- step 1500: 1.91 (val 1.96)

Tiny config: 4 layers, hidden=256, head_dim=64 (sliding=full), 4 q
heads, 2 kv heads, vocab=256, seq_len=128, sliding_window=64,
final_logit_softcap=0, ple_dim=64. ~3.4M parameters. ~20k tok/s.

### Loss curve (shakespeare, tiny gemma, bf16 path with F32 grads)
- step 50:   4.96
- step 500:  2.43 (val 2.40)
- step 1000: 2.22 (val 2.17)

Same config as F32, just `gemma.make(cfg, .Bf16, for_training=true)`.
~21k tok/s on the 3090 Ti — matches F32 wall-clock. Per-step
convergence is ~5% slower than F32 (bf16 precision cost on weight
updates).

### Speed
~24k tokens/sec end-to-end on a 3090 Ti for the shakespeare config
(6 layers, 6 heads, head_size=64, embed=384, seq_len=128). Most
likely bottlenecks:
- Attention forward materialises softmax_outputs as f32, three passes
  over the row each step. A flash-attention-style training kernel
  would drop the materialisation.
- Adam is a simple per-element kernel; for very small param tensors
  it's launch-bound.

Neither is urgent; correctness > speed for the training story.

## Things tried that didn't move the headline (don't redo)

- **Storing `tokens` for sampling in the temp_allocator while
  `defer free_all(context.temp_allocator)` lives in the loop body.**
  Memory gets wiped between iterations; the next forward sees a
  corrupted token sequence and produces gibberish. Use the default
  allocator with explicit `defer delete(tokens)`.
