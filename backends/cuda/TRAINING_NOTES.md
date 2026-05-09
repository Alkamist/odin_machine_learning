# CUDA training notes

Status of training on the CUDA backend. Re-read this before picking up
new training work.

## TL;DR

F32 Llama training works end-to-end. Shakespeare converges (loss
4.67 → 2.13 over 1400 steps at ~24k tok/s on a 3090 Ti). Bf16
mixed-precision and Gemma training are both still to do.

The reference recipe for the user is bf16 weights + bf16 activations
+ f32 Adam moments. Phase 1 (this session) shipped f32-only because
the existing `ml.alloc` API can't express per-buffer-kind dtypes.

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
- `examples/tinystories/main.odin`, `examples/dac_lm/main.odin` —
  imports switched from the deleted `vulkan` backend to `cuda`. Not
  tested end-to-end this session; should run since they use the same
  Llama op set as shakespeare.

## What's left

### Phase 2 — bf16 mixed-precision (the original ask)

The user wanted bf16 weights + bf16 activations + **f32 Adam moments**.
That requires per-buffer-kind dtypes within a single tensor (Data and
Gradient stay bf16, but Adam_M and Adam_V are f32). Today
`ml.alloc(type, ...)` uses the same byte_count for every buffer kind.

Concrete work:

1. **API change in `ml.alloc`.** Easiest: hardcode that `Adam_M` and
   `Adam_V` are always F32 regardless of the tensor's primary dtype.
   The byte_count loop in `ml.odin:217` becomes per-kind. Then the
   buffer for those kinds is sized as `count * 4`.
2. **CPU backend helpers.** `data()`, `gradient()`, `adam_m()`,
   `adam_v()` in `backends/cpu/cpu.odin` all return `[]f32`. With bf16
   data/gradient buffers they'd be reading 2 bytes per element as
   4 bytes. Either return `[]byte` and let callers reinterpret, or
   add bf16-aware variants.
3. **Adam kernel for bf16.** `adam_f32.cu` becomes `adam_bf16.cu`:
   - reads bf16 weight, bf16 grad
   - reads f32 m, f32 v (separate dtype!)
   - computes the update in f32
   - writes back bf16 weight, f32 m/v, zeroes bf16 grad
4. **Backward kernels in bf16.** Each existing `*_back_f32.cu` would
   need a bf16 sibling, OR the kernels can read bf16 input and write
   f32-into-bf16 output. The `+=` accumulation for gradients is
   precision-sensitive — within a single backward pass we may write
   to the same gradient element multiple times (e.g. residual
   connections; weight shared between matmul calls). Bf16 grad
   accumulation will be lossy for Adam's `period` accumulation across
   multiple forward/backward passes too. Worth measuring whether the
   model still converges; if not, fall back to f32 grads.
5. **`llama.make` dtype param.** Today llama.make hardcodes `.F32` for
   every weight tensor. Add a `dtype: ml.Data_Type = .F32` parameter
   like `gemma.make` already has. Then training examples can pick
   their precision.

### Phase 3 — Gemma training

Gemma's forward uses additional ops beyond Llama:
- `ml.gelu_mul` (PLE block)
- `ml.tanh` (final softcap)
- `ml.slice_trailing` (per-layer inputs)
- `ml.add_rmsnorm` (fused residual + norm)
- `ml.rmsnorm_rope` (fused norm + rope)

`gemma.make` already has a `for_training: bool` argument that
allocates the Gradient/Adam_M/Adam_V buffers, so the model side
doesn't need changes — it's purely about backward kernel coverage.

What's needed:

1. **Tanh backward.** Standard `(1 - tanh^2(x)) * dy`. Trivial.
2. **Slice_Trailing backward.** Output gradient is dy at offset
   `[start, end)`; scatter it back into the wider input gradient.
   Trivial.
3. **Gelu_Mul.** Two options:
   - Add a backward kernel for the fused op, OR
   - Decompose at the `ml.gelu_mul` call site for training: emit a
     `gelu` op + `mul` op instead. Requires a `gelu` (non-fused)
     backward kernel which doesn't exist yet either, but it's a
     standard unary backward.
   - The decomposition path is cleaner and parallels how
     `linear_q4_k_gate_up_geglu` decomposes when the capability bit
     is missing.
4. **Add_Rmsnorm.** Same options. Decomposition is `ml.add` then
   `ml.rmsnorm` — both have f32 backward now. Just need to gate
   the fused emission on a backend capability bit.
5. **Rmsnorm_Rope.** Same options. Decomposition is `ml.rmsnorm`
   then `ml.rope` — both have f32 backward.

The capability gating model is already in place. CUDA's
`Backend.capabilities` advertises `.Linear_Q4_K_Gate_Up_Geglu` and
`.Rmsnorm_Rope_Write_Cache`; for training we'd add capability bits
for the other fused ops and have CUDA either advertise them with
backward kernels or omit them so `ml.gelu_mul` etc. decompose.

### Phase 4 — pretrained weights for fine-tuning

`gemma.load_safetensors` and `gemma.load_gguf` exist, but:
- The GGUF loader produces Q4_K quantized weights that are
  forward-only (no Linear_Q4_K backward exists on either backend).
  Fine-tuning from GGUF would need either dequantization on load or
  a Linear_Q4_K backward.
- The safetensors loader produces bf16 weights. With Phase 2's
  mixed-precision support, fine-tuning from safetensors should work
  as soon as the backward kernels are bf16-capable.

### Phase 5 — fast cached sampling (optional)

`shakespeare.sample()` uses non-cached `llama.forward` because the
inference cleanup made `attention_with_cache` bf16-only. To get fast
KV-cached generation back, restore f32 variants of:
- `attention/attention_cache_vec_f32.cu`
- `attention/cache_write_f32.cu`

These were deleted in the bf16 inference cleanup. Easy to bring
back from git history if desired. Not on the critical path.

## Calibration notes

### Loss curve (shakespeare, current f32 path)
- step 50:  4.67
- step 500: 2.66 (val 2.56)
- step 1000: 2.30 (val 2.31)
- step 1400: 2.13

This is the post-bug-fix curve (the temp-allocator bug in
`sample()` — see "Things tried that didn't move the headline" — was
sampling-only, training itself was correct from the start).

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
