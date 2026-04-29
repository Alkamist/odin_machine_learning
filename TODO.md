# TODO / future work

Notes accumulated during the PyTorch-parity / GPU-optimization session.
Ordered roughly by leverage, not by difficulty.

## Multi-dtype foundation (precondition for serious LLM work)

The project goal is to infer and train Gemma-class LLMs. That requires BF16
(or at minimum FP16) tensors end-to-end: BF16 weights and activations,
FP32 master weights for the optimizer, and tensor-core matmul on top.
Without dtype support the project caps at "small models in FP32," and the
coopmat shader work below would need to be redone once buffers gain types.

**Existing scaffolding (`ml.odin`):**

- `Data_Type :: enum u8 { F32 }` and `data_type_size`. Single-variant
  today, ready to extend.
- `Tensor.type: Data_Type` is on every tensor. `alloc` hardcodes `.F32`.
- `Backend.buffer_alloc(len, ...)` takes element count, not bytes.
- `Backend.buffer_get/set` take `[]f32` — only FP32 round-trips today.

**Recommended rollout (do in this order, one phase per session):**

1. **Foundations.** Extend the enum to `{F32, F16, BF16}`. Switch
   `buffer_alloc` to bytes (or `(len, type)`), and `buffer_get/set` to
   `[]byte` with thin `tensor_set_f32`-style helpers on top. Add `type`
   parameter to `alloc`/`zeros`/`zeros_like`. Verify a BF16 tensor
   allocates and round-trips on both CPU and GPU backends with no ops
   defined yet.

2. **`cast` op.** Forward + backward, both backends. Smallest non-trivial
   dtype-aware op; proves the dispatch shape. Backward casts the gradient
   back to the source dtype. Trivial in CPU, one shader on GPU per
   `(src, dst)` pair (or one shader with a push-constant for type).

3. **First coopmat shader: BF16 `linear` forward.** Use
   `GL_KHR_cooperative_matrix` + `GL_EXT_shader_explicit_arithmetic_types_float16`
   compiled with `--target-env=vulkan1.2`. Backend dispatches based on
   `op.input.type`: F32 → existing FP32 SIMT shader, BF16 → coopmat
   shader. RTX 30+ exposes `cooperativeMatrix = true` and
   `shaderBFloat16CooperativeMatrix = true` already; query
   `vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR` at init to confirm
   a usable `(M,N,K,types)` config and fall back gracefully on
   unsupported devices. Update Vulkan device init to enable
   `VK_KHR_cooperative_matrix`, `shaderFloat16` / `shaderBFloat16`,
   `storageBuffer16BitAccess`.

4. **Roll out coopmat to the rest.** `linear` backward (input + weight),
   then `batched_matmul` forward + backward, then attention (which uses
   `batched_matmul` internally — coopmat there is the real win). Then
   the simple element-wise ops (`add`, `mul`, `gelu`, `layernorm`,
   activations) become dtype-generic — these don't need coopmat, they
   just need to read/write the right number of bytes per element.

5. **Mixed-precision recipe in `examples/`.** Show the standard "FP32
   master weights, BF16 forward/backward, FP32 optimizer state" pattern
   end-to-end on a small transformer. This is what every modern LLM
   training stack does and is the actual deliverable for "serious ML
   work."

**Design decisions to lock in at phase 1:**

- **Op dtype mixing.** When `linear(x: F32, w: BF16)` is called: error,
  not auto-cast. Forces explicit `cast` ops, keeps optimization tractable.
- **Output dtype.** Compute ops (`linear`, `add`, etc.) preserve input
  dtype. Internal accumulators in shaders are FP32 regardless (this is
  what tensor cores do anyway).
- **Optimizer state dtype.** Adam M/V stay FP32 even when the parameter
  is BF16. Hardcode in the optimizer; don't try to make `Buffer_Kind`
  per-buffer-typed.
- **Mixed precision (master weights).** Don't build this into the core
  Tensor — handle it at the example level by keeping two tensors
  (FP32 master, BF16 compute view) and casting each step. Simpler
  primitives, same end result.

## GPU performance

- **Flash Attention v2 — Q-tiled forward and tiled backward.** Forward now
  uses online softmax with K streamed in tiles of `BC=64`, removing the
  `MAX_T=4096` cap and the `scores[MAX_T]` shared array, but still runs one
  workgroup per `(head, query)` (BR=1). Real FA2 also tiles Q (BR>1) so K and
  V are reused across the whole BR×BC score block. Backward is still 3
  kernels (D-precompute, dKV, dQ) with no Q/K tiling. Tiling both should
  pull `attention_causal` from ~3x off cuDNN toward ~1.5x. Reference: ggml's
  `flash_attn.comp` and the Tri Dao FA2 paper.

- **Tensor-core matmul via VK_KHR_cooperative_matrix.** The current `linear`
  shader is FP32 SIMT. RTX 30+ tensor cores (BF16, FP16) deliver 4-8x more
  FLOPs and `cooperative_matrix` is the Vulkan path to them. Closes the
  remaining `linear_fwd` gap (~5x off cuBLAS). **Blocked on the multi-dtype
  foundation above** — coopmat needs BF16 buffers to be the natural input
  type; doing it on FP32 buffers means writing the conversion shim twice.
  Reference: ggml's `mul_mm_cm2.comp`.

- **Larger-shape GPU bench coverage.** Current speed bench uses small shapes
  where Vulkan's per-dispatch overhead dominates. Add benches at training-
  realistic shapes (e.g. transformer step at seq=512, embed=768) to see
  steady-state kernel performance, not launch overhead.

- **Cache descriptor sets in `_dispatch`.** Every `_dispatch` call does a fresh
  `vkAllocateDescriptorSets`, which is the per-shader floor (~50-100 us based
  on the small-op timings). Caching by `(pipeline, buffers)` would shave that
  for repeated identical dispatches. Broad benefit, no single op moves
  dramatically.

- **GPU-side fill kernels.** `ml.fill_value` and `ml.fill_normal` currently
  build the buffer on the CPU and do a synchronous upload, which forces a
  `vkQueueWaitIdle` mid-frame. A `fill_constant.comp` and a Philox-style
  `fill_normal.comp` would keep initialization on-device.

## GPU robustness

- **Hardcoded `head_size <= 256` in attention backward shaders.** The shared
  memory caches (`k_shared`, `v_shared`, `q_shared`, `do_shared`) are fixed
  `float[256]`. Asserted at dispatch time but should either grow dynamically
  via specialization constants or auto-fall-back to a tiled variant.

- **`head_size > WG=64` in attention backward kernels.** Per-thread d_K/d_V
  accumulators only handle one `d` element each. Heads larger than 64 would
  drop the rest. Either fix the loop or use specialization constants.

- **Vulkan specialization constants for shader tile sizes.** This session's
  `LINEAR_LOCAL_X` / `TILE_M` mismatch bug came from having a tile size in two
  places (the shader and `ops.odin`). Specialization constants make the shader
  the single source of truth. ggml does this throughout. Bigger refactor but
  removes a whole bug class.

## CPU performance

- **Real GEMM (or link BLAS) for `linear` and `batched_matmul`.** CPU
  `linear_fwd` is 6-8x off MKL/OpenBLAS at the bench shape. Fastest path is
  linking OpenBLAS for `cblas_sgemm`; afternoon of work, becomes as fast as
  PyTorch on matmul, costs a vendor dependency. Hand-rolled tiled GEMM is
  weeks-to-months for a competitive one.

- **Switch CPU attention from `softmax_outputs[H, T, T]` to LSE-only.** Current
  CPU `attention` saves the full softmax matrix for backward (~2 MB per call
  at the bench shape). Flash Attention v2 saves only `lse[H, T]` (~8 KB) and
  recomputes attention weights in backward. Same algorithm we use on GPU
  already; would unify the two scratch layouts and matter at long sequence
  lengths.

## Test coverage

- **End-to-end transformer parity test.** This session's `linear` bug (M > TILE_M
  rows uninitialized) wasn't caught by op-level parity tests because every
  `linear_*` test had M <= 32. A small "transformer training step" parity test
  (mirror the library's transformer block in PyTorch, sync weights, compare
  loss curves like `mlp_train` does) would catch op-composition bugs much
  faster than guessing which individual op might be wrong.

- **Larger-shape variants for every parity test.** Add a `_big` variant for any
  op with internal tile sizes. The new `linear_big` (M=64) is the template;
  worth adding for `attention`, `batched_matmul`, `softmax`, `layernorm` so
  any future tile-size tuning trips the suite immediately.

- **GPU stress test for races on `linear_backward` style accumulation.** The
  CPU `linear_backward` two-pass fix solved the race; the GPU version should
  be audited the same way and have a stress test that runs the same input
  many times and asserts bit-identity.

## Style / housekeeping

- The `attention_fused` vtable flag was removed once both backends had a fused
  Attention. If a third backend is added that wants the old composition path,
  resurrect `_attention_compose` (it's in git history) and the flag.

- `ml.fill_normal` / `ml.fill_value` use Odin's RNG (`core:math/rand`) seeded
  per process. PyTorch parity tests sidestep this by uploading PyTorch-
  generated weights. If you ever want bit-identical RNG between CPU and GPU
  paths, you'll need a deterministic in-library RNG.

- Bench `_checksum` downloads 1 float as a sync token. Cute, but a
  `ml.sync()` proc that just submits + waits without a download would express
  intent more clearly.
