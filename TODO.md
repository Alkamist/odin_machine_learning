# TODO / future work

Notes accumulated during the PyTorch-parity / GPU-optimization session.
Ordered roughly by leverage, not by difficulty.

## North star: what this library should run

Goals: SOTA transformer LLMs (Gemma 4 class), game-playing agents,
audio processing (ASR, embeddings), and audio generation. The Gemma 4
target is the most demanding and is largely a superset of what the
others need on the architecture side, so the transformer stack is the
trunk and everything else branches off it.

Cross-cutting capabilities the current library is missing for these
goals (in addition to the dtype/perf items below):

- **GQA + RoPE as an op + sliding-window attention.** Required by
  every modern transformer, not just Gemma 4. RoPE must be a real op
  with config (theta, p-RoPE variant), not baked into `attention`.
- **KV-cache inference path.** `attention_decode` (single-token Q
  against cached K/V) plus a way to grow K/V buffers in place. Without
  this you can train but cannot "run" a model.
- **Conv1d / Conv2d ops.** Audio frontends, vision encoders, CNN
  trunks for AlphaZero-style agents. Examples currently sidestep this
  by flattening MNIST.
- **FFT / mel-spectrogram preprocessing.** CPU-only, lives outside the
  autograd graph. Gates all audio work.
- **Autoregressive generation loop.** Categorical/Gumbel sampling +
  KV-cache step. Needed for text, audio tokens, Decision Transformer
  action rollouts.
- **VQ / residual VQ with straight-through gradient.** Neural audio
  codecs (EnCodec/SoundStream) and discrete-latent world models.

Suggested phasing:
1. Dtype foundation (phase 1 below).
2. Modern attention surface: GQA, RoPE op, sliding window, KV-cache
   decode. Makes the library "Gemma-shaped."
3. Conv1d/2d + FFT. Unlocks audio and vision frontends.
4. End-to-end checkpoint load: pull a small Gemma 4 variant (E4B)
   from HF, run inference, match logits. This is the forcing function
   that surfaces every remaining gap.
5. Branch into one of {audio generation, world-model RL, MoE
   training} — roughly equal effort off the same trunk.

## Multi-dtype foundation (precondition for serious LLM work)

The project goal is to infer and train Gemma-class LLMs. That requires BF16
(or at minimum FP16) tensors end-to-end: BF16 weights and activations,
FP32 master weights for the optimizer, and tensor-core matmul on top.
Without dtype support the project caps at "small models in FP32," and the
coopmat shader work below would need to be redone once buffers gain types.

### Status (as of last session)

**Phase 1 (foundations) — DONE.** Done across `ml.odin`,
`backends/cpu/cpu.odin`, `backends/gpu/{backend,buffer,gpu,ops}.odin`,
plus `tests/dtype_roundtrip/`.

- `Data_Type` extended to `{F32, F16, Bf16}`. Note: `Bf16` is
  `distinct u16`, with `bf16_from_f32` (round-to-nearest-even,
  NaN-preserving) and `bf16_to_f32` helpers in `ml.odin`. Style choice:
  the type is `Bf16` (not `BF16`); enum cases stay capitalized as
  `.F32`/`.F16`/`.Bf16`.
- Backend buffer interface is now byte-oriented:
  `buffer_alloc(byte_count: int, persist, loc)`,
  `buffer_get/set(buffer, []byte, loc)`. The CPU helpers
  (`data`/`gradient`/`adam_m`/`adam_v`) now slice via `[^]f32` cast +
  `t.count` since the slice header carries byte count.
- `alloc`/`zeros`/`make` carry `type: Data_Type = .F32`. `zeros_like`
  and `_zeros_*` helpers preserve source type. `alloc` rounds byte
  count up to a 4-byte multiple — required because the GPU bf16/f16
  shaders pack two halves per `uint`, so an odd element count must be
  writable through the trailing uint.
- Public typed APIs (`set_data`/`get_data`/`get_gradient`,
  `upload_tensor`/`download_tensor` in the GPU backend) keep `[]f32`
  signatures with an `assert(t.type == .F32)`. Raw byte access:
  `set_data_bytes`/`get_data_bytes`. `fill_normal`/`fill_value` switch
  on `t.type` and emit F32/F16/Bf16.
- `backward()` currently asserts the loss tensor is F32.

**Phase 2 (`cast_to` op) — DONE.** `cast` is reserved in Odin so the
proc is `ml.cast_to(input, target_type)`. The op variant is `Cast{}`
(empty marker — src and dst types are read off `op.input.type` /
`op.output.type`).

- CPU: full support for all `{F32, F16, Bf16}` pairs, forward and
  backward. Implementations: `cast_forward` / `cast_backward` in
  `cpu.odin`, with shared `_cast_bytes` and `_cast_bytes_accumulate`
  helpers. Backward accumulates (`+=`) into `input.gradient`.
- GPU: F32 ↔ Bf16 only. Four shaders under
  `backends/gpu/shaders/cast_*.comp` (with built `.spv`):
  `cast_f32_to_bf16`, `cast_bf16_to_f32`, plus `_back` variants used
  by `cast_backward`. Same-type cast forward falls through to `_copy`
  (a vk buffer copy); same-type backward currently panics (intentional
  — flag if it ever shows up). The shaders pack two bf16/uint and
  dispatch one thread per pair; no special Vulkan extensions needed.
- F16 GPU support was **deliberately deferred** — Gemma 4 is bf16, so
  this can wait until something asks for it.
- Tests: `tests/dtype_roundtrip/main.odin` covers F32/Bf16/F16 byte
  round-trip and `cast_to` forward + backward on both backends. All
  pass. `tests/gpu_unified_check` (the existing F32 op suite) still
  passes — F32 path is untouched.

**Phase 2.5 (`add` dtype-generic on Bf16) — DONE.** `ml.add` now asserts
`a.type == b.type`. CPU `add_forward`/`add_backward` switch on dtype with
f32-accumulator paths for Bf16 and F16 (f32 instead of f16 for the bf16
path because casting bf16 → f16 loses range, and CPUs widen f16 to f32
for arithmetic anyway). GPU has three new packed-bf16 shaders
(`add_bf16`, `add_back_a_bf16`, `add_back_b_bf16`) that thread per output
pair so each thread writes a whole packed uint and avoids partial-uint
races. Test added in `tests/dtype_roundtrip` (broadcast forward + dx/db
backward) covering both backends.

**Phase 3a (naive Bf16 `linear` forward) — DONE.** Took the contained
"BF16 matmul plumbing" half of phase 3 first, before the coopmat work.
`ml.linear` now asserts `input.type == weight.type`. CPU `linear_forward`
splits into `_f32` and `_bf16` paths (bf16 dot product accumulates in
f32, stores result via `bf16_from_f32`); `linear_backward` panics on any
non-F32 type. GPU has a new `linear_bf16.comp` mirroring `linear.comp`'s
tile structure but with packed-uint loads and bf16 output writes — each
thread owns a TM=4 × TN=4 sub-tile, writes 2 whole packed uints per row,
which requires `input_size` and `output_size` to be even (asserted at
dispatch). Tests added covering both a small (M=4, K=8, N=6) and a
multi-tile (M=64, K=64, N=128) shape with values chosen so the bf16
result is bit-exact against the f32 reference. `linear_backward` panics
on bf16 in the GPU backend too — that ships with phase 3b.

**Phase 3b (Bf16 `linear` forward via cooperative matrix) — DONE.** RTX
30+ exposes `VK_KHR_cooperative_matrix` + `VK_KHR_shader_bfloat16` with
`shaderBFloat16CooperativeMatrix = true` and a 16/16/16 bf16/bf16/fp32
config; we now query that at device init, set `_gpu.coopmat_bf16`, and
enable `cooperativeMatrix`, `shaderBFloat16Type`,
`shaderBFloat16CooperativeMatrix`, and `storageBuffer16BitAccess` in the
device feature chain.

The vendored `vendor:vulkan` binding doesn't yet expose the bf16
extension, so `gpu.odin` carries hand-written constants
(`KHR_SHADER_BFLOAT16_EXTENSION_NAME`, `COMPONENT_TYPE_BFLOAT16_KHR =
1000141000`, `Physical_Device_Shader_Bfloat16_Features_KHR`) until the
binding catches up.

`linear_bf16_coopmat.comp` is intentionally minimal: one subgroup per
workgroup, one 16×16 output tile per subgroup, K loop calls
`coopMatMulAdd` directly — no shared-memory staging or per-warp tiling
yet. Buffers are typed `bfloat16_t`; W is loaded ColumnMajor to take its
[N, K] storage as B's transposed [K, N]. Backend dispatches coopmat when
`_gpu.coopmat_bf16` is true and all three dims are multiples of 16; else
falls back to the naive `linear_bf16.comp`. The dtype_roundtrip
multi-tile test (M=64, K=64, N=128) exercises the coopmat path; the
small test (M=4, K=8, N=6) exercises the fallback.

**Phase 3c (Bf16 `linear_backward`) — DONE.** CPU `linear_backward`
splits into f32 and bf16 paths (bf16 dot product accumulates in f32 like
the forward). GPU has new `linear_back_input_bf16.comp` and
`linear_back_weight_bf16.comp` — naive thread-per-(c, k_pair) and
thread-per-(o, k_pair) respectively, each thread writing a whole packed
uint to dodge partial-uint races. K (input_size) must be even (asserted
at dispatch). Coopmat backward is deferred — the existing F32 backward
shaders are also naive, so this matches the project's "tiled GEMM is
overkill until backward becomes a profile blip" rule.

**Phase 3d (Bf16 `batched_matmul` fwd + bwd) — DONE.** Same shape as
linear: `ml.batched_matmul` asserts `a.type == b.type`. CPU has bf16
`*_forward_bf16` and `*_backward_bf16`. GPU has three new bf16 shaders
(`batched_matmul_bf16`, `batched_matmul_back_input_bf16`,
`batched_matmul_back_weight_bf16`), all naive thread-per-output-pair
with N (and K for backward) required even at dispatch. Coopmat for
batched_matmul is deferred to the perf pass alongside coopmat for
linear backward and attention.

## Remaining bf16 work for an end-to-end transformer step

Ordered by what unlocks the most when done:

1. **Bf16 `attention` (fused fwd + 3 bwd shaders).** This is the next
   big chunk — biggest single op, biggest blocker to a bf16 transformer
   step. The forward shader's `softmax_outputs`, `lse`, `d_p_scratch`,
   `d_acc` scratch tensors should stay F32 (that's the "internal
   accumulators are F32" rule). The Q/K/V loads from the packed
   `[tokens, 3*embedding]` input become shift-extracts on bf16 storage,
   and the output write packs back to bf16. Backward's three shaders
   follow the same pattern.

2. **Bf16 element-wise/normalization ops.** Long but mechanical list,
   all following the `add` pattern (thread-per-output-pair, f32 inner
   compute, packed bf16 storage):
   - `mul`, `sub`, `div` — same shape as `add`, can almost copy-paste.
   - `gelu`, `silu`, `relu`, `sigmoid`, `tanh`, `exp` — pure unary, even
     simpler.
   - `layernorm` — needs F32 mean/rstd scratch (already F32 on the GPU
     forward shader; just preserve that).
   - `softmax`, `log_softmax`, `entropy` — same pattern, F32 reductions
     internally.
   - `rope`, `causal_mask`, `permute`, `concat`, `slice`, `mean` —
     trivial bf16 loads/stores, no math.

3. **First bf16 transformer parity test.** Mirror the library's
   transformer block in PyTorch with bf16 weights/activations, sync
   weights in, compare loss curves the same way `mlp_train` does.
   That's the forcing function the existing F32 transformer parity
   test already serves and the cleanest way to surface remaining gaps.

4. **Coopmat perf pass.** The naive bf16 shaders win on storage but
   the coopmat shader as written is a minimum-viable subgroup-per-tile
   layout — leaves most tensor-core throughput on the floor. Add BM/BN
   per workgroup, multiple subgroups, shared-memory K-tile staging,
   per-subgroup output tiles. Then port the same pattern to backward
   (linear, batched_matmul) and to attention. This is the move from
   "correct" to "fast."

5. **Mixed-precision recipe in `examples/`.** End-to-end demo of the
   standard "FP32 master weights, BF16 forward/backward, FP32 optimizer
   state" pattern on a small transformer. Deliverable for "this library
   is a real bf16 stack."

### Open / next phases

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
