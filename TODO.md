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

1. **Bf16 `attention` (fused fwd + 3 bwd shaders) — DONE.** `ml.attention`
   now allows F32 or Bf16 input; output type matches input, scratch
   (`softmax_outputs`, `lse`, `d_p_scratch`, `d_acc`) stays F32. CPU has
   `attention_forward_bf16` / `attention_backward_bf16` mirroring the F32
   path with f32 inner compute and bf16 storage. GPU has four new shaders
   (`attention_bf16`, `attention_back_d_bf16`, `attention_back_kv_bf16`,
   `attention_back_q_bf16`) that load bf16 via packed-uint shift-extract,
   run the same online-softmax FA2 algorithm in f32, and pair-pack writes
   back to bf16. Backward kernels stage per-thread dQ/dK/dV in shared
   memory and have the first D/2 threads do the bf16 RMW so adjacent d's
   share a packed uint without races. Even `head_size` is required and
   asserted at dispatch. Tests in `tests/dtype_roundtrip` cover bf16
   attention forward + backward against an F32 reference on both backends
   (5e-2 tolerance, since softmax/exp prevent bit-exact compare).

2. **Bf16 element-wise/normalization ops.** Mechanical rollout following
   the `add` pattern (thread-per-output-pair, f32 inner compute, packed
   bf16 storage):
   - `mul`, `sub`, `div` — DONE (CPU + GPU, parity tests).
   - `gelu`, `silu`, `relu`, `sigmoid`, `tanh`, `exp` — DONE (CPU + GPU
     via `_unary_forward_gpu` / `_unary_backward_gpu` helper, parity tests).
   - `layernorm` — DONE. Stats and forward shaders use packed-bf16 I/O
     with f32 mean/rstd scratch; backward input/weight use the same
     pair-aligned RMW pattern. Even `size` required.
   - `softmax`, `log_softmax`, `entropy` — DONE. F32 reductions
     internally; even `size` required for pair-aligned writes. `entropy`
     uses one workgroup per output pair to handle the `output[count]`
     packed write without races.
   - `rope`, `causal_mask`, `permute`, `concat`, `slice`, `mean` —
     DONE (CPU + GPU). Pure-copy ops (`slice`, `concat3`, `permute`,
     `causal_mask`) use one-thread-per-pair packed-uint reads/writes
     so the two halves of one packed bf16 slot are owned by exactly
     one thread (no race on RMW). `rope` rotates whole bf16 pairs:
     `head_size` is even so `(i_lo, i_hi)` always lands on an aligned
     packed slot, one thread does both rotations and a single packed
     write. `mean` forward uses one workgroup per row plus an atomic
     CompSwap to RMW the half of the output pair belonging to that
     row; backward is one thread per input pair.
     `ml.slice` and `ml.permute` now allocate output as `input.type`
     instead of hardcoded F32.

3. **First bf16 transformer parity test — DONE.** New
   `transformer_train_bf16` test in `tests/pytorch_parity` mirrors the
   library's transformer block in PyTorch with bf16 weights/activations
   (FP32 master → bf16 cast inside `forward`, FP32 logits at the end
   so `backward()` gets an F32 loss). Loss curves match within ~3e-4
   absolute over 10 Adam steps on both CPU and GPU at the
   `mlp_train`-style tolerance.

   Gap surfaced and fixed: `slice_trailing` had no bf16 path on either
   backend — CPU `cpu.odin` used the F32-only `data()` / `gradient()`
   helpers, and the GPU shaders typed the buffer as `float[]`. Both
   read and wrote 4 bytes per "bf16 element," producing silent memory
   corruption that the dtype_roundtrip suite missed (it covers `slice`
   but not `slice_trailing`). Added bf16 paths in `cpu.odin`,
   `slice_trailing_bf16.comp`, and `slice_trailing_back_bf16.comp`
   (one-thread-per-pair packed-uint reads with RMW on the backward
   scatter, same pattern as the existing bf16 pure-copy ops).

4. **Coopmat perf pass.** Shared layout across every shader: BM=64,
   BN=64, BK=16, 4 subgroups arranged 2×2 per WG, each subgroup owning
   a 2×2 grid of 16×16 output coopmat tiles, inputs staged through
   shared memory. Subgroup size = 32 hardcoded. Build script now
   passes `--target-env=vulkan1.3` (needed for `GroupNonUniform`).
   - **Linear forward, linear backward (input + weight),
     batched_matmul forward + backward — DONE.** Eligibility for each
     shader requires the relevant output dims to be multiples of 64
     and the contraction dim a multiple of 16; smaller/odd shapes
     fall back to the existing naive packed-uint shaders. Coopmat
     backward shaders load the existing output tile as a bf16 acc
     coopmat, convert to fp32, accumulate via `coopMatMulAdd`, and
     convert back to bf16 on store, so they preserve the `+=`
     semantics that the naive backward used.

     RTX 3090 Ti microbench (BF16 peak ~150 TFLOPS):
     - linear forward:
       - 512×768×768:    2.34 → 3.58 TFLOPS  (1.5x)
       - 512×2048×2048:  5.88 → 15.62 TFLOPS (2.7x)
       - 2048×2048×2048: 7.99 → 27.89 TFLOPS (3.5x)
     - linear backward (sum of dx + dw GEMMs):
       - 512×768×768:    1.27 → 4.91 TFLOPS  (3.9x)
       - 512×2048×2048:  1.53 → 14.67 TFLOPS (9.6x)
       - 2048×2048×2048: 1.81 → 17.33 TFLOPS (9.6x)
     The backward win is largest because the old naive bf16 backward
     was bottlenecked on packed-uint RMW serialization, not compute.

     End-to-end transformer bench (`tests/gpu_transformer_bench`,
     L=12 H=8 E=512 T=256): F32 full step 76.1 ms → Bf16 full step
     30.9 ms (**2.46x**). Forward alone: 1.49x. The full-step
     speedup is dominated by backward.

     Tried-and-rejected variants for linear forward: BM=BN=128
     (faster on huge but slower on medium/big due to
     under-occupancy); BK=32 (no measurable win, slightly worse on
     big shapes).

     Coverage: `dtype_roundtrip` multi-tile linear test (M=64, N=128,
     K=64) exercises the linear coopmat path; new
     `batched_matmul (coopmat-tile)` test (BATCH=2 M=K=N=64)
     exercises both the bmm forward and backward coopmat paths.

   - **Attention forward — implemented but disabled.**
     `attention_bf16_coopmat.comp` is a full FA2-with-Q-tiling
     rewrite: BR=16 queries per workgroup, BC=64 keys per K-tile, 1
     subgroup of 32 threads, MAX_D=64 capping shared-memory at
     ~32 KB. Q@K^T and P@V both run on tensor cores via
     `coopMatMulAdd`; the online softmax stages scores through fp32
     shared memory, runs SIMT per-row max/sum via `subgroupMax`/
     `subgroupAdd` (32 threads × 2 cols each = BC=64), and writes
     bf16 P back to shared for the P@V matmul. Output is folded
     into a fp32 running accumulator with per-row alpha rescaling,
     then normalized and pair-packed to bf16 at the end. Correctness
     verified by a new `dtype_roundtrip` case at T=32, H=2, D=16
     (the smallest shape that hits the coopmat path).

     **Disabled in dispatch.** On the bench shapes (D=64, T=64–256)
     this runs ~18% *slower* than the existing SIMT bf16 shader.
     Root causes: the per-row softmax loop is sequential across
     16 rows (one `subgroupMax` + `subgroupAdd` per row), each K-
     tile costs 4 barriers (Q/K/V stage + score store + softmax +
     output fold), and shared-memory budget (~32 KB at MAX_D=64,
     ~58 KB at MAX_D=128) limits occupancy. The SIMT shader hides
     the same compute under simpler control flow with much less
     shared-memory pressure, so the tensor-core throughput
     advantage doesn't materialize at these sizes. Left as a
     foundation for the backward pass and for future tuning
     (multi-subgroup BR=32, fewer barriers via more-aggressive
     overlap, specialization-constant D, larger seq-len bench
     shapes where the coopmat advantage will dominate).

   - **Attention backward — not started.** The three backward
     shaders (`attention_back_d`, `attention_back_kv`,
     `attention_back_q`) are mostly per-token reductions plus
     `softmax_outputs[H, T, T]` recompute, not pure GEMMs, so a
     coopmat port is a bigger restructure than the linear/bmm
     ports. Best done together with the forward redesign once the
     forward profile is unblocked.

5. **Mixed-precision recipe in `examples/`.** End-to-end demo of the
   standard "FP32 master weights, BF16 forward/backward, FP32 optimizer
   state" pattern on a small transformer. Deliverable for "this library
   is a real bf16 stack."

### Open / next phases

- **Make attention coopmat profitable.** Forward shader exists
  (`attention_bf16_coopmat.comp`) but currently slower than SIMT.
  Most promising next moves: parallelize the per-row softmax across
  threads (use a (row, col) thread layout instead of sequential
  rows), reduce barrier count by fusing Q-load with the first K
  iteration, multi-subgroup BR=32 to amortize K/V staging across
  more queries, specialization constants for D so shared-memory
  isn't sized for the worst case. Once the forward wins, port the
  same Q-tile layout to backward.

- **Mixed-precision recipe in `examples/`.** Show the standard "FP32
  master weights, BF16 forward/backward, FP32 optimizer state"
  pattern end-to-end on a small transformer. The `transformer_train_bf16`
  parity test and the bf16 path in `tests/gpu_transformer_bench`
  already demonstrate the building blocks; promote to a runnable
  example once attention coopmat lands.

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
