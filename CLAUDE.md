## Intro

This is a machine learning library written completely from scratch. It has
a CPU backend (in `ml.odin`) and a Vulkan compute backend (in `gpu/` +
`gpu_transformer/`) that supports full transformer training.

## Current Goal

The GPU backend supports end-to-end training (forward + backward + Adam).
Per-kernel correctness is verified against CPU; per-step gradient
agreement is at the fp32 floor; the model trains. The GPU full training
step is ~8 ms vs CPU multi-thread ~6.7 ms at L=4 H=4 E=128 T=64 — the
GPU is competitive but doesn't dominate at this small scale.

Remaining work is in **GPU backend perf optimization** (see "GPU
backend — remaining work" below) and **scaling tests at production
shapes** where the GPU is expected to pull ahead substantially.

## Testing

You MUST confirm via testing that code changes actually preserve correctness
and (for perf changes) actually make the code faster.

**CPU changes:** `benchmarks/benchmark/benchmark.exe` is the canonical perf bench. Use
the single-threaded checksum column to verify correctness across versions
(within ~1 ULP — SIMD reduction order can shift the last bit).

**GPU changes:**
- `benchmarks/gpu_back_check`: per-kernel correctness against
  CPU/hand-computed reference. Run after editing any backward kernel.
- `benchmarks/gpu_forward_check` + `gpu_grad_check`: end-to-end CPU vs GPU
  agreement at fp32 floor (one forward / one full step). Run after
  editing any forward or backward op.
- `benchmarks/gpu_train_check`: 100-step training comparison. Run after
  editing the optimizer or `gpu_transformer.backward`.
- `benchmarks/gpu_forward_bench` and `gpu_train_bench`: perf benches. Run
  before and after any GPU perf change.

All GPU benchmarks build with the same flags as CPU plus the shader
recompile if any `.comp` files changed (see "Shader build" below).

## Build

Builds must pass `-microarch:native` (or another flag enabling AVX2 + FMA) to
get the FMA-vectorized inner loops in `_simd_dot_f32` / `_simd_axpy_f32`.
Without it the SIMD lanes still vectorize but FMA falls back to mul+add and
performance drops ~30%. Example:

    odin build benchmark -o:speed -no-bounds-check -microarch:native -out:benchmark/benchmark.exe

## CPU architecture notes

### SIMD primitives (in `ml.odin`)

`_simd_dot_f32(a, b, n) -> f32` and `_simd_axpy_f32(y, x, a, n)` are the
8-lane f32 building blocks for `linear` and `attention`. Both:
- Take `[^]f32` (multi-pointer) so callers can pass slice offsets cheaply.
- Use `intrinsics.unaligned_load` / `unaligned_store` — gradients/data slices
  aren't 32-byte aligned, and unaligned moves are full-speed on every CPU
  newer than ~Sandy Bridge.
- Handle a scalar tail when `n` isn't a multiple of `_SIMD_LANES`.

### Custom worker pool (in `ml.odin`)

`parallelize` is implemented on a hand-rolled persistent worker pool, not
`core:thread.Pool`. Each worker parks on a per-worker `sync.Sema`; dispatch
is `n-1` sema posts + main thread runs slice 0 + `wait_group_wait`. **Zero
allocations per `parallelize` call**, which matters because the transformer
training loop makes ~16 calls per step. The earlier `core:thread.Pool`
implementation allocated a Task struct per `pool_add_task` and was ~2× slower
end-to-end. Main thread participates as worker 0, so `set_thread_count(N)`
spawns `N-1` background workers.

### What didn't help (so don't redo it)

- **Hand-SIMD on the Adam `update` loop.** LLVM with `-microarch:native`
  already auto-vectorizes the scalar version, and the operation is close to
  memory-bandwidth bound at typical parameter sizes. The 4-line scalar loop
  is as fast as 30 lines of explicit SIMD.
- **Splitting `linear_backward` into two passes (weight grad, then input
  grad) for parallelization over output rows.** Reads weight twice instead
  of once → doubled L2/L3 traffic → ST regression. The fused per-sample
  pass that the current `linear_backward` uses is correct.
- **Parallelizing `linear` over output rows when `count == 1`.** The win is
  real for inference (count=1) but the user's hot path is transformer
  training where count = token_count is already > thread count, so the
  existing parallel-over-count strategy already keeps all threads busy.

### Verification protocol for perf changes

Run `./benchmark/benchmark.exe` before and after. The single-threaded
checksum column should match within ~1 ULP (SIMD reduction order can shift
the last bit). Multi-threaded checksums drift in low bits because thread
reduction order is non-deterministic — that's expected; use the ST checksum
for correctness verification.

## Project layout

- `ml.odin`: CPU primitives — tensors, autograd tape, ops, Adam optimizer,
  worker pool, SIMD kernels.
- `mlp/`, `gru/`, `transformer/`: CPU model implementations.
- `benchmark/`: CPU microbenchmarks + end-to-end transformer step timing.
- `gpu/`: Vulkan compute backend.
  - `gpu.odin`: instance / device / command pool / descriptor pool init.
  - `buffer.odin`: GpuTensor allocation, host↔device upload/download.
  - `pipeline.odin`: pipeline construction, `_dispatch`, command-buffer
    batching (`begin_batch` / `end_batch`).
  - `ops.odin`: every public op (forward + backward + optimizer).
  - `shaders/`: GLSL compute shaders + compiled `.spv` (checked in).
- `gpu_transformer/`: GPU mirror of the CPU transformer — Layer /
  Transformer with parallel `*_grad` / `*_m` / `*_v` fields per weight,
  Activations pool with paired data + grad arrays, `forward` / `backward`
  / `update` / `zero_grad`.
- `benchmarks/`:
  - `gpu_back_check`: per-kernel backward correctness against CPU /
    hand-computed reference. **Run this whenever a backward kernel changes.**
  - `gpu_grad_check`: end-to-end one-step gradient match (CPU vs GPU)
    after 3 steps of CPU pretraining. fp32 floor (~1e-8 max abs).
  - `gpu_train_check`: 100-step parallel training comparison. Verifies
    both backends drive loss to overfit.
  - `gpu_forward_check`, `gpu_forward_bench`, `gpu_train_bench`:
    correctness + perf at the canonical L=4 H=4 E=128 V=256 T=64 config.

## GPU backend — architecture notes

### Command-buffer batching (in `gpu/pipeline.odin`)

`begin_batch()` / `end_batch()` open a recording context. While active,
every `_dispatch` call records into a single command buffer with a
`SHADER_WRITE → SHADER_READ|SHADER_WRITE` global memory barrier between
dispatches; `end_batch` does one submit + one `vkQueueWaitIdle`.
Eliminates the per-dispatch drain that dominated forward-pass time. With
batching, one transformer forward is ~50× cheaper in submit overhead than
without. Outside a batch, `_dispatch` falls back to one-shot submit per
call so the upload / download paths and per-kernel correctness checks
still work without ceremony.

Transient resources (descriptor sets, the `select` indices buffer, the
attention/layernorm scratch buffers) are queued via `_queue_destroy_buffer`
and reclaimed in `end_batch`. Outside a batch, they're destroyed
immediately (the per-dispatch wait_idle has already finished by then).

### Activation pool with grads (in `gpu_transformer/`)

`Activations` is an index-based pool. `forward` resets `next = 0` and
calls `_act` to acquire data tensors; the pool grows on the first pass
and reuses thereafter. Each entry has a paired gradient tensor allocated
in lockstep, so backward can call `_act_grad(t)` to look up the matching
grad. Steady-state forward / backward calls do zero Vulkan alloc traffic.

VRAM cost: 2× activation memory during training. Acceptable at our
shapes (a few MB total).

### `gpu_transformer.backward` shape

Re-walks the forward `_act` sequence to bind every activation to a local
variable (no GPU work — just pool lookups), captures per-layer
activations in `_LayerActs`, then issues backward kernels in reverse
forward order. Activation grads are zeroed at the start of each backward
sweep; parameter grads are zeroed by Adam's update kernel as a side
effect (so `zero_grad` is only needed for gradient accumulation across
micro-batches).

### Atomic-free backward kernels

Every backward kernel structures its threads per-output-element (rather
than scattering from per-input). Reduction axes are walked in the inner
loop. This avoids needing `VK_EXT_shader_atomic_float` and matches
ggml-vulkan's pattern. Specifics:

- `linear_back_input`: thread per (sample, input_dim), reduces over output_dim.
- `linear_back_weight`: thread per (output_dim, input_dim), reduces over samples.
- `select_back`: thread per (vocab_id, embed_dim), inner loop scans the
  indices array (vocab × n × embed is cheap at training sizes).
- `attention_back_*`: 6 dispatches, all atomic-free. `attention_back_post`
  recomputes the forward softmax into a transient `[T, H, T]` buffer so
  dV / dQ / dK can read it directly instead of recomputing per-thread.

### CPU↔GPU fp32 reduction-order drift (don't redo investigation)

Per-step gradients agree at the fp32 floor (~1e-8 max abs). After Adam
updates, parameters drift because `1/(sqrt(v_hat) + eps)` amplifies tiny
grad differences when `v_hat` is small. Drift compounds across steps. CPU
and GPU loss trajectories are **not byte-equivalent** and never will be
without higher precision or matching reduction orders. Sources of
divergence:
- Parallel tree-reductions in `layernorm_stats`, `layernorm_back_input`,
  `attention_back_post`, `attention_back_pre_grad`, `cross_entropy_grad`.
- GLSL implementations may or may not use FMA depending on the compiler;
  CPU `_simd_axpy_f32` always uses FMA.

Verification bar:
- `gpu_back_check` proves every kernel matches CPU semantics on a single
  call.
- `gpu_grad_check` proves end-to-end gradients match at the fp32 floor.
- `gpu_train_check` proves both backends actually train (both reach low
  overfit loss, not the same trajectory).

If you suspect a real bug, compare against `gpu_grad_check`'s 1e-8
max-abs bar — that's the canonical "math is correct" signal.

## GPU backend — remaining work

Ordered roughly by expected impact on GPU training-step time at our
canonical shape:

1. **Tiled GEMM for `linear_back_input` / `linear_back_weight`.** The
   forward `linear` shader is a serious tiled GEMM (TILE_M=64, TILE_N=64,
   TILE_K=16, shared-mem K-staging, register-tile accumulation). The
   backward versions are still naive thread-per-output-element. Backward
   linear ops account for a large slice of the 6.3 ms in
   backward+update; bringing them to GEMM-class throughput should cut
   the full step substantially.
2. **Save mean/rstd from forward layernorm** instead of recomputing in
   `layernorm_back`. Eliminates one dispatch per layernorm × 8
   layernorms per step. Requires extending the activation pool to allow
   "sidecar" tensors per pool entry, or just allocating mean/rstd
   slots alongside each layernorm activation.
3. **Save attention `post` from forward** (the softmax output). Currently
   `attention_back_post` recomputes it, which costs ~1 forward attention
   worth of compute per layer. Same trade-off as #2 — pool extension or
   parallel sidecar.
4. **Fused activation-grad zero.** `backward` calls `gpu.zero` once per
   pool entry (~50 dispatches at our shape). Replace with one
   `vkCmdFillBuffer` per buffer, or with one fused kernel that zeroes
   all activation grads in a single dispatch given a list of
   (offset, count) pairs.
5. **Bigger shapes / scaling test.** At L=4 E=128 T=64 the 3090 Ti is
   barely flexing. A scaling test at e.g. L=12 E=512 T=256 would show
   how the GPU's lead grows, and is the most informative perf
   benchmark for the next round of work.
6. **Activation pool: separate inference vs training mode.** Currently
   the pool always allocates a paired grad buffer (2× VRAM). For
   inference-only paths (sampling from a trained model), this is
   wasted memory.
7. **Persistent staging buffers for upload/download.** Currently each
   `gpu.upload` / `gpu.download` creates and destroys a host-visible
   buffer. Setup paths only — not on the training hot path — but easy
   win and removes a class of per-call alloc.

## Code style

### Naming

- **Types**: `Ada_Case` (`Tensor`, `Layer_Acts`, `Attention_Back_Post_Params`).
  Multi-word types use underscores between words. **No leading underscore** —
  even file-local types stay unprefixed.
- **Constants** (declared with `::`): `SCREAMING_SNAKE_CASE`
  (`MAX_TENSOR_RANK`, `SIMD_LANES`, `DESCRIPTOR_POOL_MAX_SETS`). **No leading
  underscore** — even file-local constants stay unprefixed.
- **Functions and variables**: `snake_case`. File-local ones get a leading
  underscore (`_worker_proc`, `_dispatch`, `_simd_dot_f32`, `_gpu`, `_batch`).
  Public ones don't (`linear`, `update`, `parallelize`).

**Don't use `@(private)` or `@(private="file")`** — leading-underscore prefix
on functions/variables is the convention here. Types and constants are not
visibility-marked at all; if a type is conceptually internal, just don't
mention it in API docs and trust the prefix-free naming.

### API stability

- Public procs that must stay ergonomic for callers (e.g. `linear`, `update`,
  `parallelize`) keep their existing signatures. Optimizations live behind
  these signatures, not in API changes.
- GPU op signatures take `GpuTensor` for data buffers and pass shape as
  explicit `int` parameters. Don't read shape off `GpuTensor.shape` inside
  ops — that field is informational, not authoritative for kernel work.

## Shader build

After editing a `.comp` file, recompile to `.spv`:

    cd gpu/shaders && for f in *.comp; do glslc -O "$f" -o "${f%.comp}.spv" || break; done

(or just the file you changed). The `.spv` files are checked in so the
project builds without `glslc` on other machines.