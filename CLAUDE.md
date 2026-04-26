## Intro

This is a machine learning library written completely from scratch. It has
two compute backends — a CPU backend (in `ml.odin`) and a Vulkan compute
backend (in `gpu/`) — that share a single `Backend`-dispatched op system.

Models written in the library use the same `ml.*` calls regardless of
backend. The active backend is held on the active `Context` and
determines whether `ml.zeros` allocates host memory or a Vulkan buffer,
and whether `ml.add` dispatches a SIMD CPU loop or a compute shader.

There is also a parallel **legacy** GPU path in `gpu_transformer/` +
`gpu/ops.odin`'s `GpuTensor`-based procs that pre-dates the unified API.
It still trains a transformer end-to-end and is used by `gpu_grad_check`
and friends. It is being retired as ops migrate to the unified path.

## Current State

**Architecture:** Per-context state with a thread-local linked-list stack.
Backend abstraction via tagged dispatch (one `forward(op)` and one
`backward(op)` proc per backend, switching on `op.variant`). `ml.Tensor`
carries a `backend: ^Backend` and an opaque `storage: rawptr`; CPU stores
data in host slices, GPU stores it via a `gpu.Gpu_Storage` containing
`vk.Buffer` handles. Thread-safe: each host thread owns its own Context;
the shared CPU worker pool serializes `parallelize` fan-outs with a
mutex.

**Op port status (unified API):**

CPU backend — all 30 ops dispatch through `cpu_forward` / `cpu_backward`.

GPU backend — 10 ops ported and tested via `gpu_unified_check`:
- `add`, `linear`, `gelu`
- `select`, `rope`, `slice_trailing`, `concat` (3-input only)
- `softmax`, `permute`, `causal_mask`

GPU backend — not yet ported:
- `mul` (with broadcast), `batched_matmul`, `layernorm`, `cross_entropy`,
  most elementwise / reductions / activations.

**Attention is decomposed.** There is no `Attention` op variant. Both
backends compute attention as a composition of `slice_trailing` →
`reshape` → `permute` → `batched_matmul` → `mul` → `causal_mask` →
`softmax` → `batched_matmul` → `permute` → `reshape`. CPU-side this is
the only path. GPU-side `ml.attention` will work end-to-end once `mul`
(broadcast) and `batched_matmul` are ported. The legacy `gpu.attention`
kernel + 7 `attention_*.spv` shaders still exist for the
`gpu_transformer/` path and will be deleted when that path is retired.

**Decomposition perf cost.** CPU attention sub-bench is ~3-4× slower
than the old fused kernel; full transformer step ~10-22% slower across
thread counts. Acceptable cost for the simpler, composable code.

## Next steps (where to pick up)

1. **`mul` with broadcast on GPU.** New shader (forward + backward) that
   handles `len(a) % len(b) == 0` broadcast. Needed for
   `mul(scaled_scores, scalar(1/sqrt(D)))` in attention.
2. **`batched_matmul` on GPU.** New shaders: forward + backward (likely
   split into back-input and back-weight kernels like `linear`). Most
   complex remaining op; biggest single port.
3. **`set_data` backend hook.** `ml.scalar(value)` and `ml.tensor(data)`
   currently write to `t.data` directly, which is empty for GPU tensors.
   Add a `Backend.set_data(t, src)` hook: CPU does a slice copy, GPU does
   `upload_tensor`. Needed before #1/#2 are useful in `attention()`.
4. **End-to-end attention on GPU through unified API.** Add a phase to
   `gpu_unified_check` that runs `ml.attention` on GPU and compares to
   CPU. With #1/#2/#3 done, this should pass at fp32 floor.
5. **`layernorm` on GPU via unified API.** Forward already has shaders
   (`layernorm.spv` + `layernorm_stats.spv`); backward has
   `layernorm_back_input.spv` + `layernorm_back_weight.spv`. Just need
   the dispatch glue.
6. **`cross_entropy` on GPU.** Existing `cross_entropy_grad.spv` covers
   the backward; forward computes the loss CPU-side currently. Plus the
   internal softmax + NLL — figure out the right decomposition.
7. **Delete legacy GPU attention path.** Once `ml.attention` runs
   end-to-end on GPU, delete `attention*.comp/.spv`,
   `gpu.attention`/`gpu.attention_back`, and (eventually)
   `gpu_transformer/` once the unified path covers all transformer ops.

## Testing

You MUST confirm via testing that code changes preserve correctness and
(for perf changes) actually make the code faster.

**CPU changes:** `benchmarks/benchmark/benchmark.exe` is the canonical
perf bench. Use the single-threaded checksum column to verify
correctness across versions (within ~1 ULP — SIMD reduction order can
shift the last bit).

**Thread-safety changes:** `benchmarks/thread_safety_check` spawns 4 host
threads, each with its own `ml.Context`, all hammering the shared worker
pool. Forward must be bit-exact across threads; backward gradients
within ~1% (existing benign race in `linear_backward` accumulating into
shared weight rows).

**GPU changes — unified API:**
- `benchmarks/gpu_unified_check`: per-op CPU vs GPU correctness for
  every op ported through the unified `ml.Backend` dispatch. **Run this
  whenever a new op is ported or a primitive's CPU/GPU implementation
  changes.** Each op has its own phase; tolerances are op-specific (most
  are bit-exact, ops with reductions are at the fp32 floor).

**GPU changes — legacy path** (still relevant while `gpu_transformer/`
exists):
- `gpu_back_check`: per-kernel backward correctness against CPU /
  hand-computed reference.
- `gpu_grad_check`: end-to-end one-step gradient match (CPU vs GPU)
  after 3 steps of CPU pretraining. fp32 floor (~1e-8 max abs).
- `gpu_train_check`: 100-step parallel training comparison.
- `gpu_forward_check`, `gpu_forward_bench`, `gpu_train_bench`:
  correctness + perf at L=4 H=4 E=128 V=256 T=64.

## Build

Builds must pass `-microarch:native` (or another flag enabling AVX2 +
FMA) to get the FMA-vectorized inner loops in `_simd_dot_f32` /
`_simd_axpy_f32`. Without it the SIMD lanes still vectorize but FMA
falls back to mul+add and performance drops ~30%. Example:

    odin build benchmark -o:speed -no-bounds-check -microarch:native -out:benchmark/benchmark.exe

## Architecture

### `Context` and the TLS stack (in `ml.odin`)

`Context` is the per-thread state holder. It owns:
- An arena (for activation tensors).
- The autograd tape (a `[MAX_OPERATIONS]Operation` ring).
- A `backend: ^Backend` pointer.

Contexts are heap-allocated via `context_create(size, backend?)`. They
live on a thread-local stack threaded through a `previous_ctx` field;
`context_scope(ctx)` pushes via `@(deferred_none=context_end)`.

```odin
ctx := ml.context_create(N)
defer ml.context_destroy(ctx)
ml.context_scope(ctx)
```

Multiple Contexts can stack on a thread (e.g., briefly switching
backends mid-call). One Context is owned by one thread at a time.

### `Backend` and tagged dispatch (in `ml.odin`)

```odin
Backend :: struct {
    name:                    string,
    data:                    rawptr,
    alloc:                   proc(t: ^Tensor, n: int),
    clear_storage:           proc(),
    fill_gradient_with_ones: proc(t: ^Tensor),
    forward:                 proc(op: Operation),
    backward:                proc(op: Operation),
}
```

Each backend has ONE `forward` and ONE `backward` proc that internally
`switch _ in op.variant` and call into op-specific kernels. Adding an
op means adding one variant + one case in each backend's switch — no
new fields on `Backend`. CPU backend's switch is exhaustive; GPU
backend's is `#partial`-style (unported variants `panic`).

The decision to use tagged dispatch instead of a vtable was deliberate:
the switch already existed in `backward()`, dispatch overhead is dwarfed
by op work, and the variant union gives compile-time exhaustiveness on
the CPU side without the 60-field Backend struct.

### `Tensor` with backend storage (in `ml.odin`)

```odin
Tensor :: struct {
    backend:  ^Backend,
    storage:  rawptr,         // backend-specific (CPU: nil; GPU: ^gpu.Gpu_Storage)
    data:     []f32,          // CPU: real slice. GPU: empty.
    gradient: []f32,          // CPU: real slice. GPU: empty.
    shape:    [MAX_TENSOR_RANK]int,
    rank:     int,
    count:    int,            // total element count, set at allocation
}
```

CPU code reads `t.data[i]` directly. GPU code reads `t.storage` and
casts it to `^gpu.Gpu_Storage` (which holds `vk.Buffer` + `vk.DeviceMemory`
for both data and gradient). `len(t)` returns `t.count` — works for both
backends.

Allocation goes through `_current_ctx.backend.alloc(&t, n)`. CPU
allocates from the context arena. GPU creates two DEVICE_LOCAL buffers
and tracks them on the active `Gpu_Context` for bulk release on
`ml.clear()`.

### SIMD primitives (in `ml.odin`)

`_simd_dot_f32(a, b, n) -> f32` and `_simd_axpy_f32(y, x, a, n)` are the
8-lane f32 building blocks for `linear`, `batched_matmul`, and the
attention-related kernels. Both:
- Take `[^]f32` (multi-pointer) so callers can pass slice offsets cheaply.
- Use `intrinsics.unaligned_load` / `unaligned_store` — gradients/data
  slices aren't 32-byte aligned, and unaligned moves are full-speed on
  every CPU newer than ~Sandy Bridge.
- Handle a scalar tail when `n` isn't a multiple of `SIMD_LANES`.

### Custom worker pool (in `ml.odin`)

`parallelize` is implemented on a hand-rolled persistent worker pool,
not `core:thread.Pool`. Each worker parks on a per-worker `sync.Sema`;
dispatch is `n-1` sema posts + main thread runs slice 0 +
`wait_group_wait`. **Zero allocations per `parallelize` call**, which
matters because the transformer training loop makes ~16 calls per step.
Main thread participates as worker 0, so `set_thread_count(N)` spawns
`N-1` background workers.

**Thread safety:** `parallelize` takes a mutex around the fan-out. Two
host threads each calling `parallelize` serialize at the mutex —
acceptable because each fan-out saturates the cores anyway, and we'd
rather queue than oversubscribe.

### What didn't help (so don't redo it)

- **Hand-SIMD on the Adam `update` loop.** LLVM with `-microarch:native`
  already auto-vectorizes the scalar version, and the operation is close
  to memory-bandwidth bound at typical parameter sizes.
- **Splitting `linear_backward` into two passes (weight grad, then input
  grad) for parallelization over output rows.** Reads weight twice
  instead of once → doubled L2/L3 traffic → ST regression.
- **Parallelizing `linear` over output rows when `count == 1`.** The win
  is real for inference but the transformer training hot path has count
  = token_count > thread count, so parallel-over-count already keeps
  workers busy.

## GPU backend — architecture notes

### Two-tier state: `Gpu_Device` (singleton) + `Gpu_Context` (per-thread)

`Gpu_Device` (file-scope `_gpu`): instance, physical_device, device,
queue, queue_family_index, memory_properties, pipelines list, loader,
device_name. Created by `gpu.init()`, freed by `gpu.destroy()`. Shared
across all contexts on the process.

`Gpu_Context` (thread-local stack): command_pool, descriptor_pool,
batch state, allocations list. One per ml.Context that uses the GPU.
Vulkan command pools and descriptor pools are NOT thread-safe, so each
host thread that issues GPU work owns its own `Gpu_Context`.

```odin
gpu.init()
defer gpu.destroy()
gctx := gpu.context_create()
defer gpu.context_destroy(gctx)
gpu.context_scope(gctx)

ctx := ml.context_create(N, gpu.backend())
defer ml.context_destroy(ctx)
ml.context_scope(ctx)
```

### `Gpu_Storage` and the `Backend.alloc` hook

`gpu.backend()` returns a `^ml.Backend` whose `alloc` hook creates a
pair of DEVICE_LOCAL buffers (data + grad), stashes a `^Gpu_Storage` in
`Tensor.storage`, and registers it on the active `Gpu_Context`'s
allocations list. Both buffers are zeroed via a one-shot
`vkCmdFillBuffer` on alloc to match CPU's `make([]f32, n)` semantics —
critical for backward (which `+=` into gradient buffers).

`clear_storage` releases every tracked allocation in bulk; mirrors CPU's
arena reset. Per-step alloc/free overhead is real (one `_create_buffer`
call + one `_one_shot_copy` per tensor); a pool will replace this when
the perf matters.

### Command-buffer batching (in `gpu/pipeline.odin`)

`begin_batch()` / `end_batch()` open a recording context. While active,
every `_dispatch` call records into a single command buffer with a
`SHADER_WRITE → SHADER_READ|SHADER_WRITE` global memory barrier between
dispatches; `end_batch` does one submit + one `vkQueueWaitIdle`.
Outside a batch, `_dispatch` falls back to one-shot submit per call.

Transient resources (descriptor sets, the `select` indices buffer) are
queued via `_queue_destroy_buffer` and reclaimed in `end_batch`.
Outside a batch, they're destroyed immediately (the per-dispatch
wait_idle has already finished by then).

### Atomic-free backward kernels

Every backward kernel structures its threads per-output-element (rather
than scattering from per-input). Reduction axes are walked in the inner
loop. This avoids needing `VK_EXT_shader_atomic_float`. Specifics:

- `linear_back_input`: thread per (sample, input_dim), reduces over output_dim.
- `linear_back_weight`: thread per (output_dim, input_dim), reduces over samples.
- `select_back`: thread per (vocab_id, embed_dim), inner loop scans the
  indices array.
- `permute_back`: one thread per output-grad element; permute is a
  bijection so no race.
- `softmax_back`: workgroup per row, threads collaborate on the
  `<y, dy>` reduction.

### CPU↔GPU fp32 reduction-order drift

Per-step gradients agree at the fp32 floor (~1e-8 max abs for ported
ops). After Adam updates, parameters drift because
`1/(sqrt(v_hat) + eps)` amplifies tiny grad differences. Drift
compounds across steps. CPU and GPU loss trajectories are **not
byte-equivalent** and never will be without higher precision or
matching reduction orders. Sources of divergence:
- Parallel tree-reductions (softmax row reduction, layernorm stats,
  bmm reductions, cross_entropy_grad).
- GLSL implementations may or may not use FMA depending on the compiler;
  CPU `_simd_axpy_f32` always uses FMA.

Verification bar:
- `gpu_unified_check` proves every ported op matches CPU at fp32 floor
  (op-specific tolerances, mostly 1e-7 to 1e-8).
- Legacy `gpu_back_check` / `gpu_grad_check` / `gpu_train_check` cover
  the `gpu_transformer/` path until it's retired.

## Project layout

- `ml.odin`: CPU primitives, autograd tape, `Context` / `Backend` / op
  procs, Adam optimizer, worker pool, SIMD kernels.
- `mlp/`, `gru/`, `transformer/`: model implementations (backend-agnostic
  via the unified API).
- `gpu/`: Vulkan compute backend.
  - `gpu.odin`: instance / device init; `Gpu_Context` lifecycle.
  - `buffer.odin`: legacy `GpuTensor` allocation, host↔device upload/download.
  - `pipeline.odin`: pipeline construction, `_dispatch`, command-buffer
    batching.
  - `ops.odin`: legacy `GpuTensor`-based op procs + SPIRV constants and
    Param structs (shared with the new dispatch).
  - `backend.odin`: unified `ml.Backend` integration — `Gpu_Storage`,
    `_gpu_backend` instance, `gpu_*_forward` / `gpu_*_backward` dispatch
    procs, `upload_tensor` / `download_tensor` helpers.
  - `shaders/`: GLSL compute shaders + compiled `.spv` (checked in).
- `gpu_transformer/`: legacy GPU transformer using `GpuTensor` directly.
  To be retired once unified-API ops cover the full transformer.
- `benchmarks/`:
  - `benchmark`: canonical CPU perf + ST checksums.
  - `thread_safety_check`: 4-host-thread stress test of the worker pool.
  - `gpu_unified_check`: per-op CPU vs GPU correctness on the unified API.
  - `gpu_back_check`, `gpu_grad_check`, `gpu_train_check`,
    `gpu_forward_check`, `gpu_forward_bench`, `gpu_train_bench`,
    `gpu_bench`, `gpu_hello`: legacy-path tests + perf benches.
- `examples/`: `mnist`, `imitation_learning`, `ppo`, `text_generation_gru`,
  `text_generation_transformer`. All on CPU through the unified API.

## Code style

### Naming

- **Types**: `Ada_Case` (`Tensor`, `Layer_Acts`, `Batched_Matmul`,
  `Gpu_Storage`). Multi-word types use underscores between words. **No
  leading underscore** — even file-local types stay unprefixed.
- **Constants** (declared with `::`): `SCREAMING_SNAKE_CASE`
  (`MAX_TENSOR_RANK`, `SIMD_LANES`, `DESCRIPTOR_POOL_MAX_SETS`).
  **No leading underscore** — even file-local constants stay unprefixed.
- **Functions and variables**: `snake_case`. File-local ones get a
  leading underscore (`_worker_proc`, `_dispatch`, `_simd_dot_f32`,
  `_gpu`). Public ones don't (`linear`, `update`, `parallelize`).

**Don't use `@(private)` or `@(private="file")`** — the
leading-underscore prefix on functions/variables is the convention here.
Types and constants are not visibility-marked at all; if a type is
conceptually internal, just don't mention it in API docs and trust the
prefix-free naming.

**Don't use opaque abbreviations.** `batched_matmul`, not `bmm`. Be
explicit.

### API stability

- Public procs that must stay ergonomic for callers (e.g. `linear`,
  `update`, `parallelize`, `attention`) keep their existing signatures.
  Optimizations and refactors live behind these signatures.
- The unified API path (`ml.add(a, b)` etc.) is backend-agnostic. Don't
  add backend-specific overloads on the public surface.
- Legacy GPU op procs (`gpu.linear`, `gpu.attention`, etc. on `GpuTensor`)
  exist transitionally. Don't add new ones — port to the unified API
  via `Backend` dispatch instead.

## Shader build

After editing a `.comp` file, recompile to `.spv`:

    cd gpu/shaders && for f in *.comp; do glslc -O "$f" -o "${f%.comp}.spv" || break; done

(or just the file you changed). The `.spv` files are checked in so the
project builds without `glslc` on other machines.
