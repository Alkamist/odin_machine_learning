## Intro

This is a machine learning library written completely from scratch. It has
two compute backends — a CPU backend (in `ml.odin`) and a Vulkan compute
backend (in `gpu/`) — that share a single `Backend`-dispatched op system.

Models written in the library use the same `ml.*` calls regardless of
backend. The active backend is held on the active `Context` and
determines whether `ml.zeros` allocates host memory or a Vulkan buffer,
and whether `ml.add` dispatches a SIMD CPU loop or a compute shader.


## Current State

**Architecture:** Per-context state with a thread-local linked-list stack.
Backend abstraction via tagged dispatch (one `forward(op)` and one
`backward(op)` proc per backend, switching on `op.variant`) plus
hooks for: activation alloc (`alloc`, `clear_storage`), long-lived
tensors (`persistent_alloc`, `persistent_free`), parameters with Adam
state (`parameter_alloc`, `parameter_free`, `parameter_update`,
`parameter_copy`), per-Context backend state (`context_alloc`,
`context_free`, `context_begin`, `context_end`), in-flight work
(`flush`), and host↔device copies (`set_data`, `get_data`,
`fill_gradient_with_ones`). `ml.Tensor` carries a `backend: ^Backend`
and an opaque `storage: rawptr`; CPU stores data in host slices, GPU
stores it via a `gpu.Gpu_Storage` containing `vk.Buffer` handles.
Thread-safe: each host thread owns its own Context; the shared CPU
worker pool serializes `parallelize` fan-outs with a mutex.

**Switching backends is a one-line change.** CPU and GPU look identical
at the call site — pick a backend at context creation and forget it:

    // CPU
    ctx := ml.context_create(size)

    // GPU — `gpu.backend()` lazy-inits Vulkan on first call, and
    // `ml.context_create` allocates + binds the per-thread Gpu_Context
    // automatically via Backend.context_alloc.
    ctx := ml.context_create(size, gpu.backend())

There's no `gpu.init` / `gpu.context_create` / `gpu.context_scope` in
model code, and no `gpu.begin_batch` / `gpu.end_batch` either — every
GPU dispatch auto-opens a batch CB, and `ml.clear` / `Backend.get_data`
flush it implicitly. `ml.sync()` is exposed for cases that need an
explicit fence (timing benchmarks, etc.).

**Op port status (unified API):**

CPU backend — all 30 ops dispatch through `cpu_forward` / `cpu_backward`.

GPU backend — all 30 ops ported and tested via `gpu_unified_check`:
- Linear / batched: `add`, `sub`, `mul` (with broadcast), `div`
  (broadcast), `min`, `max`, `linear`, `batched_matmul`
- Activations: `gelu`, `relu`, `sigmoid`, `silu`, `tanh`, `exp`, `clamp`
- Shape: `permute`, `transpose`, `slice`, `slice_trailing`, `concat`
  (3-input only)
- Reductions / loss heads: `softmax`, `log_softmax`, `causal_mask`,
  `layernorm`, `mean`, `entropy`, `mean_squared_error`,
  `cross_entropy`
- Embedding / position: `select`, `rope`

**End-to-end GPU training works.** A `tfm.Transformer` allocated under
a GPU `ml.Context` has its parameters (`data`/`gradient`/`adam_m`/
`adam_v`) on device. The full training step — forward + backward +
Adam(W) update — runs as a single batched command buffer per step.
`gpu_unified_check` phase 18 verifies CPU vs GPU parameter values
match within the fp32 reduction floor after 3 Adam steps on a small
network. `examples/gpu_text_generation_transformer` is the working
demonstration.

**`gpu_transformer_bench` numbers (RTX 3090 Ti):**
- Small (L=4 H=4 E=128 V=256 T=64): GPU full step 7.2 ms vs CPU
  10.8 ms (~1.5×).
- Large (L=12 H=8 E=512 V=256 T=256): GPU full step 342 ms vs CPU
  1202 ms (~3.5×). GPU forward alone is ~7× faster here. Adam adds
  essentially zero overhead on GPU (342.3 vs 342.4 ms with vs without).

At the small architecture the per-dispatch driver overhead still
dominates compute — bigger models pay off more.

**Attention is decomposed.** There is no `Attention` op variant. Both
backends compute attention as a composition of `slice_trailing` →
`reshape` → `permute` → `batched_matmul` → `mul` → `causal_mask` →
`softmax` → `batched_matmul` → `permute` → `reshape`. Both backends
run this path end-to-end and match at the fp32 floor (verified by
`gpu_unified_check` phase 14).

## Next steps (where to pick up)

1. **Split the CPU backend out of `ml.odin`.** Move the CPU-only
   pieces — `cpu_alloc` / `cpu_clear_storage` / `cpu_set_data` /
   `cpu_fill_gradient_with_ones`, `cpu_forward` / `cpu_backward` and
   every `cpu_*` op kernel, the SIMD primitives (`_simd_dot_f32`,
   `_simd_axpy_f32`), and the `parallelize` worker pool — into a
   separate file (or `cpu/` subpackage), leaving `ml.odin` with the
   backend-agnostic surface (`Backend`, `Context`, `Tensor`, op
   variants, public op constructors, autograd tape, optimizer). Keep
   the CPU backend the default; mirrors how `gpu/` is already its own
   package.

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

**GPU perf:** `benchmarks/gpu_transformer_bench` runs the same
transformer step on CPU and GPU at small + large architectures and
reports forward / forward+backward / full-step (with Adam) timings.

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
device_name. Lazy-initialized by `gpu.backend()` on first call (also
explicitly via `gpu.init()` if needed). The Vulkan instance / device
live for the rest of the process — there's no shutdown call in the
public API.

`Gpu_Context` (thread-local stack): command_pool, descriptor_pool,
batch state, allocations list, per-context buffer pool, persistent
staging buffer. One per `ml.Context` that uses the GPU. Vulkan command
pools and descriptor pools are NOT thread-safe, so each host thread
that issues GPU work owns its own `Gpu_Context`.

The `Gpu_Context` lifecycle is bound to its owning `ml.Context` via
the `Backend.context_alloc` / `_free` / `_begin` / `_end` hooks. From
model code that's invisible — same three lines as CPU:

```odin
ctx := ml.context_create(N, gpu.backend())
defer ml.context_destroy(ctx)
ml.context_scope(ctx)
```

`gpu.context_create` / `gpu.context_scope` are still available as a
power-user escape hatch for code that wants multiple `Gpu_Context`s
on a single `ml.Context` (rare).

### `Gpu_Storage` and the `Backend.alloc` hook

`gpu.backend()` returns a `^ml.Backend` whose `alloc` hook creates a
pair of DEVICE_LOCAL buffers (data + grad), stashes a `^Gpu_Storage` in
`Tensor.storage`, and registers it on the active `Gpu_Context`'s
allocations list. Both buffers are zeroed via a one-shot
`vkCmdFillBuffer` on alloc to match CPU's `make([]f32, n)` semantics —
critical for backward (which `+=` into gradient buffers).

**One-shot writes during a recording batch are a footgun.** When a
batch CB is open, the batch may already have a future-executing
`CmdFillBuffer` (e.g. an alloc-time zero-fill from `gpu_alloc`)
queued for a buffer. If something then issues a one-shot
submit-and-wait that writes the same buffer, the one-shot completes
*before* the batch is submitted — so when the batch finally runs, its
earlier-recorded fill clobbers the one-shot's write. All buffer
writes that need to be ordered against in-batch dispatches must
record into the batch CB. Use `_record_fill` / `_record_fill_zero`;
the per-dispatch barrier covers the resulting `TRANSFER_WRITE`s. The
same goes for any future host→device upload helper that gets called
mid-batch.

`clear_storage` doesn't actually destroy the GPU storage — it pushes
every live allocation back onto a per-context **pool keyed by element
count**. The next `gpu_alloc` for the same `n` pops a recycled buffer
pair. This is the dominant perf fix on the GPU side: a transformer
step makes ~30+ activation tensors and prior to pooling each one paid
a full `vkCreateBuffer` + `vkAllocateMemory` round-trip per step. After
the first cycle the pool is warm and `gpu_alloc` only runs a
`CmdFillBuffer` to re-zero the recycled buffers. The fill records into
the active batch command buffer (free; no extra submit), and the
per-dispatch global memory barrier covers `TRANSFER_WRITE`s so the
zeroes are visible to subsequent shader reads. Net effect on
`gpu_transformer_bench` at L=12 E=512 T=256: GPU forward 268 ms → 31 ms
(8.7×), GPU forward+backward 553 ms → 346 ms; on the
`gpu_text_generation_transformer` example, ~10 tok/sec → ~230 tok/sec.

### Command-buffer batching (in `gpu/pipeline.odin`)

Every `_dispatch` call records into a single open command buffer with
a global memory barrier
(`{SHADER_WRITE, TRANSFER_WRITE} → {SHADER_READ, SHADER_WRITE}` on
COMPUTE+TRANSFER → COMPUTE) before each dispatch — so successive ops
see prior writes without per-buffer dependency tracking.

Batches are opened automatically: the first `_dispatch` after a flush
calls `begin_batch()` itself, and the batch is closed by
`Backend.flush()` (called from `ml.clear`, `Backend.get_data`, and
`ml.sync()`). `end_batch` does one queue submit + one `vkQueueWaitIdle`.
The user-facing API never exposes the open/close points; the explicit
`gpu.begin_batch` / `gpu.end_batch` calls remain only as a power-user
escape hatch for code that wants to span multiple `ml.clear`s in one
batch (rare).

Transient resources (descriptor sets, the `select` indices buffer) are
queued via `_queue_destroy_buffer` and reclaimed in `end_batch`.

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

Verification bar: `gpu_unified_check` proves every op matches CPU at
fp32 floor (op-specific tolerances, mostly 1e-7 to 1e-8) plus a
multi-step training-loop phase that confirms parameter values agree
after several Adam updates.

## Project layout

- `ml.odin`: CPU primitives, autograd tape, `Context` / `Backend` / op
  procs, Adam optimizer, worker pool, SIMD kernels.
- `mlp/`, `gru/`, `transformer/`: model implementations (backend-agnostic
  via the unified API).
- `gpu/`: Vulkan compute backend.
  - `gpu.odin`: instance / device init (lazy via `gpu.backend()`);
    `Gpu_Context` lifecycle (per-`ml.Context` state — command pool,
    descriptor pool, batch state, allocations, buffer pool, staging).
  - `buffer.odin`: low-level Vulkan buffer helpers (`_create_buffer`,
    `_one_shot_copy`, `_record_fill`, `_ensure_staging`).
  - `pipeline.odin`: pipeline construction, `_dispatch` (auto-batches),
    command-buffer batching.
  - `backend.odin`: `ml.Backend` integration — `Gpu_Storage`, every
    backend hook (`gpu_alloc` / `gpu_persistent_alloc` /
    `gpu_parameter_*` / `gpu_set_data` / `gpu_get_data` /
    `gpu_context_*` / `gpu_flush`), every op's `gpu_*_forward` /
    `gpu_*_backward`, and `upload_tensor` / `download_tensor` helpers.
  - `ops.odin`: SPIRV `#load` constants, push-constant struct
    definitions, and lazily-bound pipeline pointers.
  - `shaders/`: GLSL compute shaders + compiled `.spv` (checked in).
- `benchmarks/`:
  - `benchmark`: canonical CPU perf + ST checksums.
  - `thread_safety_check`: 4-host-thread stress test of the worker pool.
  - `gpu_unified_check`: per-op CPU vs GPU correctness on the unified
    API plus a multi-step training-loop verification.
  - `gpu_transformer_bench`: end-to-end transformer per-step timing,
    CPU vs GPU, at small + large architectures.
- `examples/`: `mnist`, `imitation_learning`, `ppo`, `text_generation_gru`,
  `text_generation_transformer` (CPU), `gpu_text_generation_transformer`
  (single GPU context — train + generate end-to-end on device).
- `transformer/`: model definition. `Transformer` is `Parameter`-based
  and works on either backend — `tfm.make` allocates parameters via
  whatever backend the active `ml.Context` uses, `tfm.forward` is
  unified-API ops, `tfm.update` calls `ml.update` which dispatches the
  backend's `parameter_update` hook.

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
  add backend-specific overloads on the public surface — every new op
  goes through `Backend.forward` / `Backend.backward` dispatch.

## Shader build

After editing a `.comp` file, recompile to `.spv`:

    cd gpu/shaders && for f in *.comp; do glslc -O "$f" -o "${f%.comp}.spv" || break; done

(or just the file you changed). The `.spv` files are checked in so the
project builds without `glslc` on other machines.
