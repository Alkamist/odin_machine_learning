# machine_learning

An Odin machine-learning library: a tensor/autograd core (`ml`) with CPU and CUDA backends, plus packages for networks, loaders, tokenizers, sampling, and datasets.

## Building and testing

- `odin check . -no-entry-point` (root library), same per package.
- `odin test tests` — CPU suite. `-define:ML_CPU_POISON=true` fills transient CPU buffers with NaN to catch reads-before-writes.
- `odin test tests\golden` — reference-value tests (some skip when model assets are absent).
- `odin test tests\parity` — CPU-vs-CUDA parity; needs a CUDA GPU. `-define:ML_REQUIRE_CUDA=true` turns the no-GPU skip into a failure.
- Optimized builds: `-o:speed`.

## Threading contract

- **Tier 1 (guaranteed):** a `Context` and every transient tensor created under it belong to one thread at a time. Each thread creates its own context (`cpu.context_create` / `cuda.context_create`) and activates it with `context_begin`/`context_scope`. Contexts are ownership-checked: activating a context that is active on another thread, or destroying one that is still active anywhere, asserts. All library-internal globals (the NVRTC pipeline cache, the CUDA primary context and its refcount, the CPU worker pool) are internally synchronized — N threads each with their own context need no user locking.
- **Tier 2 (guaranteed with a rule):** persistent tensors (parameters, anything from `alloc(persistent=true)`) may be shared read-only across threads for inference. Writes must be published before other threads read. Publish points: `set_data`/`checkpoint_load`/loader writes are synchronous — they have published when they return. A persistent tensor written by *ops* (e.g. `lerp_assign`) is published once the writing context completes `ml.clear()` or any `get_data` (both synchronize the writing context's stream). `tests/parallel_inference_check.odin` and `tests/parity/parallel_gpu_check.odin` exercise this tier.
- **Tier 3 (non-goal):** concurrent training on shared parameters. Backward accumulates gradients and optimizer state is unsynchronized — data-parallel training is per-thread replicas plus an explicit reduce, owned by the caller.
- The CPU worker pool is a single global pool; concurrent `parallelize` calls from multiple contexts serialize behind its mutex by design (prevents oversubscription). Calling into an op from inside a `parallelize` job asserts — it would deadlock the pool.

## Other library contracts

- **RNG:** all library randomness (`fill_normal`, dropout masks, `randomize`, `sampling.sample`, `dataset` shuffles) draws from `context.random_generator`. Install a seeded generator on the thread for reproducibility; the default generator is per-thread, so threads never race on RNG state.
- **Transient buffers:** transient `.Data` buffers are uninitialized storage — every op fully overwrites its output and never reads it before writing. `.Gradient` and persistent buffers are zeroed on allocation.
- **Optimizer:** `Backend.update` zeroes the tensor's gradient as a side effect on both backends; one accumulation window ends exactly at `update`.
- **Gradient sinks:** a tensor without a gradient buffer is a gradient sink — backward passes skip accumulation into it. This is what makes frozen weights and `scratch` inputs safe, and gives stop-gradient semantics for free.
- **Failures:** file/checkpoint I/O returns errors (`gguf.Error`, `safetensors.Error`, `ml.Checkpoint_Error`, tokenizer `Error`). Shape/dtype misuse asserts with the caller's location. Backend-internal CUDA/driver/NVRTC failures panic — they are not recoverable mid-graph.
- **Device-lifetime state:** the CUDA pipeline cache and CPU worker-pool structures are allocated with the default heap allocator, not the ambient `context.allocator` — they outlive any caller's allocator scope and are invisible to per-scope leak tracking by design.
