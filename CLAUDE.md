## Intro

This is a machine learning library written completely from scratch, currently designed only for the CPU. The multi-threaded code is not that optimized, and I know from using GGML that it is possible to run a lot faster on the CPU.

## Current Goal

Use ggml as a reference to develop much faster code on the CPU.

## Testing

You MUST confirm via testing that code changes actually make the code faster.

The benchmark in `benchmark/main.odin` is the canonical way to verify perf
changes. Use the single-threaded checksum column to verify correctness across
versions (within ~1 ULP — SIMD reduction order can shift the last bit).

## Build

Builds must pass `-microarch:native` (or another flag enabling AVX2 + FMA) to
get the FMA-vectorized inner loops in `_simd_dot_f32` / `_simd_axpy_f32`.
Without it the SIMD lanes still vectorize but FMA falls back to mul+add and
performance drops ~30%. Example:

    odin build benchmark -o:speed -no-bounds-check -microarch:native -out:benchmark/benchmark.exe

## Code style

- File-local helpers and globals use a leading underscore (`_worker_proc`,
  `_dispatch`, `_simd_dot_f32`). **Don't use `@(private)` or `@(private="file")`** —
  underscore prefix is the convention here.
- Public procs that must stay ergonomic for callers (e.g. `linear`, `update`,
  `parallelize`) keep their existing signatures. Optimizations live behind
  these signatures, not in API changes.

## Architecture notes

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

- `ml.odin`: Holds all of the machine-learning related primitive code.
- `mlp/`: Holds an MLP implementation.
- `gru/`: Holds a GRU implementation.
- `transformer/`: Holds a transformer implementation.
- `benchmark/`: Microbenchmarks + end-to-end transformer step timing.