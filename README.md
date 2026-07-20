# machine_learning

An Odin machine-learning library: define-by-run tensor graphs with reverse-mode
autodiff, a threaded CPU backend, a CUDA backend with graph-captured decode, and
enough model plumbing (loaders, tokenizers, sampling, networks) to train small
models and run quantized LLMs locally.

## Layout

- Root package (`ml`): tensors, the operation tape, autodiff (`backward`), the
  optimizer, parameter registry, checkpointing, KV-cache types.
- `backends/cpu`, `backends/cuda`: implementations of the `ml.Backend` proc
  table. The CUDA backend NVRTC-compiles kernels from `backends/cuda/kernels/`.
- `loaders/`: gguf, safetensors, and a `weights` convenience layer over both.
- `tokenizers/`: gemma (SentencePiece-style BPE) and gpt2 (byte-level BPE with
  a Digits pretokenizer, matching SmolLM2's tokenizer.json).
- `networks/`: gemma, llama, mlp, jepa, lora.
- `sampling/`: temperature/top-k/top-p sampling and a generate loop.
- `examples/`: mnist, llm_chat, cartpole (interactive + headless), fetch.
- `tests/`: see `tests/README.md`.

## Context and pass discipline

All tensor work happens against a thread-local active context:

```odin
ctx := cpu.context_create(1024 * 1024 * 256)
defer cpu.context_destroy(ctx)
ml.context_scope(ctx)
```

Each unit of graph building is a pass. `ml.pass()` (inference) or
`ml.pass(training=true)` clears the tape and scopes the pass to the enclosing
block; recording ops after the scope closes is an assert. `ml.pass_begin(...)` is the
unscoped equivalent for code that manages pass boundaries manually. A training
pass builds a graph, reduces to a scalar loss, then calls `ml.backward(loss)`.

Optimizer updates gate on gradient accumulation:

```odin
ml.backward(loss)
stepped := ml.registry_step(&opt, &registry, max_grad_norm=1.0)
```

or with the lower-level pieces: `if ml.optimizer_step(&opt) { update(...) }`.
Calling `ml.update` before the first `ml.optimizer_step` is an assert.

## Threading contract

- A context may be active on at most one thread at a time (`context_begin`
  enforces this atomically); use one context per thread.
- Weight tensors may be shared read-only across threads, each thread running
  its own context over the same parameters. `tests/parallel_inference_check.odin`
  and `tests/parity/parallel_gpu_check.odin` assert outputs are bit-identical
  to a single-threaded reference under this pattern.
- Mutating parameters (training, `lerp_assign`, loading) while another thread
  reads them is a data race; synchronize externally.

## Backend contract

A backend fills the `ml.Backend` proc table and declares its capabilities in
`forward_ops` / `backward_ops`; the core asserts before dispatching anything a
backend did not declare.

- `buffer_alloc(persist=true)` allocations live until `buffer_free` or context
  destruction; a backend's `context_destroy` must free all remaining persistent
  buffers (both backends sweep them).
- Gradient buffers are zero-initialized at allocation and accumulated with `+=`
  in backward implementations; `clear` resets transient state per pass.
- KV-cache writes are explicit in the recorded graph: `Attention_Cache` carries
  `k_cached` / `v_cached` telling the backend which cache rows were already
  written this pass (by `Rmsnorm_Rope_Write_Cache` or by a KV-shared source
  layer). A backend must write exactly the rows those flags leave to it.
- `rmsnorm_rope_write_cache` returns `wrote_cache`; callers thread that into
  `attention_with_cache(k_already_cached=...)`.

## Sampling eval contract

`sampling.generate` drives an `Eval_Proc`; during chunked prefill it passes
`logits_out = nil` for every chunk except the last, and eval implementations
must tolerate that (skip the readback). Network `forward_cached` takes
`logits_mode=.Last` to compute the lm_head only for the final position.
Sampling draws from `context.random_generator`; seed it for reproducible
generation (see `tests/sampling_check.odin`).

## Building

`odin check <pkg> -no-entry-point` for library packages. Optimized builds:
`-o:speed`, plus `-microarch:x86-64-v3` to enable the CPU SIMD paths (the
`tests` suite is kept green under that flag). CUDA needs an NVIDIA driver and
NVRTC at runtime; `examples/llm_chat` selects the backend with
`-define:ML_BACKEND=cuda`.
