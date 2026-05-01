## Instructions

- ALWAYS reference `odin_skill`.
- Pay close attention to the style of the code in ml.odin and backends/cpu/, those represent good style.
- Be very picky about adding comments.
- Use descriptive variable names.

## Project Layout

Adjust this when as it changes.

- `ml.odin` — core public interface for the library: tensors, ops, autograd, optimizers.
- `backends/` — compute backends implementing the ops surface.
  - `cpu/cpu.odin` — reference CPU backend.
  - `gpu/` — GPU backend (`backend.odin`, `buffer.odin`, `gpu.odin`, `ops.odin`, `pipeline.odin`) with compute kernels under `shaders/` (one subdirectory per op, plus `bf16.glsl` and `build.bat`).
- `networks/` — reusable network building blocks.
  - `mlp/mlp.odin`, `gru/gru.odin`, `transformer/transformer.odin`.
  - `llama/` — Llama-style transformer (`llama.odin`) with safetensors weight loading (`loader.odin`).
- `loaders/` — file-format loaders.
  - `safetensors/safetensors.odin` — safetensors reader.
- `tokenizers/` — tokenizer implementations.
  - `gpt2/gpt2.odin` — GPT-2 BPE tokenizer.
- `examples/` — runnable demos and training scripts: `mnist`, `circles`, `cartpole`, `ppo`, `imitation_learning`, `freeplay`, `text_generation_gru`, `text_generation_transformer`, `smollm_chat`, `smollm_inference`, plus shared `data/` and `utility/`.
- `tests/` — correctness and performance checks: `pytorch_parity`, `benchmark`, `bf16_linear_bench`, `dtype_roundtrip`, `gpt2_tokenizer`, `gpu_transformer_bench`, `gpu_unified_check`, `smollm_smoke`, `thread_safety_check`.
- `tools/` — supporting scripts (e.g. `smollm_dump.py`).
- `smollm_data/` — SmolLM model weights and assets used by the smollm examples/tests.
- `TODO.md` — running task list.