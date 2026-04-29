## Instructions

- Use the odin Skill.
- Pay close attention to the style of the code in ml.odin and backends/cpu/, those represent good style.
- Be very picky about adding comments.
- Use descriptive variable names.

## Project Layout

Adjust this when as it changes.

- `ml.odin` — core public interface for the library: tensors, ops, autograd, optimizers.
- `backends/` — compute backends implementing the ops surface.
  - `cpu/cpu.odin` — reference CPU backend.
  - `gpu/` — GPU backend (`backend.odin`, `buffer.odin`, `gpu.odin`, `ops.odin`, `pipeline.odin`) with WGSL/compute kernels under `shaders/`.
- `networks/` — reusable network building blocks.
  - `mlp/mlp.odin`, `gru/gru.odin`, `transformer/transformer.odin`.
- `examples/` — runnable demos and training scripts: `mnist`, `circles`, `cartpole`, `ppo`, `imitation_learning`, `freeplay`, `text_generation_gru`, `text_generation_transformer`, plus shared `data/` and `utility/`.
- `tests/` — correctness and performance checks: `pytorch_parity`, `benchmark`, `gpu_transformer_bench`, `gpu_unified_check`, `thread_safety_check`.
- `TODO.md` — running task list.