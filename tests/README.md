# Test suites

Three packages, run independently:

```
odin test tests            # CPU-only: gradient checks, op/optimizer/loader/checkpoint tests
odin test tests\golden     # fixture-based goldens: tokenizers, quant dequantization, fixed-weight MLP
odin test tests\parity     # CPU vs CUDA parity; needs an NVIDIA GPU with driver + NVRTC
```

Useful defines:

- `-define:ML_CPU_POISON=true` — fills transient CPU `.Data` buffers with NaN to enforce the "ops fully overwrite their output" contract. The `tests` suite is expected to stay green under it.
- `-define:ML_REQUIRE_CUDA=true` — makes the parity suite fail loudly instead of silently skipping when no CUDA device is present. Use on machines that are supposed to have a GPU.
- `-define:ODIN_TEST_NAMES=pkg.test_name,...` — run a subset.

Structure:

- `tests/cases/` — the shared op-case registry. Each entry drives both the CPU central-difference gradient check (`grad_check.odin`) and the CPU↔CUDA parity sweep. `parity_only` entries (large shapes that cross CUDA block/tile boundaries) are skipped by the gradient check; `parity_tol` overrides the parity tolerance for accumulation-order-sensitive cases (cuBLAS matmuls).
- `tests/parity/` — parity for the registry cases plus Bf16 forward/backward sweeps (`parity_bf16.odin`), fused/quant op oracles vs their decompositions (`fused_gpu_check.odin`), Adam and gradient-clip parity. The parity gate asserts every registry case's op is in CUDA `forward_ops`, so adding a case guarantees parity coverage.
- `tests/golden/` — reference fixtures generated outside the library (HF tokenizers, llama.cpp quant block spec). Tests log a loud skip when optional model assets are absent.

- `tests/parallel_inference_check.odin` and `tests/parity/parallel_gpu_check.odin` — the Tier 2 threading-contract tests: shared read-only weights, one context per thread, outputs bit-identical to a single-threaded reference (see the threading contract in the root `README.md`).
