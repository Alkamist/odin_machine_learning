"""Speed comparison: Odin ML library vs PyTorch on matched shapes.

Runs the Odin speed_runner binary, runs the equivalent PyTorch benchmarks
in-process, and prints a side-by-side table.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR  = Path(__file__).parent.resolve()
RUNNER_EXE  = SCRIPT_DIR / "speed_runner.exe"

WARMUP     = 3
ITERATIONS = 30

LINEAR_BATCH, LINEAR_INPUT, LINEAR_OUTPUT = 64, 512, 2048
LAYERNORM_BATCH, LAYERNORM_SIZE           = 64, 512
SOFTMAX_BATCH, SOFTMAX_SIZE               = 64, 1024
ATTN_TOKENS, ATTN_HEADS, ATTN_EMBED       = 256, 8, 512
MLP_BATCH, MLP_IN, MLP_HID, MLP_OUT       = 64, 256, 256, 64


_DEVICE = "cpu"

def _sync():
    if _DEVICE == "cuda":
        torch.cuda.synchronize()

def time_run(run_fn):
    for _ in range(WARMUP):
        run_fn()
        _sync()

    min_ns   = 10**18
    total_ns = 0
    for _ in range(ITERATIONS):
        _sync()
        start = time.perf_counter_ns()
        run_fn()
        _sync()
        dt = time.perf_counter_ns() - start
        if dt < min_ns:
            min_ns = dt
        total_ns += dt
    return min_ns / 1e6, total_ns / ITERATIONS / 1e6


def _t(*args, **kwargs):
    return torch.tensor(*args, **kwargs, device=_DEVICE) if "device" not in kwargs else torch.tensor(*args, **kwargs)


def _full(shape, value, **kwargs):
    return torch.full(shape, value, device=_DEVICE, **kwargs)


def _randn(*shape, **kwargs):
    return torch.randn(*shape, device=_DEVICE, **kwargs)


def _ones(shape, **kwargs):
    return torch.ones(shape, device=_DEVICE, **kwargs)


def _triu_bool(n):
    return torch.triu(torch.ones(n, n, dtype=torch.bool, device=_DEVICE), diagonal=1)


def torch_linear_fwd():
    x = _full((LINEAR_BATCH, LINEAR_INPUT), 0.01)
    w = _randn(LINEAR_OUTPUT, LINEAR_INPUT) * 0.02

    def run():
        torch.nn.functional.linear(x, w)

    return time_run(run)


def torch_linear_fwdbwd():
    w = (_randn(LINEAR_OUTPUT, LINEAR_INPUT) * 0.02).requires_grad_(True)

    def run():
        x = _full((LINEAR_BATCH, LINEAR_INPUT), 0.01, requires_grad=True)
        if w.grad is not None: w.grad = None
        y = torch.nn.functional.linear(x, w)
        y.sum().backward()

    return time_run(run)


def torch_layernorm():
    w = _ones(LAYERNORM_SIZE, requires_grad=True)

    def run():
        x = _full((LAYERNORM_BATCH, LAYERNORM_SIZE), 0.01, requires_grad=True)
        if w.grad is not None: w.grad = None
        y = torch.nn.functional.layer_norm(x, [LAYERNORM_SIZE], w, None, 1e-5)
        y.sum().backward()

    return time_run(run)


def torch_softmax():
    def run():
        x = _full((SOFTMAX_BATCH, SOFTMAX_SIZE), 0.01, requires_grad=True)
        y = torch.nn.functional.softmax(x, dim=-1)
        y.sum().backward()

    return time_run(run)


def torch_attention_causal():
    head_size = ATTN_EMBED // ATTN_HEADS
    mask = _triu_bool(ATTN_TOKENS)

    def run():
        x = _full((ATTN_TOKENS, 3 * ATTN_EMBED), 0.01, requires_grad=True)
        q_flat = x[:, 0:ATTN_EMBED]
        k_flat = x[:, ATTN_EMBED:2 * ATTN_EMBED]
        v_flat = x[:, 2 * ATTN_EMBED:3 * ATTN_EMBED]
        q = q_flat.reshape(ATTN_TOKENS, ATTN_HEADS, head_size).permute(1, 0, 2)
        k = k_flat.reshape(ATTN_TOKENS, ATTN_HEADS, head_size).permute(1, 0, 2)
        v = v_flat.reshape(ATTN_TOKENS, ATTN_HEADS, head_size).permute(1, 0, 2)
        scores = torch.matmul(q, k.transpose(1, 2)) / (head_size ** 0.5)
        scores = scores.masked_fill(mask, float("-inf"))
        attn = torch.nn.functional.softmax(scores, dim=-1)
        out  = torch.matmul(attn, v).permute(1, 0, 2).reshape(ATTN_TOKENS, ATTN_EMBED)
        out.sum().backward()

    return time_run(run)


def torch_mlp_step():
    model = torch.nn.Sequential(
        torch.nn.Linear(MLP_IN,  MLP_HID), torch.nn.ReLU(),
        torch.nn.Linear(MLP_HID, MLP_HID), torch.nn.ReLU(),
        torch.nn.Linear(MLP_HID, MLP_OUT),
    ).to(_DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    x = _full((MLP_BATCH, MLP_IN),  0.01)
    y = _full((MLP_BATCH, MLP_OUT), 0.5)

    def run():
        optimizer.zero_grad()
        pred = model(x)
        loss = ((pred - y) ** 2).mean()
        loss.backward()
        optimizer.step()

    return time_run(run)


TORCH_BENCHES = [
    ("linear_fwd",       torch_linear_fwd),
    ("linear_fwdbwd",    torch_linear_fwdbwd),
    ("layernorm",        torch_layernorm),
    ("softmax",          torch_softmax),
    ("attention_causal", torch_attention_causal),
    ("mlp_step",         torch_mlp_step),
]


def run_odin(backend: str, threads: int) -> dict:
    if not RUNNER_EXE.exists():
        print(f"speed runner not built: {RUNNER_EXE}", file=sys.stderr)
        sys.exit(1)
    result = subprocess.run(
        [str(RUNNER_EXE), backend, str(threads)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        sys.stderr.write(result.stdout)
        sys.stderr.write(result.stderr)
        sys.exit(1)
    out = {}
    for line in result.stdout.strip().splitlines():
        if "," not in line:
            continue
        name, min_ms, mean_ms, _checksum = line.split(",")
        out[name] = (float(min_ms), float(mean_ms))
    return out


def main():
    global _DEVICE
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--device",  choices=["cpu", "cuda"], default=None,
                        help="PyTorch device. Defaults to cuda when --backend=gpu, cpu otherwise.")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()

    if args.device is None:
        args.device = "cuda" if args.backend == "gpu" else "cpu"
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available", file=sys.stderr)
        sys.exit(1)

    _DEVICE = args.device

    if args.device == "cpu":
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)

    label = f"odin={args.backend}  torch={args.device}"
    if args.backend == "cpu":
        label += f"  threads={args.threads}"
    print(f"{label}  warmup={WARMUP}  iterations={ITERATIONS}")
    print(f"{'benchmark':<20} {'odin min':>10} {'torch min':>10} {'ratio':>8}   {'odin mean':>10} {'torch mean':>10} {'ratio':>8}")
    print("-" * 92)

    odin_results = run_odin(args.backend, args.threads)

    for name, fn in TORCH_BENCHES:
        torch_min, torch_mean = fn()
        odin_min, odin_mean   = odin_results[name]
        min_ratio  = odin_min  / torch_min  if torch_min  > 0 else float("inf")
        mean_ratio = odin_mean / torch_mean if torch_mean > 0 else float("inf")
        print(
            f"{name:<20} "
            f"{odin_min:>9.3f}ms {torch_min:>9.3f}ms {min_ratio:>7.2f}x   "
            f"{odin_mean:>9.3f}ms {torch_mean:>9.3f}ms {mean_ratio:>7.2f}x"
        )


if __name__ == "__main__":
    main()
