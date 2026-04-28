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


def time_run(run_fn):
    for _ in range(WARMUP):
        run_fn()

    min_ns   = 10**18
    total_ns = 0
    for _ in range(ITERATIONS):
        start = time.perf_counter_ns()
        run_fn()
        dt = time.perf_counter_ns() - start
        if dt < min_ns:
            min_ns = dt
        total_ns += dt
    return min_ns / 1e6, total_ns / ITERATIONS / 1e6


def torch_linear_fwd():
    x = torch.full((LINEAR_BATCH, LINEAR_INPUT), 0.01)
    w = torch.randn(LINEAR_OUTPUT, LINEAR_INPUT) * 0.02

    def run():
        torch.nn.functional.linear(x, w)

    return time_run(run)


def torch_linear_fwdbwd():
    w = (torch.randn(LINEAR_OUTPUT, LINEAR_INPUT) * 0.02).requires_grad_(True)

    def run():
        x = torch.full((LINEAR_BATCH, LINEAR_INPUT), 0.01, requires_grad=True)
        if w.grad is not None: w.grad = None
        y = torch.nn.functional.linear(x, w)
        y.sum().backward()

    return time_run(run)


def torch_layernorm():
    w = torch.ones(LAYERNORM_SIZE, requires_grad=True)

    def run():
        x = torch.full((LAYERNORM_BATCH, LAYERNORM_SIZE), 0.01, requires_grad=True)
        if w.grad is not None: w.grad = None
        y = torch.nn.functional.layer_norm(x, [LAYERNORM_SIZE], w, None, 1e-5)
        y.sum().backward()

    return time_run(run)


def torch_softmax():
    def run():
        x = torch.full((SOFTMAX_BATCH, SOFTMAX_SIZE), 0.01, requires_grad=True)
        y = torch.nn.functional.softmax(x, dim=-1)
        y.sum().backward()

    return time_run(run)


def torch_attention_causal():
    head_size = ATTN_EMBED // ATTN_HEADS
    mask = torch.triu(torch.ones(ATTN_TOKENS, ATTN_TOKENS, dtype=torch.bool), diagonal=1)

    def run():
        x = torch.full((ATTN_TOKENS, 3 * ATTN_EMBED), 0.01, requires_grad=True)
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
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    x = torch.full((MLP_BATCH, MLP_IN),  0.01)
    y = torch.full((MLP_BATCH, MLP_OUT), 0.5)

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


def run_odin(threads: int) -> dict:
    if not RUNNER_EXE.exists():
        print(f"speed runner not built: {RUNNER_EXE}", file=sys.stderr)
        sys.exit(1)
    result = subprocess.run([str(RUNNER_EXE), str(threads)], capture_output=True, text=True)
    if result.returncode != 0:
        sys.stderr.write(result.stdout)
        sys.stderr.write(result.stderr)
        sys.exit(1)
    out = {}
    for line in result.stdout.strip().splitlines():
        name, min_ms, mean_ms, _checksum = line.split(",")
        out[name] = (float(min_ms), float(mean_ms))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)

    print(f"threads={args.threads}  warmup={WARMUP}  iterations={ITERATIONS}")
    print(f"{'benchmark':<20} {'odin min':>10} {'torch min':>10} {'ratio':>8}   {'odin mean':>10} {'torch mean':>10} {'ratio':>8}")
    print("-" * 92)

    odin_results = run_odin(args.threads)

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
