"""PyTorch parity tests for the Odin ML library.

Generates random inputs, computes PyTorch reference outputs and gradients,
invokes the Odin runner, then compares the two using np.allclose.
"""

import argparse
import os
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR    = Path(__file__).parent.resolve()
ARTIFACTS_DIR = SCRIPT_DIR / "artifacts"
RUNNER_EXE    = SCRIPT_DIR / "runner.exe"

MAGIC = b"TNSR"
SEED  = 0xC0FFEE
ATOL  = 1e-5
RTOL  = 1e-4


def write_tensor(path: Path, array: np.ndarray) -> None:
    array = np.ascontiguousarray(array, dtype=np.float32)
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<I", array.ndim))
        for d in array.shape:
            f.write(struct.pack("<I", d))
        f.write(array.tobytes())


def read_tensor(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        data = f.read()
    assert data[:4] == MAGIC, f"bad magic in {path}"
    rank = struct.unpack_from("<I", data, 4)[0]
    shape = [struct.unpack_from("<I", data, 8 + i * 4)[0] for i in range(rank)]
    header_end = 8 + rank * 4
    arr = np.frombuffer(data[header_end:], dtype=np.float32).copy()
    return arr.reshape(shape)


def assert_close(name: str, actual: np.ndarray, expected: np.ndarray) -> None:
    if actual.shape != expected.shape:
        raise AssertionError(f"{name}: shape mismatch {actual.shape} vs {expected.shape}")
    if not np.allclose(actual, expected, atol=ATOL, rtol=RTOL):
        diff = np.abs(actual - expected)
        idx  = np.unravel_index(np.argmax(diff), diff.shape)
        raise AssertionError(
            f"{name}: max abs diff {diff.max():.3e} at {idx} "
            f"(actual={actual[idx]:.6f}, expected={expected[idx]:.6f})"
        )


def run_runner(test_name: str, test_dir: Path, *, backend: str = "cpu", threads: int = 1) -> None:
    result = subprocess.run(
        [str(RUNNER_EXE), backend, test_name, str(test_dir), str(threads)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        sys.stderr.write(result.stdout)
        sys.stderr.write(result.stderr)
        raise RuntimeError(f"runner failed for {test_name}")


def setup_test(name: str) -> Path:
    test_dir = ARTIFACTS_DIR / name
    test_dir.mkdir(parents=True, exist_ok=True)
    return test_dir


def gen_binary_op(name: str, op_name: str, a_shape, b_shape, *, b_offset: float = 0.0):
    rng = np.random.default_rng(SEED)
    test_dir = setup_test(name)

    a_np = rng.standard_normal(a_shape).astype(np.float32)
    b_np = rng.standard_normal(b_shape).astype(np.float32) + b_offset

    write_tensor(test_dir / "input_a.bin", a_np)
    write_tensor(test_dir / "input_b.bin", b_np)

    a = torch.tensor(a_np, requires_grad=True)
    b = torch.tensor(b_np, requires_grad=True)

    a_view = a.view(-1, b.numel()) if a.numel() != b.numel() else a.view(b.shape)
    b_view = b.view(b.shape)

    if op_name == "add":
        out_view = a_view + b_view
    elif op_name == "sub":
        out_view = a_view - b_view
    elif op_name == "mul":
        out_view = a_view * b_view
    elif op_name == "div":
        out_view = a_view / b_view
    else:
        raise ValueError(op_name)

    out = out_view.reshape(a.shape)
    out.sum().backward()

    return test_dir, {
        "out":    out.detach().numpy(),
        "grad_a": a.grad.numpy(),
        "grad_b": b.grad.numpy(),
    }


def verify_binary_op(test_dir: Path, refs: dict) -> None:
    assert_close("out",    read_tensor(test_dir / "odin_out.bin"),    refs["out"])
    assert_close("grad_a", read_tensor(test_dir / "odin_grad_a.bin"), refs["grad_a"])
    assert_close("grad_b", read_tensor(test_dir / "odin_grad_b.bin"), refs["grad_b"])


def gen_linear(name: str, x_shape, w_shape):
    rng = np.random.default_rng(SEED)
    test_dir = setup_test(name)

    x_np = rng.standard_normal(x_shape).astype(np.float32) * 0.1
    w_np = rng.standard_normal(w_shape).astype(np.float32) * 0.1

    write_tensor(test_dir / "input_x.bin", x_np)
    write_tensor(test_dir / "input_w.bin", w_np)

    x = torch.tensor(x_np, requires_grad=True)
    w = torch.tensor(w_np, requires_grad=True)

    out = torch.nn.functional.linear(x, w)
    out.sum().backward()

    return test_dir, {
        "out":    out.detach().numpy(),
        "grad_x": x.grad.numpy(),
        "grad_w": w.grad.numpy(),
    }


def verify_linear(test_dir: Path, refs: dict) -> None:
    assert_close("out",    read_tensor(test_dir / "odin_out.bin"),    refs["out"])
    assert_close("grad_x", read_tensor(test_dir / "odin_grad_x.bin"), refs["grad_x"])
    assert_close("grad_w", read_tensor(test_dir / "odin_grad_w.bin"), refs["grad_w"])


def write_int_array(path: Path, values) -> None:
    arr = np.asarray(values, dtype=np.int32)
    with open(path, "wb") as f:
        f.write(struct.pack("<I", arr.size))
        f.write(arr.tobytes())


def gen_unary(name: str, op_name: str, x_shape):
    rng = np.random.default_rng(SEED)
    test_dir = setup_test(name)

    x_np = rng.standard_normal(x_shape).astype(np.float32)
    write_tensor(test_dir / "input_x.bin", x_np)

    x = torch.tensor(x_np, requires_grad=True)
    if op_name == "mean":
        out = x.mean(dim=-1)
    elif op_name == "softmax":
        out = torch.nn.functional.softmax(x, dim=-1)
    elif op_name == "log_softmax":
        out = torch.nn.functional.log_softmax(x, dim=-1)
    else:
        raise ValueError(op_name)

    out.sum().backward()
    return test_dir, {"out": out.detach().numpy(), "grad_x": x.grad.numpy()}


def verify_unary(test_dir: Path, refs: dict) -> None:
    assert_close("out",    read_tensor(test_dir / "odin_out.bin"),    refs["out"])
    assert_close("grad_x", read_tensor(test_dir / "odin_grad_x.bin"), refs["grad_x"])


def gen_layernorm(name: str, x_shape):
    rng = np.random.default_rng(SEED)
    test_dir = setup_test(name)

    feature_size = x_shape[-1]
    x_np = rng.standard_normal(x_shape).astype(np.float32)
    w_np = (rng.standard_normal((feature_size,)).astype(np.float32) * 0.1 + 1.0)

    write_tensor(test_dir / "input_x.bin", x_np)
    write_tensor(test_dir / "input_w.bin", w_np)

    x = torch.tensor(x_np, requires_grad=True)
    w = torch.tensor(w_np, requires_grad=True)
    out = torch.nn.functional.layer_norm(x, [feature_size], weight=w, bias=None, eps=1e-5)
    out.sum().backward()

    return test_dir, {
        "out":    out.detach().numpy(),
        "grad_x": x.grad.numpy(),
        "grad_w": w.grad.numpy(),
    }


def verify_layernorm(test_dir: Path, refs: dict) -> None:
    assert_close("out",    read_tensor(test_dir / "odin_out.bin"),    refs["out"])
    assert_close("grad_x", read_tensor(test_dir / "odin_grad_x.bin"), refs["grad_x"])
    assert_close("grad_w", read_tensor(test_dir / "odin_grad_w.bin"), refs["grad_w"])


def gen_cross_entropy(name: str, sample_count: int, class_size: int):
    rng = np.random.default_rng(SEED)
    test_dir = setup_test(name)

    x_np       = rng.standard_normal((sample_count, class_size)).astype(np.float32)
    targets_np = rng.integers(0, class_size, size=sample_count).astype(np.int64)

    write_tensor(test_dir / "input_x.bin", x_np)
    write_int_array(test_dir / "targets.bin", targets_np)

    x = torch.tensor(x_np, requires_grad=True)
    t = torch.tensor(targets_np, dtype=torch.long)
    out = torch.nn.functional.cross_entropy(x, t, reduction="none")
    out.sum().backward()

    return test_dir, {"out": out.detach().numpy(), "grad_x": x.grad.numpy()}


def verify_cross_entropy(test_dir: Path, refs: dict) -> None:
    assert_close("out",    read_tensor(test_dir / "odin_out.bin"),    refs["out"])
    assert_close("grad_x", read_tensor(test_dir / "odin_grad_x.bin"), refs["grad_x"])


def gen_batched_matmul(name: str, batch: int, m: int, k: int, n: int):
    rng = np.random.default_rng(SEED)
    test_dir = setup_test(name)

    a_np = rng.standard_normal((batch, m, k)).astype(np.float32) * 0.1
    b_np = rng.standard_normal((batch, k, n)).astype(np.float32) * 0.1

    write_tensor(test_dir / "input_a.bin", a_np)
    write_tensor(test_dir / "input_b.bin", b_np)

    a = torch.tensor(a_np, requires_grad=True)
    b = torch.tensor(b_np, requires_grad=True)
    out = torch.bmm(a, b)
    out.sum().backward()

    return test_dir, {
        "out":    out.detach().numpy(),
        "grad_a": a.grad.numpy(),
        "grad_b": b.grad.numpy(),
    }


def gen_permute(name: str, x_shape, axes):
    rng = np.random.default_rng(SEED)
    test_dir = setup_test(name)

    x_np = rng.standard_normal(x_shape).astype(np.float32)
    write_tensor(test_dir / "input_x.bin", x_np)
    write_int_array(test_dir / "axes.bin", axes)

    x = torch.tensor(x_np, requires_grad=True)
    out = x.permute(*axes).contiguous()
    out.sum().backward()

    return test_dir, {"out": out.detach().numpy(), "grad_x": x.grad.numpy()}


class TorchMlp(torch.nn.Module):
    def __init__(self, sizes):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [torch.nn.Linear(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)]
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = torch.relu(x)
        return x


def gen_mlp_train(name: str, sizes, sample_count: int, step_count: int, learning_rate: float):
    rng = np.random.default_rng(SEED)
    test_dir = setup_test(name)

    input_size  = sizes[0]
    output_size = sizes[-1]

    x_np = rng.standard_normal((sample_count, input_size)).astype(np.float32)
    y_np = rng.standard_normal((sample_count, output_size)).astype(np.float32)

    write_tensor(test_dir / "input_x.bin", x_np)
    write_tensor(test_dir / "input_y.bin", y_np)
    write_int_array(test_dir / "config.bin", [step_count, *sizes])

    torch.manual_seed(SEED)
    model = TorchMlp(sizes)

    for layer_index, layer in enumerate(model.layers):
        write_tensor(test_dir / f"init_w_{layer_index}.bin", layer.weight.detach().numpy())
        write_tensor(test_dir / f"init_b_{layer_index}.bin", layer.bias.detach().numpy())

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
    )

    x = torch.tensor(x_np)
    y = torch.tensor(y_np)

    losses = np.zeros(step_count, dtype=np.float32)
    for step in range(step_count):
        optimizer.zero_grad()
        pred = model(x)
        loss = ((pred - y) ** 2).mean()
        losses[step] = loss.item()
        loss.backward()
        optimizer.step()

    return test_dir, {"losses": losses}


def verify_mlp_train(test_dir: Path, refs: dict) -> None:
    odin_losses = read_tensor(test_dir / "odin_losses.bin")
    torch_losses = refs["losses"]
    if odin_losses.shape != torch_losses.shape:
        raise AssertionError(f"loss shape mismatch {odin_losses.shape} vs {torch_losses.shape}")
    diff = np.abs(odin_losses - torch_losses)
    rel  = diff / np.maximum(np.abs(torch_losses), 1e-8)
    if diff.max() > 1e-3 and rel.max() > 5e-3:
        worst = int(np.argmax(diff))
        raise AssertionError(
            f"loss curve diverged: max abs {diff.max():.3e} at step {worst} "
            f"(odin={odin_losses[worst]:.6f}, torch={torch_losses[worst]:.6f})"
        )


def gen_attention(name: str, tokens: int, embed: int, head_count: int, causal: bool):
    rng = np.random.default_rng(SEED)
    test_dir = setup_test(name)

    x_np = rng.standard_normal((tokens, 3 * embed)).astype(np.float32) * 0.1
    write_tensor(test_dir / "input_x.bin", x_np)
    write_int_array(test_dir / "config.bin", [head_count, 1 if causal else 0])

    head_size = embed // head_count
    x = torch.tensor(x_np, requires_grad=True)

    q_flat = x[:, 0:embed]
    k_flat = x[:, embed:2 * embed]
    v_flat = x[:, 2 * embed:3 * embed]

    q = q_flat.reshape(tokens, head_count, head_size).permute(1, 0, 2)
    k = k_flat.reshape(tokens, head_count, head_size).permute(1, 0, 2)
    v = v_flat.reshape(tokens, head_count, head_size).permute(1, 0, 2)

    scores = torch.matmul(q, k.transpose(1, 2)) / (head_size ** 0.5)
    if causal:
        mask = torch.triu(torch.ones(tokens, tokens, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
    attn = torch.nn.functional.softmax(scores, dim=-1)
    out_per_head = torch.matmul(attn, v)
    out = out_per_head.permute(1, 0, 2).reshape(tokens, embed)

    out.sum().backward()
    return test_dir, {"out": out.detach().numpy(), "grad_x": x.grad.numpy()}


TESTS = [
    ("add_equal",     lambda: gen_binary_op("add_equal",     "add", (4, 8), (4, 8)),    verify_binary_op),
    ("add_broadcast", lambda: gen_binary_op("add_broadcast", "add", (4, 8), (8,)),      verify_binary_op),
    ("sub_broadcast", lambda: gen_binary_op("sub_broadcast", "sub", (4, 8), (8,)),      verify_binary_op),
    ("mul_broadcast", lambda: gen_binary_op("mul_broadcast", "mul", (4, 8), (8,)),      verify_binary_op),
    ("div_broadcast", lambda: gen_binary_op("div_broadcast", "div", (4, 8), (8,), b_offset=2.0), verify_binary_op),
    ("linear_1d",     lambda: gen_linear("linear_1d", (16,),    (32, 16)), verify_linear),
    ("linear_2d",     lambda: gen_linear("linear_2d", (8, 16),  (32, 16)), verify_linear),
    ("mean",          lambda: gen_unary("mean",        "mean",        (4, 8)),  verify_unary),
    ("softmax",       lambda: gen_unary("softmax",     "softmax",     (4, 8)),  verify_unary),
    ("log_softmax",   lambda: gen_unary("log_softmax", "log_softmax", (4, 8)),  verify_unary),
    ("layernorm",     lambda: gen_layernorm("layernorm", (4, 16)), verify_layernorm),
    ("cross_entropy", lambda: gen_cross_entropy("cross_entropy", sample_count=8, class_size=10), verify_cross_entropy),
    ("batched_matmul", lambda: gen_batched_matmul("batched_matmul", batch=4, m=6, k=5, n=7), verify_binary_op),
    ("permute",       lambda: gen_permute("permute", (3, 4, 5), [1, 0, 2]), verify_unary),
    ("attention_causal",   lambda: gen_attention("attention_causal",   tokens=8, embed=16, head_count=2, causal=True),  verify_unary),
    ("attention_acausal",  lambda: gen_attention("attention_acausal",  tokens=8, embed=16, head_count=2, causal=False), verify_unary),
    ("mlp_train", lambda: gen_mlp_train("mlp_train", sizes=[4, 8, 8, 1], sample_count=16, step_count=50, learning_rate=0.01), verify_mlp_train),
]


def main() -> int:
    if not RUNNER_EXE.exists():
        print(f"runner not built: {RUNNER_EXE}", file=sys.stderr)
        print("build with: odin build tests/pytorch_parity/runner -o:speed -out:tests/pytorch_parity/runner.exe", file=sys.stderr)
        return 1

    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--backends", nargs="+", default=["cpu"], choices=["cpu", "gpu"])
    args = parser.parse_args()

    fails = 0
    total = 0
    for backend in args.backends:
        thread_counts = args.threads if backend == "cpu" else [1]
        for thread_count in thread_counts:
            label = f"backend={backend}" if backend == "gpu" else f"backend={backend} threads={thread_count}"
            print(f"--- {label} ---")
            for name, gen, verify in TESTS:
                total += 1
                try:
                    test_dir, refs = gen()
                    run_runner(name, test_dir, backend=backend, threads=thread_count)
                    verify(test_dir, refs)
                    print(f"  PASS  {name}")
                except Exception as e:
                    print(f"  FAIL  {name}: {e}")
                    fails += 1
            print()

    print(f"{total - fails}/{total} passed")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
