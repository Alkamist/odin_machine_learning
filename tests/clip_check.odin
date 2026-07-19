package ml_tests

import "core:mem"
import "core:math"
import "core:testing"

import ml  "../"
import cpu "../backends/cpu"

_clip_case :: proc(t: ^testing.T, name: string, grads: [][]f32, max_norm: f32) {
	n := len(grads)
	tensors := make([]ml.Tensor, n)
	defer delete(tensors)
	r: ml.Registry

	for g, i in grads {
		shape := [1]int{len(g)}
		tensors[i] = ml.alloc(.F32, shape[:], persistent=true, buffers=ml.DEFAULT_PARAMETER_BUFFERS)
		ml.set_bytes(tensors[i], .Gradient, mem.slice_to_bytes(g))
		ml.parameter_register(&r, "", "", tensors[i], init=ml.Init_None{})
	}

	total_sq := f64(0)
	for g in grads {
		for x in g {
			total_sq += f64(x) * f64(x)
		}
	}
	ref_norm := f32(math.sqrt(total_sq))

	scale := f32(1)
	if ref_norm > max_norm && ref_norm != 0 {
		scale = max_norm / ref_norm
	}

	got_norm := ml.clip_gradient_norm(&r, max_norm)

	norm_denom := max(max(abs(f64(ref_norm)), abs(f64(got_norm))), 1e-4)
	norm_rel   := abs(f64(ref_norm) - f64(got_norm)) / norm_denom
	testing.expectf(t, norm_rel <= 1e-5, "%s: norm ref=%.7g got=%.7g rel_err=%.4g", name, ref_norm, got_norm, norm_rel)

	for g, i in grads {
		got := make([]f32, len(g), context.temp_allocator)
		ml.get_gradient(tensors[i], got)
		for j in 0 ..< len(g) {
			expected := g[j] * scale
			denom := max(max(abs(f64(expected)), abs(f64(got[j]))), 1e-4)
			rel   := abs(f64(expected) - f64(got[j])) / denom
			testing.expectf(t, rel <= 1e-5, "%s: grad param %d elem %d expected=%.7g got=%.7g rel_err=%.4g", name, i, j, expected, got[j], rel)
		}
	}

	ml.registry_destroy(&r)
}

@(test)
test_clip_gradient_norm :: proc(t: ^testing.T) {
	ctx := cpu.context_create(1 * 1024 * 1024)
	ml.context_begin(ctx)

	clip_grads := [][]f32{{3, 4}, {0, 12}}
	_clip_case(t, "clip_triggered", clip_grads, 5.0)

	noop_grads := [][]f32{{0.1, -0.2}, {0.15}}
	_clip_case(t, "under_threshold", noop_grads, 1.0)

	zero_grads := [][]f32{{0, 0, 0}, {0, 0}}
	_clip_case(t, "zero_norm", zero_grads, 1.0)

	ml.context_end()
	cpu.context_destroy(ctx)
}
