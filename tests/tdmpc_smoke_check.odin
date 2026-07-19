package ml_tests

import "core:math"
import "core:testing"

import ml  "../"
import cpu "../backends/cpu"
import     "../networks/mlp"

@(test)
test_tdmpc_smoke :: proc(t: ^testing.T) {
	BATCH   :: 8
	SENSORS :: 5
	ACTIONS :: 3
	HIDDEN  :: 16

	ctx := cpu.context_create(16 * 1024 * 1024)
	defer cpu.context_destroy(ctx)
	ml.context_scope(ctx)

	policy     := mlp.make(SENSORS, HIDDEN, ACTIONS)
	value      := mlp.make(SENSORS, HIDDEN, ACTIONS)
	target     := mlp.make(SENSORS, HIDDEN, ACTIONS)
	policy_opt := ml.optimizer_make(learning_rate=1e-3)
	value_opt  := ml.optimizer_make(learning_rate=1e-3)

	defer {
		ml.optimizer_destroy(&policy_opt)
		ml.optimizer_destroy(&value_opt)
		mlp.destroy(policy)
		mlp.destroy(value)
		mlp.destroy(target)
	}

	mlp.copy(target, value)

	states  := make([]f32, BATCH * SENSORS)
	targets := make([]f32, BATCH)
	neg_q   := make([]f32, BATCH * ACTIONS)
	gather  := make([]int, BATCH)
	result  := make([]f32, BATCH * ACTIONS)
	defer {
		delete(states)
		delete(targets)
		delete(neg_q)
		delete(gather)
		delete(result)
	}

	for i in 0 ..< BATCH * SENSORS {
		states[i] = f32(i % 7) * 0.1
	}
	for b in 0 ..< BATCH {
		targets[b] = f32(b) * 0.25
		gather[b]  = b * ACTIONS + b % ACTIONS
	}
	for i in 0 ..< BATCH * ACTIONS {
		neg_q[i] = -f32(i % 5) * 0.2
	}

	ml.clear()
	{
		x := ml.tensor(states, []int{BATCH, SENSORS})
		ml.get_data(ml.softmax(mlp.forward(policy, x)), result)
		ml.get_data(mlp.forward(target, x), result)
	}

	ml.clear(training=true)
	{
		x          := ml.tensor(states,  []int{BATCH, SENSORS})
		y          := ml.tensor(targets, []int{BATCH})
		q_values   := mlp.forward(value, x)
		flat       := ml.reshape(q_values, []int{BATCH * ACTIONS})
		prediction := ml.select(flat, gather)
		loss       := ml.mean(ml.mean_squared_error(prediction, y))

		ml.backward(loss)

		if ml.optimizer_step(&value_opt) {
			mlp.update(&value_opt, value)
		}
	}

	for layer, layer_index in value.layers {
		ml.lerp_assign(target.layers[layer_index].weight, layer.weight, 0.01)
		ml.lerp_assign(target.layers[layer_index].bias,   layer.bias,   0.01)
	}

	ml.clear(training=true)
	{
		x             := ml.tensor(states, []int{BATCH, SENSORS})
		nq            := ml.tensor(neg_q,  []int{BATCH, ACTIONS})
		logits        := mlp.forward(policy, x)
		probabilities := ml.softmax(logits)
		value_term    := ml.mean(ml.sum(ml.mul(probabilities, nq)))
		entropy_term  := ml.mul(ml.mean(ml.entropy(probabilities)), ml.scalar(.F32, -1.0))
		loss          := ml.add(value_term, entropy_term)

		loss_value: [1]f32
		ml.backward(loss)
		ml.get_data(loss, loss_value[:])

		testing.expect(t, !math.is_nan(loss_value[0]), "policy loss is NaN")

		if ml.optimizer_step(&policy_opt) {
			mlp.update(&policy_opt, policy)
		}
	}
}
