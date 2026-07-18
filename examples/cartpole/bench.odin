package main

import "core:fmt"
import "core:time"

import ml "../../"
import    "../../networks/mlp"

// Breaks a training step and an inference step into their phases so the cost
// can be attributed to graph construction, arithmetic, or the optimizer rather
// than guessed at.
bench :: proc() {
	ITERATIONS :: 2000

	model := mlp.make(MODEL_INPUT, tuning.hidden, OBS_SIZE, allocator=context.allocator)
	defer mlp.destroy(model)

	opt: ml.Optimizer

	inputs  := make([]f32, tuning.train_batch * MODEL_INPUT)
	targets := make([]f32, tuning.train_batch * OBS_SIZE)
	defer delete(inputs)
	defer delete(targets)

	measure :: proc(name: string, iterations: int, body: proc()) {
		start := time.tick_now()
		for _ in 0 ..< iterations {
			body()
		}
		spent := time.tick_since(start)
		fmt.printfln("  %-28s %7.4f ms", name, time.duration_milliseconds(spent) / f64(iterations))
	}

	@(static) b_model:   mlp.Mlp
	@(static) b_opt:     ^ml.Optimizer
	@(static) b_inputs:  []f32
	@(static) b_targets: []f32

	b_model   = model
	b_opt     = &opt
	b_inputs  = inputs
	b_targets = targets

	fmt.printfln("training step, batch %d, hidden %d:", tuning.train_batch, tuning.hidden)

	measure("clear only", ITERATIONS, proc() {
		ml.clear(training=true)
	})

	measure("clear + input tensors", ITERATIONS, proc() {
		ml.clear(training=true)
		_ = ml.reshape(ml.tensor(b_inputs),  {tuning.train_batch, MODEL_INPUT})
		_ = ml.reshape(ml.tensor(b_targets), {tuning.train_batch, OBS_SIZE})
	})

	measure("+ forward", ITERATIONS, proc() {
		ml.clear(training=true)
		x := ml.reshape(ml.tensor(b_inputs),  {tuning.train_batch, MODEL_INPUT})
		y := ml.reshape(ml.tensor(b_targets), {tuning.train_batch, OBS_SIZE})
		_ = ml.mean(ml.mean_squared_error(mlp.forward(b_model, x), y))
	})

	measure("+ backward", ITERATIONS, proc() {
		ml.clear(training=true)
		x := ml.reshape(ml.tensor(b_inputs),  {tuning.train_batch, MODEL_INPUT})
		y := ml.reshape(ml.tensor(b_targets), {tuning.train_batch, OBS_SIZE})
		ml.backward(ml.mean(ml.mean_squared_error(mlp.forward(b_model, x), y)))
	})

	measure("+ optimizer (full step)", ITERATIONS, proc() {
		ml.clear(training=true)
		x := ml.reshape(ml.tensor(b_inputs),  {tuning.train_batch, MODEL_INPUT})
		y := ml.reshape(ml.tensor(b_targets), {tuning.train_batch, OBS_SIZE})
		ml.backward(ml.mean(ml.mean_squared_error(mlp.forward(b_model, x), y)))
		if ml.optimizer_step(b_opt, period=1, learning_rate=tuning.learning_rate) {
			mlp.update(b_opt^, b_model)
		}
	})

	fmt.printfln("individual ops, batch %d:", tuning.train_batch)

	measure("linear 1 (in->hidden)", ITERATIONS, proc() {
		ml.clear()
		x := ml.reshape(ml.tensor(b_inputs), {tuning.train_batch, MODEL_INPUT})
		_ = ml.linear(x, b_model.layers[0].weight)
	})

	measure("linear 1 + bias", ITERATIONS, proc() {
		ml.clear()
		x := ml.reshape(ml.tensor(b_inputs), {tuning.train_batch, MODEL_INPUT})
		_ = ml.add(ml.linear(x, b_model.layers[0].weight), b_model.layers[0].bias)
	})

	measure("linear 1 + bias + relu", ITERATIONS, proc() {
		ml.clear()
		x := ml.reshape(ml.tensor(b_inputs), {tuning.train_batch, MODEL_INPUT})
		_ = ml.relu(ml.add(ml.linear(x, b_model.layers[0].weight), b_model.layers[0].bias))
	})

	fmt.printfln("inference step (one rollout step), batch %d:", tuning.plan_samples)

	plan_inputs := make([]f32, tuning.plan_samples * MODEL_INPUT)
	plan_out    := make([]f32, tuning.plan_samples * OBS_SIZE)
	defer delete(plan_inputs)
	defer delete(plan_out)

	@(static) b_plan_inputs: []f32
	@(static) b_plan_out:    []f32
	b_plan_inputs = plan_inputs
	b_plan_out    = plan_out

	measure("clear only", ITERATIONS, proc() {
		ml.clear()
	})

	measure("+ forward", ITERATIONS, proc() {
		ml.clear()
		_ = mlp.forward(b_model, ml.reshape(ml.tensor(b_plan_inputs), {tuning.plan_samples, MODEL_INPUT}))
	})

	measure("+ get_data", ITERATIONS, proc() {
		ml.clear()
		out := mlp.forward(b_model, ml.reshape(ml.tensor(b_plan_inputs), {tuning.plan_samples, MODEL_INPUT}))
		ml.get_data(out, b_plan_out)
	})
}
