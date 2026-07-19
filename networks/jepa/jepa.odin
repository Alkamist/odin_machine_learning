package jepa

import "base:builtin"
import "base:runtime"

import "core:fmt"
import "core:log"
import "core:math"
import "core:math/rand"
import "core:slice"

import ml "../../"
import    "../../networks/mlp"

Loss_Kind :: enum {
	Mean_Squared_Error,
	Smooth_L1,
}

Config :: struct {
	state_size:  int,
	action_size: int,
	hidden_size: int,
	latent_size: int,

	loss:              Loss_Kind,
	normalize_targets: bool,
	rollout_steps:     int,
	variance_weight:   f32,
	variance_target:   f32,
}

DEFAULT_CONFIG :: Config{
	hidden_size       = 64,
	latent_size       = 16,
	loss              = .Mean_Squared_Error,
	normalize_targets = true,
	rollout_steps     = 1,
	variance_weight   = 0,
	variance_target   = 1,
}

Jepa :: struct {
	config: Config,

	encoder:        mlp.Mlp,
	target_encoder: mlp.Mlp,
	predictor:      mlp.Mlp,

	decoder: mlp.Mlp,
}

make :: proc(config: Config, allocator := context.allocator, loc := #caller_location) -> (jepa: Jepa) {
	assert(config.state_size > 0, "jepa config requires state_size > 0", loc=loc)
	assert(config.action_size > 0, "jepa config requires action_size > 0", loc=loc)
	assert(config.hidden_size > 0, "jepa config requires hidden_size > 0", loc=loc)
	assert(config.latent_size > 0, "jepa config requires latent_size > 0", loc=loc)
	assert(config.rollout_steps >= 1, "jepa config requires rollout_steps >= 1", loc=loc)

	jepa.config = config

	jepa.encoder        = mlp.make(config.state_size, config.hidden_size, config.latent_size, allocator=allocator)
	jepa.target_encoder = mlp.make(config.state_size, config.hidden_size, config.latent_size, flags=ml.Parameter_Flags{.Checkpoint}, allocator=allocator)
	jepa.predictor      = mlp.make(config.latent_size + config.action_size, config.hidden_size, config.latent_size, allocator=allocator)
	jepa.decoder        = mlp.make(config.latent_size, config.hidden_size, config.state_size, allocator=allocator)

	mlp.copy(jepa.target_encoder, jepa.encoder)

	return
}

destroy :: proc(jepa: Jepa) {
	mlp.destroy(jepa.encoder)
	mlp.destroy(jepa.target_encoder)
	mlp.destroy(jepa.predictor)
	mlp.destroy(jepa.decoder)
}

@(require_results)
encode :: proc(jepa: Jepa, states: ml.Tensor) -> ml.Tensor {
	return mlp.forward(jepa.encoder, states)
}

@(require_results)
encode_target :: proc(jepa: Jepa, states: ml.Tensor) -> ml.Tensor {
	return mlp.forward(jepa.target_encoder, states)
}

@(require_results)
predict :: proc(jepa: Jepa, latents, actions: ml.Tensor) -> ml.Tensor {
	return mlp.forward(jepa.predictor, ml.concat(latents, actions))
}

@(require_results)
decode :: proc(jepa: Jepa, latents: ml.Tensor) -> ml.Tensor {
	return mlp.forward(jepa.decoder, latents)
}

update_decoder :: proc(opt: ^ml.Optimizer, jepa: Jepa) {
	mlp.update(opt, jepa.decoder)
}

train_decoder_step :: proc(jepa: Jepa, states: []f32, count: int, loc := #caller_location) -> (loss: f32) {
	runtime.DEFAULT_TEMP_ALLOCATOR_TEMP_GUARD()

	c := jepa.config
	assert(len(states) == count * c.state_size, "states length mismatch", loc=loc)

	ml.clear()
	target_latents := encode_target(jepa, ml.tensor(states, []int{count, c.state_size}))
	latent_data    := builtin.make([]f32, count * c.latent_size, allocator=context.temp_allocator)
	ml.get_data(target_latents, latent_data)
	if c.normalize_targets {
		_normalize_rows(latent_data, count, c.latent_size)
	}

	ml.clear(training=true)
	decoded  := decode(jepa, ml.tensor(latent_data, []int{count, c.latent_size}))
	loss_t   := ml.mean(ml.mean_squared_error(decoded, ml.tensor(states)))
	ml.backward(loss_t, loc=loc)

	value: [1]f32
	ml.get_data(loss_t, value[:])
	return value[0]
}

ema_update :: proc(jepa: Jepa, momentum: f32) {
	alpha := 1 - momentum
	for i in 0 ..< len(jepa.encoder.layers) {
		ml.lerp_assign(jepa.target_encoder.layers[i].weight, jepa.encoder.layers[i].weight, alpha)
		ml.lerp_assign(jepa.target_encoder.layers[i].bias,   jepa.encoder.layers[i].bias,   alpha)
	}
}

update :: proc(opt: ^ml.Optimizer, jepa: Jepa) {
	mlp.update(opt, jepa.encoder)
	mlp.update(opt, jepa.predictor)
}

Planner_Config :: struct {
	sequence_count: int,
	horizon:        int,
	iterations:     int,
	elite_count:    int,
	action_repeat:  int,
}

DEFAULT_PLANNER_CONFIG :: Planner_Config{
	sequence_count = 64,
	horizon        = 30,
	iterations     = 3,
	elite_count    = 8,
	action_repeat  = 2,
}

_Ranked :: struct {
	cost:  f32,
	index: int,
}

@(require_results)
plan :: proc(jepa: Jepa, planner: Planner_Config, state: []f32, reward: proc(state: []f32) -> f32, loc := #caller_location) -> (action: int) {
	runtime.DEFAULT_TEMP_ALLOCATOR_TEMP_GUARD()

	c := jepa.config
	n := planner.sequence_count
	h := planner.horizon
	a := c.action_size
	l := c.latent_size

	assert(len(state) == c.state_size, "state length mismatch", loc=loc)
	assert(reward != nil, "requires a reward proc", loc=loc)
	assert(n > 0 && h > 0 && planner.iterations > 0 && planner.action_repeat > 0, "planner config values must be positive", loc=loc)
	assert(planner.elite_count > 0 && planner.elite_count <= n, "elite_count must be in [1, sequence_count]", loc=loc)

	ml.clear()
	start := encode(jepa, ml.tensor(state, []int{1, c.state_size}))
	start_data := builtin.make([]f32, l, allocator=context.temp_allocator)
	ml.get_data(start, start_data)

	tiled_start := builtin.make([]f32, n * l, allocator=context.temp_allocator)
	for i in 0 ..< n {
		builtin.copy(tiled_start[i * l:][:l], start_data)
	}

	probabilities := builtin.make([]f32, h * a, allocator=context.temp_allocator)
	for &p in probabilities {
		p = 1.0 / f32(a)
	}

	sampled    := builtin.make([]u8,      n * h,            allocator=context.temp_allocator)
	costs      := builtin.make([]f32,     n,                allocator=context.temp_allocator)
	one_hot    := builtin.make([]f32,     n * a,            allocator=context.temp_allocator)
	state_data := builtin.make([]f32,     n * c.state_size, allocator=context.temp_allocator)
	ranked     := builtin.make([]_Ranked, n,                allocator=context.temp_allocator)

	best_cost := math.INF_F32

	for _ in 0 ..< planner.iterations {
		for i in 0 ..< n {
			for t in 0 ..< h {
				sampled[i * h + t] = u8(_sample_categorical(probabilities[t * a:][:a]))
			}
		}

		for &cost in costs {
			cost = 0
		}

		ml.clear()
		z := ml.tensor(tiled_start, []int{n, l})

		for t in 0 ..< h {
			for &v in one_hot {
				v = 0
			}
			for i in 0 ..< n {
				one_hot[i * a + int(sampled[i * h + t])] = 1
			}
			actions := ml.tensor(one_hot, []int{n, a})

			for _ in 0 ..< planner.action_repeat {
				z = predict(jepa, z, actions)
				decoded := decode(jepa, z)
				ml.get_data(decoded, state_data)
				for i in 0 ..< n {
					costs[i] -= reward(state_data[i * c.state_size:][:c.state_size])
				}
			}
		}

		for i in 0 ..< n {
			ranked[i] = {cost=costs[i], index=i}
		}
		slice.sort_by(ranked, proc(x, y: _Ranked) -> bool {
			return x.cost < y.cost
		})

		if ranked[0].cost < best_cost {
			best_cost = ranked[0].cost
			action    = int(sampled[ranked[0].index * h])
		}

		for t in 0 ..< h {
			row := probabilities[t * a:][:a]
			for &p in row {
				p *= 0.5
			}
			for e in 0 ..< planner.elite_count {
				row[int(sampled[ranked[e].index * h + t])] += 0.5 / f32(planner.elite_count)
			}
		}
	}

	return
}

_sample_categorical :: proc(probabilities: []f32) -> int {
	r := rand.float32()
	cumulative: f32
	for p, i in probabilities {
		cumulative += p
		if r < cumulative {
			return i
		}
	}
	return builtin.len(probabilities) - 1
}

parameters :: proc(jepa: Jepa, dst: ^ml.Registry) {
	mlp.parameters(jepa.encoder,        dst, prefix="encoder")
	mlp.parameters(jepa.target_encoder, dst, prefix="target_encoder")
	mlp.parameters(jepa.predictor,      dst, prefix="predictor")
}

decoder_parameters :: proc(jepa: Jepa, dst: ^ml.Registry) {
	mlp.parameters(jepa.decoder, dst, prefix="decoder")
}

save :: proc(jepa: Jepa, path: string, opt: ^ml.Optimizer = nil, iteration: u64 = 0, decoder_iteration: u64 = 0, loc := #caller_location) -> bool {
	runtime.DEFAULT_TEMP_ALLOCATOR_TEMP_GUARD()

	gathered: ml.Registry
	gathered.parameters.allocator = context.temp_allocator
	parameters(jepa, &gathered)
	decoder_parameters(jepa, &gathered)

	metadata := builtin.make(map[string]string, allocator=context.temp_allocator)
	metadata["ml.iteration"]         = fmt.tprintf("%d", iteration)
	metadata["ml.iteration.decoder"] = fmt.tprintf("%d", decoder_iteration)
	metadata["jepa.state_size"]      = fmt.tprintf("%d", jepa.config.state_size)
	metadata["jepa.action_size"]     = fmt.tprintf("%d", jepa.config.action_size)
	metadata["jepa.hidden_size"]     = fmt.tprintf("%d", jepa.config.hidden_size)
	metadata["jepa.latent_size"]     = fmt.tprintf("%d", jepa.config.latent_size)

	return ml.checkpoint_save(path, &gathered, opt, metadata, loc=loc)
}

@(require_results)
load :: proc(config: Config, path: string, opt: ^ml.Optimizer = nil, allocator := context.allocator, loc := #caller_location) -> (jepa: Jepa, iteration: u64, decoder_iteration: u64, ok: bool) {
	runtime.DEFAULT_TEMP_ALLOCATOR_TEMP_GUARD()

	jepa = make(config, allocator=allocator, loc=loc)

	gathered: ml.Registry
	gathered.parameters.allocator = context.temp_allocator
	parameters(jepa, &gathered)
	decoder_parameters(jepa, &gathered)

	metadata, load_ok := ml.checkpoint_load(path, &gathered, opt, loc=loc)
	if !load_ok {
		destroy(jepa)
		return {}, 0, 0, false
	}
	defer ml.checkpoint_metadata_destroy(metadata)

	if !_metadata_dim_matches(metadata, "jepa.state_size",  config.state_size)  ||
	   !_metadata_dim_matches(metadata, "jepa.action_size", config.action_size) ||
	   !_metadata_dim_matches(metadata, "jepa.hidden_size", config.hidden_size) ||
	   !_metadata_dim_matches(metadata, "jepa.latent_size", config.latent_size) {
		log.errorf("config dims in %v do not match the requested config", path, location=loc)
		destroy(jepa)
		return {}, 0, 0, false
	}

	iteration         = ml.checkpoint_metadata_u64(metadata, "ml.iteration")
	decoder_iteration = ml.checkpoint_metadata_u64(metadata, "ml.iteration.decoder")
	return jepa, iteration, decoder_iteration, true
}

_metadata_dim_matches :: proc(metadata: map[string]string, key: string, expected: int) -> bool {
	return int(ml.checkpoint_metadata_u64(metadata, key, u64(expected))) == expected
}

Batch :: struct {
	batch_size:  int,
	states:      []f32,
	actions:     []f32,
	next_states: []f32,
}

Step_Metrics :: struct {
	loss:          f32,
	variance_loss: f32,
	latents:       Latent_Stats,
}

train_step :: proc(jepa: Jepa, batch: Batch, loc := #caller_location) -> (metrics: Step_Metrics) {
	runtime.DEFAULT_TEMP_ALLOCATOR_TEMP_GUARD()

	c := jepa.config
	b := batch.batch_size
	k := c.rollout_steps

	assert(b >= 2, "train_step requires batch_size >= 2", loc=loc)
	assert(len(batch.states) == b * c.state_size, "states length mismatch", loc=loc)
	assert(len(batch.actions) == k * b * c.action_size, "actions length mismatch", loc=loc)
	assert(len(batch.next_states) == k * b * c.state_size, "next_states length mismatch", loc=loc)

	ml.clear()
	target_latents := encode_target(jepa, ml.tensor(batch.next_states, []int{k * b, c.state_size}))
	target_data    := builtin.make([]f32, k * b * c.latent_size, allocator=context.temp_allocator)
	ml.get_data(target_latents, target_data)
	if c.normalize_targets {
		_normalize_rows(target_data, k * b, c.latent_size)
	}

	ml.clear(training=true)

	latents := encode(jepa, ml.tensor(batch.states, []int{b, c.state_size}))

	first_latents := latents
	prediction_loss: ml.Tensor
	for t in 0 ..< k {
		actions := ml.tensor(batch.actions[t * b * c.action_size:][:b * c.action_size], []int{b, c.action_size})
		latents  = predict(jepa, latents, actions)

		targets := ml.tensor(target_data[t * b * c.latent_size:][:b * c.latent_size])

		step_loss: ml.Tensor
		switch c.loss {
		case .Mean_Squared_Error:
			step_loss = ml.mean_squared_error(latents, targets)
		case .Smooth_L1:
			step_loss = ml.smooth_l1(latents, targets)
		}
		step_loss = ml.mean(step_loss)

		prediction_loss = step_loss if t == 0 else ml.add(prediction_loss, step_loss)
	}
	if k > 1 {
		prediction_loss = ml.mul(prediction_loss, ml.scalar(.F32, 1.0 / f32(k)))
	}

	total_loss := prediction_loss

	variance_loss: ml.Tensor
	if c.variance_weight > 0 {
		variance_loss = _variance_hinge(first_latents, c.variance_target)
		total_loss    = ml.add(total_loss, ml.mul(variance_loss, ml.scalar(.F32, c.variance_weight)))
	}

	ml.backward(total_loss, loc=loc)

	loss_value: [1]f32
	ml.get_data(prediction_loss, loss_value[:])
	metrics.loss = loss_value[0]

	if c.variance_weight > 0 {
		ml.get_data(variance_loss, loss_value[:])
		metrics.variance_loss = loss_value[0]
	}

	latent_data := builtin.make([]f32, b * c.latent_size, allocator=context.temp_allocator)
	ml.get_data(first_latents, latent_data)
	metrics.latents = latent_stats(latent_data, b, c.latent_size)

	return
}

@(require_results)
_variance_hinge :: proc(latents: ml.Tensor, target_std: f32) -> ml.Tensor {
	means    := ml.mean(ml.transpose(latents))
	centered := ml.sub(latents, means)
	variance := ml.mean(ml.transpose(ml.mul(centered, centered)))
	std      := ml.sqrt(ml.add(variance, ml.scalar(.F32, 1e-8)))
	deficit  := ml.relu(ml.mul(ml.sub(std, ml.scalar(.F32, target_std)), ml.scalar(.F32, -1)))
	return ml.mean(deficit)
}

_normalize_rows :: proc(values: []f32, count, size: int) {
	for i in 0 ..< count {
		row := values[i * size:][:size]

		mean: f32
		for v in row {
			mean += v
		}
		mean /= f32(size)

		variance: f32
		for v in row {
			diff := v - mean
			variance += diff * diff
		}
		variance /= f32(size)

		inverse_std := 1.0 / math.sqrt(variance + 1e-6)
		for &v in row {
			v = (v - mean) * inverse_std
		}
	}
}

Latent_Stats :: struct {
	mean_std:    f32,
	min_std:     f32,
	mean_cosine: f32,
}

@(require_results)
latent_stats :: proc(latents: []f32, batch_size, latent_size: int, loc := #caller_location) -> (stats: Latent_Stats) {
	assert(len(latents) == batch_size * latent_size, "latents length must be batch_size * latent_size", loc=loc)
	assert(batch_size >= 2, "batch_size must be at least 2", loc=loc)

	stats.min_std = math.INF_F32
	for d in 0 ..< latent_size {
		mean: f32
		for i in 0 ..< batch_size {
			mean += latents[i * latent_size + d]
		}
		mean /= f32(batch_size)

		variance: f32
		for i in 0 ..< batch_size {
			diff := latents[i * latent_size + d] - mean
			variance += diff * diff
		}
		variance /= f32(batch_size)

		std := math.sqrt(variance)
		stats.mean_std += std
		stats.min_std   = min(stats.min_std, std)
	}
	stats.mean_std /= f32(latent_size)

	pair_count := 0
	for i in 0 ..< batch_size - 1 {
		a := latents[i * latent_size:][:latent_size]
		b := latents[(i + 1) * latent_size:][:latent_size]

		dot, a_sq, b_sq: f32
		for d in 0 ..< latent_size {
			dot  += a[d] * b[d]
			a_sq += a[d] * a[d]
			b_sq += b[d] * b[d]
		}

		denominator := math.sqrt(a_sq * b_sq)
		if denominator > 1e-12 {
			stats.mean_cosine += abs(dot) / denominator
			pair_count        += 1
		}
	}
	if pair_count > 0 {
		stats.mean_cosine /= f32(pair_count)
	}

	return
}
