// Benchmark harness for PPO on CartPole.
//
// Mirrors examples/ppo/main.odin but:
//   - Seeds rand for reproducibility.
//   - Selects a variant at runtime so we can compare correctness fixes.
//   - Emits CSV: seed,variant,step,wall_seconds,score
//
// Usage:
//   ppo_bench --seed N --variant {baseline,no_vclip,epochs4,both} --steps S [--solve-stop]

package main

import "core:os"
import "core:fmt"
import "core:math"
import "core:math/rand"
import "core:slice"
import "core:time"
import "core:strconv"
import "../utility"
import ml "../../"
import cpu "../../backends/cpu"
import "../../networks/mlp"

import game "../cartpole"

Variant :: enum {
	Baseline,
	No_Vclip,
	Epochs4,
	Both,
}

DEFAULT_STEPS :: 5000

TRAJECTORIES  :: 32
LEARNING_RATE :: 0.001
PERIOD        :: 128

HIDDEN_SIZE :: 128

EVALUATION_GAMES    :: 1
EVALUATION_INTERVAL :: 25

GAMMA              :: 0.99
LAMBDA             :: 0.95
CLIP_EPSILON       :: 0.2
VALUE_CLIP_EPSILON :: 0.5
ENTROPY            :: 0.01

g_seed:       u64
g_variant:    Variant = .Baseline
g_steps:      int     = DEFAULT_STEPS
g_solve_stop: bool

main :: proc() {
	parse_args()

	rand.reset(g_seed)

	ctx := cpu.context_create(1024 * 1024)
	defer cpu.context_destroy(ctx)
	ml.context_scope(ctx)

	train()
}

parse_args :: proc() {
	args := os.args
	i := 1
	for i < len(args) {
		arg := args[i]
		switch arg {
		case "--seed":
			i += 1
			val, ok := strconv.parse_u64(args[i])
			if !ok do panic("bad --seed")
			g_seed = val
		case "--variant":
			i += 1
			switch args[i] {
			case "baseline": g_variant = .Baseline
			case "no_vclip": g_variant = .No_Vclip
			case "epochs4":  g_variant = .Epochs4
			case "both":     g_variant = .Both
			case: panic("bad --variant")
			}
		case "--steps":
			i += 1
			val, ok := strconv.parse_int(args[i])
			if !ok do panic("bad --steps")
			g_steps = val
		case "--solve-stop":
			g_solve_stop = true
		case:
			fmt.eprintfln("unknown arg: %s", arg)
			os.exit(1)
		}
		i += 1
	}
}

variant_name :: proc(v: Variant) -> string {
	switch v {
	case .Baseline: return "baseline"
	case .No_Vclip: return "no_vclip"
	case .Epochs4:  return "epochs4"
	case .Both:     return "both"
	}
	return "?"
}

train :: proc() {
	model := model_make()
	defer model_destroy(&model)

	start := time.tick_now()

	for {
		defer free_all(context.temp_allocator)

		if model.step >= g_steps {
			break
		}

		improve(&model)

		if model.step % EVALUATION_INTERVAL == 0 {
			score    := evaluate(&model)
			elapsed  := time.duration_seconds(time.tick_diff(start, time.tick_now()))
			fmt.printfln("%d,%s,%d,%.3f,%.4f", g_seed, variant_name(g_variant), model.step, elapsed, score)

			if g_solve_stop && score >= game.SOLVE_SCORE {
				break
			}
		}
	}
}

Frame :: struct {
	embedding:         game.Embedding,
	action:            game.Action,
	log_probability:   f32,
	reward:            f32,
	discounted_return: f32,
	value:             f32,
	advantage:         f32,
}

Network :: struct {
	mlp: mlp.Mlp,
}

network_make :: proc(output_size: int, allocator := context.allocator) -> (network: Network) {
	network.mlp = mlp.make(len(game.Embedding), HIDDEN_SIZE, output_size, allocator=allocator)
	return
}

network_destroy :: proc(network: Network) {
	mlp.destroy(network.mlp)
}

network_forward :: proc(network: Network, input: []f32) -> ml.Tensor {
	return mlp.forward(network.mlp, ml.tensor(input))
}

network_update :: proc(opt: ml.Optimizer, network: Network) {
	mlp.update(opt, network.mlp)
}

Model :: struct {
	actor:  Network,
	critic: Network,

	opt: ml.Optimizer,

	step: int,

	game_state: game.State,
	frames:     [dynamic]Frame,
}

model_make :: proc(allocator := context.allocator) -> (model: Model) {
	model.actor  = network_make(len(game.Action), allocator=allocator)
	model.critic = network_make(1,                allocator=allocator)

	model.frames = make([dynamic]Frame, 0, 60 * 60 * TRAJECTORIES, allocator=allocator)

	game.init(&model.game_state)

	return
}

model_destroy :: proc(model: ^Model) {
	network_destroy(model.actor)
	network_destroy(model.critic)

	delete(model.frames)

	game.destroy(&model.game_state)
}

choose_action :: proc(network: Network, embedding: game.Embedding, sample := false) -> (action: game.Action, log_probability: f32) {
	ml.clear()

	embedding         := embedding
	logits            := network_forward(network, embedding[:])
	probabilities     := ml.softmax(logits)
	log_probabilities := ml.log_softmax(logits)

	probabilities_data     := make([]f32, ml.len(probabilities),     allocator=context.temp_allocator)
	log_probabilities_data := make([]f32, ml.len(log_probabilities), allocator=context.temp_allocator)
	ml.get_data(probabilities,     probabilities_data)
	ml.get_data(log_probabilities, log_probabilities_data)

	if sample {
		action = game.Action(utility.sample_probability_distribution(probabilities_data))
	} else {
		action = game.Action(slice.max_index(probabilities_data))
	}

	log_probability = log_probabilities_data[action]

	return
}

predict_value :: proc(network: Network, embedding: game.Embedding) -> f32 {
	embedding := embedding
	output    := network_forward(network, embedding[:])

	value: [1]f32
	ml.get_data(output, value[:])
	return value[0]
}

record_trajectory :: proc(model: ^Model) {
	game.reset(&model.game_state)

	start_index := len(model.frames)

	bootstrap:     f32
	truncated_end: bool

	for {
		frame: Frame

		frame.embedding = game.embedding(model.game_state)

		ml.clear()
		frame.value = predict_value(model.critic, frame.embedding)

		frame.action, frame.log_probability = choose_action(model.actor, frame.embedding, sample=true)

		done, truncated: bool
		frame.reward, done, truncated = game.step(&model.game_state, frame.action, game.FIXED_DELTA)

		append(&model.frames, frame)

		if done {
			if truncated {
				ml.clear()
				bootstrap     = predict_value(model.critic, game.embedding(model.game_state))
				truncated_end = true
			}
			break
		}
	}

	gae: f32
	for i := len(model.frames) - 1; i >= start_index; i -= 1 {
		value: f32 = model.frames[i].value

		next_value: f32
		if i + 1 < len(model.frames) {
			next_value = model.frames[i + 1].value
		} else if truncated_end {
			next_value = bootstrap
		}

		delta := model.frames[i].reward + GAMMA * next_value - value
		gae    = delta + GAMMA * LAMBDA * gae

		model.frames[i].advantage         = gae
		model.frames[i].discounted_return = value + gae
	}
}

normalize_advantages :: proc(model: Model) {
	mean:  f32
	count: int
	for frame in model.frames {
		mean  += frame.advantage
		count += 1
	}
	mean /= f32(count)

	sum: f32
	for frame in model.frames {
		diff := frame.advantage - mean
		sum  += diff * diff
	}
	std := math.sqrt(sum / f32(count))

	if std > 1e-8 {
		for &frame in model.frames {
			frame.advantage = (frame.advantage - mean) / std
		}
	}
}

play_game :: proc(model: ^Model) -> (score: f32) {
	game.reset(&model.game_state)

	for {
		action, _  := choose_action(model.actor, game.embedding(model.game_state))
		_, done, _ := game.step(&model.game_state, action, game.FIXED_DELTA)
		if done {
			score = model.game_state.score
			break
		}
	}

	return
}

evaluate :: proc(model: ^Model) -> (score: f32) {
	for _ in 0 ..< EVALUATION_GAMES {
		score += play_game(model)
	}
	score /= f32(EVALUATION_GAMES)
	return
}

improve :: proc(model: ^Model) {
	clear(&model.frames)

	for _ in 0 ..< TRAJECTORIES {
		record_trajectory(model)
	}

	normalize_advantages(model^)

	lr := utility.linear_learning_rate(LEARNING_RATE, 0, model.step, g_steps)

	epochs := 1
	if g_variant == .Epochs4 || g_variant == .Both {
		epochs = 4
	}

	use_value_clip := g_variant == .Baseline || g_variant == .Epochs4

	for _ in 0 ..< epochs {
		rand.shuffle(model.frames[:])

		for &frame in model.frames {
			// Actor.
			ml.clear()

			logits            := network_forward(model.actor, frame.embedding[:])
			log_probabilities := ml.log_softmax(logits)
			log_probability   := ml.select(log_probabilities, {int(frame.action)})

			ratio         := ml.exp(ml.sub(log_probability, ml.scalar(.F32, frame.log_probability)))
			clipped_ratio := ml.clamp(ratio, 1.0 - CLIP_EPSILON, 1.0 + CLIP_EPSILON)

			advantage         := ml.scalar(.F32, frame.advantage)
			objective         := ml.mul(ratio,         advantage)
			clipped_objective := ml.mul(clipped_ratio, advantage)

			actor_loss := ml.mul(ml.min(objective, clipped_objective), ml.scalar(.F32, -1))

			when ENTROPY > 0 {
				probabilities := ml.softmax(logits)
				entropy       := ml.entropy(probabilities)
				entropy_loss  := ml.mul(entropy, ml.scalar(.F32, -ENTROPY))
				actor_loss     = ml.add(actor_loss, entropy_loss)
			}

			ml.backward()

			// Critic.
			ml.clear()

			value  := network_forward(model.critic, frame.embedding[:])
			target := ml.scalar(.F32, frame.discounted_return)

			critic_loss: ml.Tensor
			if use_value_clip {
				clipped_value := ml.clamp(value, frame.value - VALUE_CLIP_EPSILON, frame.value + VALUE_CLIP_EPSILON)

				unclipped_loss := ml.mean_squared_error(value,         target)
				clipped_loss   := ml.mean_squared_error(clipped_value, target)

				critic_loss = ml.max(unclipped_loss, clipped_loss)
			} else {
				critic_loss = ml.mean_squared_error(value, target)
			}
			_ = critic_loss

			ml.backward()

			if ml.optimize(&model.opt, period=PERIOD, learning_rate=lr) {
				network_update(model.opt, model.actor)
				network_update(model.opt, model.critic)
			}
		}
	}

	model.step += 1
}