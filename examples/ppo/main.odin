// In this example, Proximal Policy Optimization is used to
// train a Multilayer Perceptron to play either CartPole or
// Circles, based on the import.
//
// Comment or uncomment train or play to train a model and
// save it to a file, or load a model from a file and play
// with it.

package main

import "core:os"
import "core:fmt"
import "core:math"
import "core:math/rand"
import "core:slice"
import "core:encoding/json"
import "../utility"
import ml "../../"
import cpu "../../backends/cpu"
import "../../networks/mlp"

import game "../cartpole"

MODEL_FILE :: "model.json"

STEPS         :: 5000  // How many steps until training is done, also decays learning rate.
EPOCHS        :: 1     // Pass over the frames of each step this many times. Typically 1-10.
TRAJECTORIES  :: 32    // Learn from this many games each PPO step.
LEARNING_RATE :: 0.001 // How fast the model should learn.
PERIOD        :: 128   // How many frames to accumulate gradients for.

HIDDEN_SIZE :: 128 // How powerful the multilayer perceptron is.

EVALUATION_GAMES    :: 1   // How many games to evaluate the model on.
EVALUATION_INTERVAL :: 100 // How often to evaluate the model.

GAMMA        :: 0.99 // How much future rewards are worth relative to immediate ones.
LAMBDA       :: 0.95 // How much future rewards contribute to advantage estimates.
CLIP_EPSILON :: 0.2  // How much is the model allowed to change each update.
ENTROPY      :: 0.01 // How much is exploration encouraged.

main :: proc() {
	defer fmt.println("Finished")

	ctx := cpu.context_create(1024 * 1024)
	defer cpu.context_destroy(ctx)
	ml.context_scope(ctx)

	train()
	// play()
}

train :: proc() {
	model := model_make()
	defer model_destroy(&model)

	for {
		defer free_all(context.temp_allocator)

		if model.step >= STEPS {
			evaluate(&model, MODEL_FILE)
			break
		}

		improve(&model)

		if model.step % EVALUATION_INTERVAL == 0 {
			score := evaluate(&model, MODEL_FILE)
			fmt.printfln("%v Score: %.2f", model.step, score)

			if score >= game.SOLVE_SCORE {
				break
			}
		}
	}
}

play :: proc() {
	model := model_load(MODEL_FILE)
	defer model_destroy(&model)

	timestep: utility.Fixed_Timestep

	game_state := &model.game_state

	game_state.high_score = 0
	game.reset(game_state)

	game.open_window()
	defer game.close_window()

	for !game.window_should_close() {
		defer free_all(context.temp_allocator)

		game.begin_frame()

		for utility.fixed_timestep(&timestep, game.FIXED_DELTA) {
			action, _ := choose_action(model.actor, game.embedding(game_state^))
			_, done   := game.step(game_state, action, game.FIXED_DELTA)
			if done {
				game.reset(game_state)
			}
		}

		game.draw(game_state^, timestep.interpolation, is_human=true)

		game.end_frame()
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

Checkpoint :: struct {
	actor:  Network,
	critic: Network,

	opt: ml.Optimizer,

	step:       int,
	best_score: f32,
}

Model :: struct {
	actor:  Network,
	critic: Network,

	opt: ml.Optimizer,

	step:       int,
	best_score: f32,

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

model_load :: proc(file_name: string, allocator := context.allocator) -> (model: Model) {
	data, file_err := os.read_entire_file(file_name, allocator=context.temp_allocator)
	if file_err != nil {
		fmt.println("Failed to load model file")
		return model_make()
	}

	checkpoint: Checkpoint
	json_err := json.unmarshal(data, &checkpoint, allocator=allocator)
	if json_err != nil {
		fmt.println("Failed to unmarshal model from JSON")
		return model_make()
	}

	model.actor      = checkpoint.actor
	model.critic     = checkpoint.critic
	model.opt        = checkpoint.opt
	model.step       = checkpoint.step
	model.best_score = checkpoint.best_score

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

model_save :: proc(model: Model, file_name: string) {
	checkpoint := Checkpoint{
		actor      = model.actor,
		critic     = model.critic,
		opt        = model.opt,
		step       = model.step,
		best_score = model.best_score,
	}

	data, json_err := json.marshal(checkpoint)
	if json_err != nil {
		return
	}

	_ = os.write_entire_file(file_name, data)
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

	for {
		frame: Frame

		frame.embedding = game.embedding(model.game_state)

		ml.clear()
		frame.value = predict_value(model.critic, frame.embedding)

		frame.action, frame.log_probability = choose_action(model.actor, frame.embedding, sample=true)

		done: bool
		frame.reward, done = game.step(&model.game_state, frame.action, game.FIXED_DELTA)

		append(&model.frames, frame)

		if done {
			break
		}
	}

	// Calculate generalized advantage estimate.
	gae: f32
	for i := len(model.frames) - 1; i >= start_index; i -= 1 {
		value      := model.frames[i].value
		next_value := i + 1 < len(model.frames) ? model.frames[i + 1].value : 0

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

	// Small epsilon to prevent division by zero.
	if std > 1e-8 {
		for &frame in model.frames {
			frame.advantage = (frame.advantage - mean) / std
		}
	}
}

play_game :: proc(model: ^Model) -> (score: f32) {
	game.reset(&model.game_state)

	for {
		action, _ := choose_action(model.actor, game.embedding(model.game_state))
		_, done   := game.step(&model.game_state, action, game.FIXED_DELTA)
		if done {
			score = model.game_state.score
			break
		}
	}

	return
}

evaluate :: proc(model: ^Model, save_file: string) -> (score: f32) {
	for _ in 0 ..< EVALUATION_GAMES {
		score += play_game(model)
	}
	score /= f32(EVALUATION_GAMES)

	if score > model.best_score {
		model.best_score = score
		model_save(model^, save_file)
	}

	return
}

improve :: proc(model: ^Model) {
	clear(&model.frames)

	for _ in 0 ..< TRAJECTORIES {
		record_trajectory(model)
	}

	normalize_advantages(model^)

	lr := utility.linear_learning_rate(LEARNING_RATE, 0, model.step, STEPS)

	for _ in 0 ..< EPOCHS {
		rand.shuffle(model.frames[:])

		for &frame in model.frames {
			// Calculate actor gradients.
			ml.clear()

			logits            := network_forward(model.actor, frame.embedding[:])
			log_probabilities := ml.log_softmax(logits)
			log_probability   := ml.select(log_probabilities, {int(frame.action)})

			ratio         := ml.exp(ml.sub(log_probability, ml.scalar(frame.log_probability)))
			clipped_ratio := ml.clamp(ratio, 1.0 - CLIP_EPSILON, 1.0 + CLIP_EPSILON)

			advantage         := ml.scalar(frame.advantage)
			objective         := ml.mul(ratio,         advantage)
			clipped_objective := ml.mul(clipped_ratio, advantage)

			actor_loss := ml.mul(ml.min(objective, clipped_objective), ml.scalar(-1))

			when ENTROPY > 0 {
				probabilities := ml.softmax(logits)
				entropy       := ml.entropy(probabilities)
				entropy_loss  := ml.mul(entropy, ml.scalar(-ENTROPY))
				actor_loss     = ml.add(actor_loss, entropy_loss)
			}

			ml.backward()

			// Calculate critic gradients.
			ml.clear()

			value         := network_forward(model.critic, frame.embedding[:])
			clipped_value := ml.clamp(value, frame.value - CLIP_EPSILON, frame.value + CLIP_EPSILON)

			target := ml.scalar(frame.discounted_return)

			unclipped_loss := ml.mean_squared_error(value,         target)
			clipped_loss   := ml.mean_squared_error(clipped_value, target)

			critic_loss := ml.max(unclipped_loss, clipped_loss)

			ml.backward()

			if ml.optimize(&model.opt, period=PERIOD, learning_rate=lr) {
				network_update(model.opt, model.actor)
				network_update(model.opt, model.critic)
			}
		}
	}

	model.step += 1
}