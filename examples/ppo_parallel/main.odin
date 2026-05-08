// Parallel variant of the PPO example. Trajectory collection is the
// only phase that runs across multiple host threads: each worker owns
// its own ml.Context, game state, and frame buffer, and reads the
// shared actor/critic weights for inference. The learning phase stays
// single-threaded so the optimizer state remains coherent.

package main

import "core:os"
import "core:fmt"
import "core:math"
import "core:math/rand"
import "core:slice"
import "core:sync"
import "core:thread"
import "core:encoding/json"
import "../utility"
import ml "../../"
import cpu "../../backends/cpu"
import gpu "../../backends/gpu"
import "../../networks/mlp"

import game "../cartpole"

MODEL_FILE :: "model.json"

STEPS         :: 5000  // How many steps until training is done, also decays learning rate.
EPOCHS        :: 4     // Pass over the frames of each step this many times. Typically 1-10.
TRAJECTORIES  :: 32    // Learn from this many games each PPO step.
LEARNING_RATE :: 0.001 // How fast the model should learn.
PERIOD        :: 128   // How many frames to accumulate gradients for.

PARALLEL_INSTANCES :: 8
WORKER_ARENA_SIZE  :: 1 * 1024 * 1024

HIDDEN_SIZE :: 128 // How powerful the multilayer perceptron is.

EVALUATION_GAMES    :: 1 // How many games to evaluate the model on.
EVALUATION_INTERVAL :: 1 // How often to evaluate the model.

GAMMA              :: 0.99 // How much future rewards are worth relative to immediate ones.
LAMBDA             :: 0.95 // How much future rewards contribute to advantage estimates.
CLIP_EPSILON       :: 0.2  // How much the policy is allowed to change each update.
VALUE_CLIP_EPSILON :: 0.5  // How much the value function is allowed to change each update.
ENTROPY            :: 0.01 // How much is exploration encouraged.

main :: proc() {
	defer fmt.println("Finished")

	// cpu.set_thread_count(24)

	ctx := cpu.context_create(1024 * 1024)
	defer cpu.context_destroy(ctx)

	// ctx := gpu.context_create()
	// defer gpu.context_destroy(ctx)

	ml.context_scope(ctx)

	train()
	// play()
}

train :: proc() {
	model := model_make()
	defer model_destroy(&model)

	pool := pool_make(&model)
	defer pool_destroy(pool)

	for {
		defer free_all(context.temp_allocator)

		if model.step >= STEPS {
			evaluate(&model, MODEL_FILE)
			break
		}

		improve(&model, pool)

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
			action, _  := choose_action(model.actor, game.embedding(game_state^))
			_, done, _ := game.step(game_state, action, game.FIXED_DELTA)
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
		fmt.eprintfln("Failed to marshal json: %v",json_err)
		return
	}

	file_err := os.write_entire_file(file_name, data)
	if file_err != nil {
		fmt.eprintfln("Failed to save model file: %v", file_err)
		return
	}
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

Pool :: struct {
	workers:  []^Worker,
	done_wg:  sync.Wait_Group,
	shutdown: bool,
}

Worker :: struct {
	pool:       ^Pool,
	thread:     ^thread.Thread,
	start_sem:  sync.Sema,

	ctx:        ^ml.Context,
	game_state: game.State,
	frames:     [dynamic]Frame,

	actor:  Network,
	critic: Network,

	start, end: int,
}

pool_make :: proc(model: ^Model) -> ^Pool {
	pool         := new(Pool)
	pool.workers  = make([]^Worker, PARALLEL_INSTANCES)
	for i in 0 ..< PARALLEL_INSTANCES {
		w     := new(Worker)
		w.pool = pool
		w.ctx  = cpu.context_create(WORKER_ARENA_SIZE)
		w.actor  = model.actor
		w.critic = model.critic
		w.frames = make([dynamic]Frame, 0, 60 * 60 * TRAJECTORIES / PARALLEL_INSTANCES + 60 * 60)
		game.init(&w.game_state)
		pool.workers[i] = w
	}

	when PARALLEL_INSTANCES > 1 {
		for w in pool.workers {
			w.thread      = thread.create(worker_loop)
			w.thread.data = w
			thread.start(w.thread)
		}
	}

	return pool
}

pool_destroy :: proc(pool: ^Pool) {
	when PARALLEL_INSTANCES > 1 {
		pool.shutdown = true
		for w in pool.workers {
			sync.sema_post(&w.start_sem)
		}
		for w in pool.workers {
			thread.join(w.thread)
			thread.destroy(w.thread)
		}
	}

	for w in pool.workers {
		game.destroy(&w.game_state)
		delete(w.frames)
		cpu.context_destroy(w.ctx)
		free(w)
	}
	delete(pool.workers)
	free(pool)
}

worker_loop :: proc(t: ^thread.Thread) {
	w := cast(^Worker)t.data

	ml.context_scope(w.ctx)

	for {
		sync.sema_wait(&w.start_sem)
		if w.pool.shutdown do return

		for _ in w.start ..< w.end {
			record_trajectory_into(w)
			free_all(context.temp_allocator)
		}

		sync.wait_group_done(&w.pool.done_wg)
	}
}

record_trajectory_into :: proc(w: ^Worker) {
	game.reset(&w.game_state)

	start_index := len(w.frames)

	bootstrap:     f32
	truncated_end: bool

	for {
		frame: Frame

		frame.embedding = game.embedding(w.game_state)

		ml.clear()
		frame.value = predict_value(w.critic, frame.embedding)

		frame.action, frame.log_probability = choose_action(w.actor, frame.embedding, sample=true)

		done, truncated: bool
		frame.reward, done, truncated = game.step(&w.game_state, frame.action, game.FIXED_DELTA)

		append(&w.frames, frame)

		if done {
			if truncated {
				ml.clear()
				bootstrap     = predict_value(w.critic, game.embedding(w.game_state))
				truncated_end = true
			}
			break
		}
	}

	gae: f32
	for i := len(w.frames) - 1; i >= start_index; i -= 1 {
		value: f32 = w.frames[i].value

		next_value: f32
		if i + 1 < len(w.frames) {
			next_value = w.frames[i + 1].value
		} else if truncated_end {
			next_value = bootstrap
		}

		delta := w.frames[i].reward + GAMMA * next_value - value
		gae    = delta + GAMMA * LAMBDA * gae

		w.frames[i].advantage         = gae
		w.frames[i].discounted_return = value + gae
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
		action, _  := choose_action(model.actor, game.embedding(model.game_state))
		_, done, _ := game.step(&model.game_state, action, game.FIXED_DELTA)
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
		// model_save(model^, save_file)
	}

	return
}

collect_trajectories :: proc(model: ^Model, pool: ^Pool) {
	clear(&model.frames)

	for w, i in pool.workers {
		clear(&w.frames)
		// Even split with the remainder spread across the first few workers.
		base      := TRAJECTORIES / PARALLEL_INSTANCES
		remainder := TRAJECTORIES % PARALLEL_INSTANCES
		w.start = i * base + min(i, remainder)
		w.end   = w.start + base + (1 if i < remainder else 0)
	}

	when PARALLEL_INSTANCES <= 1 {
		// Single-instance path runs inline on the main thread to avoid
		// any synchronization cost.
		ml.context_scope(pool.workers[0].ctx)
		for _ in pool.workers[0].start ..< pool.workers[0].end {
			record_trajectory_into(pool.workers[0])
			free_all(context.temp_allocator)
		}
	} else {
		active := 0
		for w in pool.workers do if w.start < w.end do active += 1
		sync.wait_group_add(&pool.done_wg, active)
		for w in pool.workers {
			if w.start < w.end {
				sync.sema_post(&w.start_sem)
			}
		}
		sync.wait_group_wait(&pool.done_wg)
	}

	for w in pool.workers {
		for frame in w.frames {
			append(&model.frames, frame)
		}
	}
}

improve :: proc(model: ^Model, pool: ^Pool) {
	collect_trajectories(model, pool)

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

			// Calculate critic gradients.
			ml.clear()

			value         := network_forward(model.critic, frame.embedding[:])
			clipped_value := ml.clamp(value, frame.value - VALUE_CLIP_EPSILON, frame.value + VALUE_CLIP_EPSILON)

			target := ml.scalar(.F32, frame.discounted_return)

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