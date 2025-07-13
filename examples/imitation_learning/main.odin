package main

import "core:fmt"
import "core:math/rand"
import "core:slice"
import "../utility"
import "../../ml"
import "../../ml/mlp"

import game "../cartpole"


// In this example, you can play either CartPole or Circles by
// swapping out the import above. An agent will learn from your
// recorded gameplay.


main :: proc() {
	defer fmt.println("Finished")

	ml.init(1024 * 1024)

	model := make_model()
	defer destroy_model(model)

	// The replay buffer is too big for the stack because
	// it is statically holding the frames.
	replay_buffer := new(Replay_Buffer)
	defer free(replay_buffer)

	timestep: utility.Fixed_Timestep

	human_game_state: game.State
	game.init(&human_game_state)
	defer game.destroy(&human_game_state)

	agent_game_state: game.State
	game.init(&agent_game_state)
	defer game.destroy(&agent_game_state)

	human_action: game.Action

	game.open_window()
	defer game.close_window()

	for !game.window_should_close() {
		defer free_all(context.temp_allocator)

		// Have the agent learn from a random sampling of the replay buffer.
		if replay_buffer.len >= 60 * 10 {
			for _ in 0 ..< 128 {
				i := rand.int_max(replay_buffer.len)
				learn(&model, replay_buffer.frames[i])
			}
		}

		game.begin_frame()

		human_action = game.human_action(human_action)

		for utility.fixed_timestep(&timestep, game.FIXED_DELTA) {
			// Poll the human's action, add the human's game state and action
			// to the replay buffer, and then step the human's game state.
			add_frame(replay_buffer, Frame{game.embedding(human_game_state), human_action})
			_, human_done := game.step(&human_game_state, human_action, game.FIXED_DELTA)

			// Sample the agent's action and step the agent's game state.
			agent_action  := choose_action(model, game.embedding(agent_game_state))
			_, agent_done := game.step(&agent_game_state, agent_action, game.FIXED_DELTA)

			// Reset game states when done.
			if human_done {
				game.reset(&human_game_state)
			}
			if agent_done {
				game.reset(&agent_game_state)
			}
		}

		game.draw(agent_game_state, timestep.interpolation, is_human=false)
		game.draw(human_game_state, timestep.interpolation, is_human=true)

		game.end_frame()
	}
}

Model :: struct {
	mlp: mlp.Mlp,
	opt: ml.Optimizer,
}

make_model :: proc(allocator := context.allocator) -> (res: Model) {
	res.mlp = mlp.make(len(game.Embedding), 128, len(game.Action))
	return
}

destroy_model :: proc(model: Model) {
	mlp.destroy(model.mlp)
}

forward :: proc(model: Model, embedding: game.Embedding) -> ml.Array {
	embedding := embedding
	return mlp.forward(model.mlp, ml.array(embedding[:]))
}

choose_action :: proc(model: Model, embedding: game.Embedding) -> (res: game.Action) {
	ml.clear()

	logits        := forward(model, embedding)
	probabilities := ml.softmax(logits)

	return game.Action(utility.sample_probability_distribution(probabilities.data))
}

learn :: proc(model: ^Model, frame: Frame) {
	ml.clear()

	logits := forward(model^, frame.embedding)
	_       = ml.cross_entropy(logits, {int(frame.action)})

	ml.backward()

	if ml.optimize(&model.opt) {
		mlp.update(model.opt, model.mlp)
	}
}

Frame :: struct {
	embedding: game.Embedding,
	action:    game.Action,
}

Replay_Buffer :: struct {
	len:    int,
	index:  int,
	frames: [60 * 60 * 5]Frame,
}

add_frame :: proc(buffer: ^Replay_Buffer, frame: Frame) {
	buffer.frames[buffer.index] = frame
	buffer.len = min(buffer.len + 1, len(buffer.frames))
	buffer.index += 1
	if buffer.index >= len(buffer.frames) {
		buffer.index = 0
	}
}