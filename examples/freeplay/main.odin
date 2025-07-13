package main

import "core:fmt"
import "../utility"

import game "../cartpole"


// In this example, you can freely play a game without any AI involved.


main :: proc() {
	defer fmt.println("Finished")

	game_state: game.State
	game.init(&game_state)
	defer game.destroy(&game_state)

	action:   game.Action
	timestep: utility.Fixed_Timestep
	
	game.open_window()
	defer game.close_window()

	for !game.window_should_close() {
		game.begin_frame()

		action = game.human_action(action)

		for utility.fixed_timestep(&timestep, game.FIXED_DELTA) {
			_, done := game.step(&game_state, action, game.FIXED_DELTA)
			if done {
				game.reset(&game_state)
			}
		}

		game.draw(game_state, timestep.interpolation)

		game.end_frame()
	}
}