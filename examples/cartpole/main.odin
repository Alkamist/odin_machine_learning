package main

FIXED_DELTA :: 1.0 / 60.0

PIXELS_PER_METER :: 24

main :: proc() {
	game_state: State
	init(&game_state)
	defer destroy(&game_state)

	action:   Action
	timestep: Fixed_Timestep

	window_open()
	defer window_close()

	for !window_should_close() {
		defer free_all(context.temp_allocator)

		frame_begin()

		action = human_action(action)

		for fixed_timestep(&timestep, FIXED_DELTA) {
			if step(&game_state, action, FIXED_DELTA) {
				reset(&game_state)
			}
		}

		draw(game_state, timestep.interpolation)

		frame_end()
	}
}
