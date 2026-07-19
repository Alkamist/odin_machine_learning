package main

import ml  "../../"
import cpu "../../backends/cpu"

FIXED_DELTA :: 1.0 / 60.0

PIXELS_PER_METER :: 24

THREAD_COUNT :: 4

main :: proc() {
	cpu.set_thread_count(THREAD_COUNT)

	ctx := cpu.context_create(1024 * 1024 * 256)
	defer cpu.context_destroy(ctx)

	ml.context_scope(ctx)

	game_state: State
	init(&game_state)
	defer destroy(&game_state)

	agent := agent_make()
	defer agent_destroy(agent)

	human:    bool
	action:   Action
	timestep: Fixed_Timestep

	window_open()
	defer window_close()

	for !window_should_close() {
		defer free_all(context.temp_allocator)

		frame_begin()

		if toggle_pressed() {
			human = !human
		}

		if mouse_pressed() {
			mouse_begin(&game_state, mouse_position())
		}
		if mouse_held() {
			game_state.mouse_target = mouse_position()
		}
		else {
			mouse_end(&game_state)
		}

		if human {
			action = human_action(action)
		}

		for fixed_timestep(&timestep, FIXED_DELTA) {
			if !human {
				action = agent_step(agent, game_state)
			}

			if step(&game_state, action, FIXED_DELTA) {
				reset(&game_state)
				agent_forget_episode(agent)
			}
		}

		draw(game_state, timestep.interpolation)
		draw_status(human, agent.decisions, agent.agreement)

		frame_end()
	}
}
