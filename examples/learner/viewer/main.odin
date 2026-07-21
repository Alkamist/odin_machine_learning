package main

import cpu "../../../backends/cpu"

import "../cartpole"
import "../agent"
import "../world"

THREAD_COUNT :: 4

main :: proc() {
	cpu.set_thread_count(THREAD_COUNT)

	game_state: cartpole.State
	cartpole.init(&game_state)
	defer cartpole.destroy(&game_state)

	brain := agent.make(cartpole.reward)
	defer agent.destroy(brain)
	agent.start(brain)
	defer agent.stop(brain)

	human:    bool
	control:  f32
	controls: Human_Control
	timestep: Fixed_Timestep

	sim_time: f64
	episode:  u64 = 1

	window_open()
	defer window_close()

	for !window_should_close() {
		defer free_all(context.temp_allocator)

		frame_begin()

		if toggle_pressed() {
			human = !human
			if human {
				cartpole.mouse_end(&game_state)
				human_begin(&controls)
			}
			else {
				human_end()
			}
		}

		if !human {
			if mouse_pressed() {
				cartpole.mouse_begin(&game_state, mouse_position())
			}
			if mouse_held() {
				game_state.mouse_target = mouse_position()
			}
			else {
				cartpole.mouse_end(&game_state)
			}
		}

		if human {
			human_accumulate(&controls)
		}

		applied: world.Action

		for fixed_timestep(&timestep, cartpole.FIXED_DELTA) {
			if !human {
				action := agent.act(brain)
				control = action[world.ACTION_AXIS_X]
				applied = action
			}
			else {
				control = human_consume(&controls)
				applied = world.Action{world.ACTION_AXIS_X=control}
			}

			done := cartpole.step(&game_state, control, cartpole.FIXED_DELTA)

			sim_time += f64(cartpole.FIXED_DELTA)
			agent.sense(brain, sim_time, cartpole.observe(game_state), applied, episode)

			if done {
				cartpole.reset(&game_state)
				episode += 1
			}
		}

		draw(game_state, timestep.interpolation)
		draw_status(human, agent.decisions(brain), agent.policy_match(brain))

		frame_end()
	}
}
