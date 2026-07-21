package main

import cpu "../../backends/cpu"

import "sim"
import "agent"
import "world"

THREAD_COUNT :: 4

main :: proc() {
	cpu.set_thread_count(THREAD_COUNT)

	game_state: sim.State
	sim.init(&game_state)
	defer sim.destroy(&game_state)

	brain := agent.make(sim.reward)
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
				sim.mouse_end(&game_state)
				human_begin(&controls)
			}
			else {
				human_end()
			}
		}

		if !human {
			if mouse_pressed() {
				sim.mouse_begin(&game_state, mouse_position())
			}
			if mouse_held() {
				game_state.mouse_target = mouse_position()
			}
			else {
				sim.mouse_end(&game_state)
			}
		}

		if human {
			human_accumulate(&controls)
		}

		applied: world.Action

		for fixed_timestep(&timestep, sim.FIXED_DELTA) {
			if !human {
				action := agent.act(brain)
				control = action[world.ACTION_AXIS_X]
				applied = action
			}
			else {
				control = human_consume(&controls)
				applied = world.Action{world.ACTION_AXIS_X=control}
			}

			done := sim.step(&game_state, control, sim.FIXED_DELTA)

			sim_time += f64(sim.FIXED_DELTA)
			agent.sense(brain, sim_time, sim.observe(game_state), applied, episode)

			if done {
				sim.reset(&game_state)
				episode += 1
			}
		}

		draw(game_state, timestep.interpolation)
		draw_status(human, agent.decisions(brain), agent.policy_match(brain))

		frame_end()
	}
}
