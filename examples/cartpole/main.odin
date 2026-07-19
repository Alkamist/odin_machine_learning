package main

import "core:math"

import cpu "../../backends/cpu"

import "./agent"

FIXED_DELTA :: 1.0 / 60.0

PIXELS_PER_METER :: 24

THREAD_COUNT :: 4

X_SCALE :: f32(CART_LIMIT)
V_SCALE :: f32(CART_SPEED)
W_SCALE :: f32(8)

ENERGY_SCALE :: f32(POLE_SIZE.y) / (6.0 * GRAVITY)

UPRIGHT_WEIGHT :: f32(3)
ENERGY_WEIGHT  :: f32(3)
CENTER_WEIGHT  :: f32(3)
SPIN_WEIGHT    :: f32(2)
BARRIER_ONSET  :: f32(0.5)
BARRIER_WEIGHT :: f32(20)

#assert(len(Action) == agent.ACTION_COUNT)

@(require_results)
observe :: proc(state: State) -> (sensor: [agent.SENSOR_SIZE]f32) {
	position := cart_position(state)
	velocity := cart_velocity(state)
	angle    := pole_angle(state)
	spin     := pole_spin(state)

	sensor[0] = position / X_SCALE
	sensor[1] = velocity / V_SCALE
	sensor[2] = math.sin(angle)
	sensor[3] = math.cos(angle)
	sensor[4] = spin / W_SCALE
	return
}

@(require_results)
reward :: proc(sensor: [agent.SENSOR_SIZE]f32) -> (reward: f32, dead: bool) {
	cos_angle := sensor[3]
	spin      := sensor[4] * W_SCALE

	upright := -cos_angle
	energy  := ENERGY_SCALE * spin * spin + 0.5 * (1 - cos_angle)

	energy_error := energy - 1

	reward  = UPRIGHT_WEIGHT * upright
	reward -= ENERGY_WEIGHT * energy_error * energy_error
	reward -= CENTER_WEIGHT * sensor[0] * sensor[0]

	if upright > 0 {
		reward -= SPIN_WEIGHT * upright * sensor[4] * sensor[4]
	}

	barrier := max(abs(sensor[0]) - BARRIER_ONSET, 0)
	reward  -= BARRIER_WEIGHT * barrier * barrier

	dead = abs(sensor[0]) > 0.9
	return
}

main :: proc() {
	cpu.set_thread_count(THREAD_COUNT)

	game_state: State
	init(&game_state)
	defer destroy(&game_state)

	brain := agent.make(reward)
	defer agent.destroy(brain)
	agent.start(brain)
	defer agent.stop(brain)

	human:    bool
	action:   Action
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
				action = Action(agent.act(brain))
			}

			done := step(&game_state, action, FIXED_DELTA)

			sim_time += FIXED_DELTA
			agent.sense(brain, sim_time, observe(game_state), int(action), episode)

			if done {
				reset(&game_state)
				episode += 1
			}
		}

		draw(game_state, timestep.interpolation)
		draw_status(human, agent.decisions(brain), agent.agreement(brain))

		frame_end()
	}
}
