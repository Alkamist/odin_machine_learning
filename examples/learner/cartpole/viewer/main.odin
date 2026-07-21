package main

import "core:math"

import rl "vendor:raylib"

import cpu "../../../../backends/cpu"

import          "../../agent"
import          "../../utility"
import cartpole ".."

THREAD_COUNT :: 4

MOUSE_GAIN  :: f32(1.0)
MOUSE_SPEED :: f32(2000)

Human_Control :: struct {
	pending:  f32,
	settling: bool,
}

_human_begin :: proc(controls: ^Human_Control) {
	rl.DisableCursor()
	controls^ = {settling=true}
}

_human_end :: proc() {
	rl.EnableCursor()
}

_human_accumulate :: proc(controls: ^Human_Control) {
	mouse_delta := rl.GetMouseDelta()
	if controls.settling {
		controls.settling = false
		return
	}
	controls.pending += mouse_delta.x * MOUSE_GAIN
}

@(require_results)
_human_consume :: proc(controls: ^Human_Control) -> f32 {
	velocity        := clamp(controls.pending / cartpole.FIXED_DELTA, -MOUSE_SPEED, MOUSE_SPEED)
	controls.pending = 0
	return velocity / cartpole.CART_SPEED
}

@(require_results)
_mouse_position :: proc() -> [2]f32 {
	position := rl.GetMousePosition()
	return {
		  position.x - f32(rl.GetScreenWidth())  / 2.0,
		-(position.y - f32(rl.GetScreenHeight()) / 2.0),
	}
}

main :: proc() {
	cpu.set_thread_count(THREAD_COUNT)

	game: cartpole.State
	cartpole.init(&game)
	defer cartpole.destroy(&game)

	brain := agent.make(cartpole.reward)
	defer agent.destroy(brain)
	agent.start(brain)
	defer agent.stop(brain)

	human:    bool
	control:  f32
	controls: Human_Control
	timestep: utility.Fixed_Timestep

	sim_time: f64
	episode:  u64 = 1

	rl.SetConfigFlags({.WINDOW_RESIZABLE})
	rl.InitWindow(1280, 720, "CartPole")
	defer rl.CloseWindow()
	rl.SetTargetFPS(240)

	for !rl.WindowShouldClose() {
		defer free_all(context.temp_allocator)

		if rl.IsKeyPressed(.TAB) {
			human = !human
			if human {
				cartpole.mouse_end(&game)
				_human_begin(&controls)
			}
			else {
				_human_end()
			}
		}

		if human {
			_human_accumulate(&controls)
		}
		else {
			if rl.IsMouseButtonPressed(.LEFT) {
				cartpole.mouse_begin(&game, _mouse_position())
			}
			if rl.IsMouseButtonDown(.LEFT) {
				game.mouse_target = _mouse_position()
			}
			else {
				cartpole.mouse_end(&game)
			}
		}

		applied: agent.Action

		for utility.fixed_timestep(&timestep, cartpole.FIXED_DELTA) {
			if human {
				control = _human_consume(&controls)
				applied = agent.Action{agent.ACTION_AXIS_X=control}
			}
			else {
				applied = agent.act(brain)
				control = applied[agent.ACTION_AXIS_X]
			}

			done := cartpole.step(&game, control, cartpole.FIXED_DELTA)

			sim_time += f64(cartpole.FIXED_DELTA)
			agent.sense(brain, sim_time, cartpole.observe(game), applied, episode)

			if done {
				cartpole.reset(&game)
				episode += 1
			}
		}

		rl.BeginDrawing()
		rl.ClearBackground({12, 12, 12, 255})

		_draw_world(game, timestep.interpolation)
		_draw_status(game, human, agent.decisions(brain), agent.policy_match(brain))

		rl.EndDrawing()
	}
}

_box_draw :: proc(box: cartpole.Box, color: rl.Color, interpolation: f32) {
	position := math.lerp(box.position_, cartpole.box_position(box), interpolation)
	rotation := utility.lerp_angle(-rl.RAD2DEG * box.rotation_, -rl.RAD2DEG * cartpole.box_rotation(box), interpolation)
	rl.DrawRectanglePro(
		{position.x, -position.y, box.size.x, box.size.y},
		box.size / 2.0,
		rotation,
		color,
	)
}

_draw_text_centered :: proc(text: cstring, font_size: int, x, y: f32, color: rl.Color) {
	width := rl.MeasureText(text, i32(font_size))
	rl.DrawText(text, i32(x) - width / 2, i32(-y), i32(font_size), color)
}

_draw_world :: proc(game: cartpole.State, interpolation: f32) {
	camera: rl.Camera2D
	camera.offset = {
		f32(rl.GetScreenWidth())  / 2.0,
		f32(rl.GetScreenHeight()) / 2.0,
	}
	camera.zoom = 1
	rl.BeginMode2D(camera)

	_box_draw(game.left_wall,  rl.RED, interpolation)
	_box_draw(game.right_wall, rl.RED, interpolation)

	_box_draw(game.cart, rl.DARKBLUE, interpolation)
	_box_draw(game.pole, rl.GREEN,    interpolation)

	if game.mouse_active {
		position := math.lerp(game.mouse_position_, cartpole.mouse_position(game), interpolation)
		rl.DrawCircleV({position.x, -position.y}, cartpole.MOUSE_RADIUS, rl.YELLOW)
	}

	position := math.lerp(game.cart.position_, cartpole.box_position(game.cart), interpolation)
	_draw_text_centered(rl.TextFormat("%.2f", game.score), 10, position.x, position.y + 50, rl.WHITE)

	rl.EndMode2D()
}

_draw_status :: proc(game: cartpole.State, human: bool, decisions: int, policy_match: f32) {
	if human {
		rl.DrawText("Human (TAB to hand back to the agent) - move the mouse to steer", 20, 20, 20, rl.WHITE)
	}
	else {
		rl.DrawText(rl.TextFormat("Agent, %d decisions learned (TAB to take over)", decisions), 20, 20, 20, rl.WHITE)
	}

	rl.DrawText(rl.TextFormat("%d FPS", rl.GetFPS()), 20, 44, 20, rl.WHITE)
	rl.DrawText(rl.TextFormat("High Score: %.2f", game.high_score), 20, 68, 20, rl.WHITE)
	rl.DrawText(rl.TextFormat("Time: %.2f", cartpole.TIME_LIMIT - game.time), 20, 92, 20, rl.WHITE)
	rl.DrawText(rl.TextFormat("Policy match: %.0f%%", policy_match * 100), 20, 116, 20, rl.WHITE)
}
