package main

import "core:math"

import rl "vendor:raylib"

import "sim"

window_open :: proc() {
	rl.SetConfigFlags({.WINDOW_RESIZABLE})
	rl.InitWindow(1280, 720, "CartPole")
	rl.SetTargetFPS(240)
}

window_close :: proc() {
	rl.CloseWindow()
}

window_should_close :: proc() -> bool {
	return rl.WindowShouldClose()
}

frame_begin :: proc() {
	rl.BeginDrawing()
	rl.ClearBackground({12, 12, 12, 255})
}

frame_end :: proc() {
	rl.EndDrawing()
}

toggle_pressed :: proc() -> bool {
	return rl.IsKeyPressed(.TAB)
}

@(require_results)
mouse_position :: proc() -> [2]f32 {
	position := rl.GetMousePosition()
	return {
		position.x - f32(rl.GetScreenWidth())  / 2.0,
		-(position.y - f32(rl.GetScreenHeight()) / 2.0),
	}
}

@(require_results)
mouse_held :: proc() -> bool {
	return rl.IsMouseButtonDown(.LEFT)
}

@(require_results)
mouse_pressed :: proc() -> bool {
	return rl.IsMouseButtonPressed(.LEFT)
}

MOUSE_GAIN  :: f32(1.0)
MOUSE_SPEED :: f32(2000)

Human_Control :: struct {
	pending:  f32,
	settling: bool,
}

human_begin :: proc(controls: ^Human_Control) {
	rl.DisableCursor()
	controls^ = {settling = true}
}

human_end :: proc() {
	rl.EnableCursor()
}

human_accumulate :: proc(controls: ^Human_Control) {
	mouse_delta := rl.GetMouseDelta()
	if controls.settling {
		controls.settling = false
		return
	}
	controls.pending += mouse_delta.x * MOUSE_GAIN
}

human_consume :: proc(controls: ^Human_Control) -> f32 {
	velocity        := clamp(controls.pending / sim.FIXED_DELTA, -MOUSE_SPEED, MOUSE_SPEED)
	controls.pending = 0
	return velocity / sim.CART_SPEED
}

box_draw :: proc(box: sim.Box, color: rl.Color, interpolation: f32) {
	position := math.lerp(box.position_, sim.box_position(box), interpolation)
	rotation := lerp_angle(-rl.RAD2DEG * box.rotation_, -rl.RAD2DEG * sim.box_rotation(box), interpolation)
	rl.DrawRectanglePro(
		{position.x, -position.y, box.size.x, box.size.y},
		box.size / 2.0,
		rotation,
		color,
	)
}

draw :: proc(state: sim.State, interpolation: f32) {
	camera: rl.Camera2D
	camera.offset = {
		f32(rl.GetScreenWidth())  / 2.0,
		f32(rl.GetScreenHeight()) / 2.0,
	}
	camera.zoom = 1
	rl.BeginMode2D(camera)

	box_draw(state.left_wall,  rl.RED, interpolation)
	box_draw(state.right_wall, rl.RED, interpolation)

	box_draw(state.cart, rl.DARKBLUE, interpolation)
	box_draw(state.pole, rl.GREEN,    interpolation)

	if state.mouse_active {
		position := math.lerp(state.mouse_position_, sim.mouse_position(state), interpolation)
		rl.DrawCircleV({position.x, -position.y}, sim.MOUSE_RADIUS, rl.YELLOW)
	}

	position := math.lerp(state.cart.position_, sim.box_position(state.cart), interpolation)
	draw_text_centered(rl.TextFormat("%.2f", state.score), 10, position.x, position.y + 50, rl.WHITE)

	rl.EndMode2D()

	rl.DrawText(rl.TextFormat("High Score: %.2f", state.high_score),            20, 68, 20, rl.WHITE)
	rl.DrawText(rl.TextFormat("Time: %.2f",       sim.TIME_LIMIT - state.time), 20, 92, 20, rl.WHITE)
}

draw_status :: proc(human: bool, decisions: int, agreement: f32) {
	if human {
		rl.DrawText("Human (TAB to hand back to the agent) - move the mouse to steer", 20, 20, 20, rl.WHITE)
	}
	else {
		rl.DrawText(rl.TextFormat("Agent, %d decisions learned (TAB to take over)", decisions), 20, 20, 20, rl.WHITE)
	}

	rl.DrawText(rl.TextFormat("%d FPS", rl.GetFPS()), 20, 44, 20, rl.WHITE)
	rl.DrawText(rl.TextFormat("Reflex agreement: %.0f%%", agreement * 100), 20, 116, 20, rl.WHITE)
}

draw_text :: proc(text: cstring, font_size: int, x, y: f32, color: rl.Color) {
	rl.DrawText(text, i32(x), i32(-y), i32(font_size), color)
}

draw_text_centered :: proc(text: cstring, font_size: int, x, y: f32, color: rl.Color) {
	width := rl.MeasureText(text, i32(font_size))
	rl.DrawText(text, i32(x) - width / 2, i32(-y), i32(font_size), color)
}
