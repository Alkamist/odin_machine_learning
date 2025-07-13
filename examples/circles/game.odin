package circles

import "core:time"
import "core:math"
import "core:math/rand"
import "core:math/linalg"
import rl "vendor:raylib"
import "../utility"


// In this game, the goal is to move the smaller circle into
// the bigger circle, as centered as possible.


FIXED_DELTA :: 1.0 / 60.0

TIME_LIMIT :: 6

SOLVE_SCORE :: 44.8

open_window :: proc() {
    rl.SetConfigFlags({.WINDOW_RESIZABLE})
    rl.InitWindow(1280, 720, "Circles")
}

close_window :: proc() {
    rl.CloseWindow()
}

window_should_close :: proc() -> bool {
    return rl.WindowShouldClose()
}

human_action :: proc(previous: Action) -> (res: Action) {
    if rl.IsKeyDown(.A) do res = .Left
    if rl.IsKeyDown(.D) do res = .Right
    if rl.IsKeyDown(.S) do res = .Down
    if rl.IsKeyDown(.W) do res = .Up
    return
}

begin_frame :: proc() {
    rl.BeginDrawing()
    rl.ClearBackground({12, 12, 12, 255})
}

end_frame :: proc() {
    rl.EndDrawing()
}

Embedding :: [6]f32

embedding :: proc(state: State) -> Embedding {
	return {
		state.position.x / 1000.0, state.position.y / 1000.0,
		state.velocity.x / 1000.0, state.velocity.y / 1000.0,
		state.goal.x     / 1000.0, state.goal.y     / 1000.0,
	}
}

Action :: enum {
	None,
	Left,
	Right,
	Down,
	Up,
}

State :: struct {
    high_score: f32,
	score:      f32,
	time:       f32,

	velocity: [2]f32,
	position: [2]f32,
	goal:     [2]f32,

    position_: [2]f32, // Previous position for visual interpolation
}

init :: proc(state: ^State) {
	reset(state)
}

destroy :: proc(state: ^State) {
}

reset :: proc(state: ^State) {
	state.score    = 0
	state.time     = 0
	state.position = {rand.float32_range(-50, 50), rand.float32_range(-50, 50)}
	state.velocity = 0
	state.goal     = state.position + utility.rotate([2]f32{0, 200}, rand.float32_range(0, math.PI * 2))
}

step :: proc(state: ^State, action: Action, delta: f32) -> (reward: f32, done: bool) {
	state.time += delta

	switch action {
	case .None:
	case .Left:  state.velocity.x += -300 * delta
	case .Right: state.velocity.x +=  300 * delta
	case .Down:  state.velocity.y += -300 * delta
	case .Up:    state.velocity.y +=  300 * delta
	}

    state.position_ = state.position
	state.position += state.velocity * delta

	distance := linalg.distance(state.position, state.goal)

	reward = max(0, 50 - distance) * 0.2 * delta
	state.score += reward

    if state.time > TIME_LIMIT {
        if state.score > state.high_score {
            state.high_score = state.score
        }
        done = true
    }

	return
}

draw :: proc(state: State, interpolation: f32, is_human := true) {
    // Offset the camera so that {0, 0} is in the middle of the window.
    camera: rl.Camera2D
    camera.offset = {
        f32(rl.GetScreenWidth())  / 2.0, 
        f32(rl.GetScreenHeight()) / 2.0,
    }
    camera.zoom = 1
    rl.BeginMode2D(camera)

    // Agent colors are partially transparent.
    goal_color   := rl.DARKGREEN
    player_color := rl.YELLOW
    if !is_human {
        goal_color.a   = 32
        player_color.a = 32
    }

    // The player and goal are just simple circles.
    PLAYER_RADIUS :: 16

    position := math.lerp(state.position_, state.position, interpolation)

    rl.DrawCircleV({state.goal.x, -state.goal.y}, 50,            color=goal_color)
    rl.DrawCircleV({position.x,   -position.y},   PLAYER_RADIUS, color=player_color)

    draw_text_centered(rl.TextFormat("%.2f", state.score), 10, position.x, position.y + PLAYER_RADIUS  + 12, rl.WHITE)

    if is_human {
        draw_text         (rl.TextFormat("High Score: %.2f", state.high_score),  20, -500, 340, rl.WHITE)
        draw_text_centered(rl.TextFormat("Time: %.2f", TIME_LIMIT - state.time), 20, 0,    340, rl.WHITE)
    } else {
        text_color := rl.WHITE
        text_color.a = 32
        draw_text(rl.TextFormat("High Score: %.2f", state.high_score), 20, -280, 340, text_color)
    }

    rl.EndMode2D()
}

draw_text :: proc(text: cstring, font_size: int, x, y: f32, color: rl.Color) {
	rl.DrawText(text, i32(x), i32(-y), i32(font_size), color)
}

draw_text_centered :: proc(text: cstring, font_size: int, x, y: f32, color: rl.Color) {
	width := rl.MeasureText(text, i32(font_size))
	rl.DrawText(text, i32(x) - width / 2, i32(-y), i32(font_size), color)
}