package main

import "core:math"

import rl "vendor:raylib"

import        "../../agent"
import        "../../utility"
import lander ".."

THREAD_COUNT :: 4

CAMERA_ZOOM   :: f32(0.55)
CAMERA_TARGET :: [2]f32{0, -60}

MOUSE_GAIN       :: f32(-1.0)
MOUSE_SPIN_SPEED :: f32(600)

PAD_HEIGHT :: f32(8)

FLAME_LENGTH :: f32(70)
FLAME_WIDTH  :: f32(26)

OUTCOME_HOLD :: f32(2.0)

Reason :: enum {
	None,
	Too_Fast,
	Too_Tilted,
	Off_Map,
	Flew_Away,
}

@(require_results)
_verdict :: proc(sensor: []f32) -> (reason: Reason, value, limit: f32) {
	velocity_x := sensor[lander.SENSOR_VELOCITY_X]
	velocity_y := sensor[lander.SENSOR_VELOCITY_Y]

	speed     := math.sqrt(velocity_x * velocity_x + velocity_y * velocity_y)
	cos_angle := sensor[lander.SENSOR_ANGLE_COS]
	offset    := abs(sensor[lander.SENSOR_X])
	height    := sensor[lander.SENSOR_Y]
	contact   := sensor[lander.SENSOR_CONTACT] > 0.5

	switch {
	case contact && speed > lander.LAND_SPEED_NORM:
		return .Too_Fast, speed, lander.LAND_SPEED_NORM
	case contact && cos_angle < lander.LAND_UPRIGHT_COS:
		return .Too_Tilted, cos_angle, lander.LAND_UPRIGHT_COS
	case offset > lander.X_BOUND_NORM:
		return .Off_Map, offset, lander.X_BOUND_NORM
	case height > lander.H_MAX_NORM:
		return .Flew_Away, height, lander.H_MAX_NORM
	}
	return .None, 0, 0
}

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

_human_consume :: proc(controls: ^Human_Control, action: []f32) {
	action[lander.ACTION_AXIS_X] = clamp(controls.pending / lander.FIXED_DELTA / MOUSE_SPIN_SPEED, -1, 1)
	action[lander.ACTION_AXIS_Y] = rl.IsMouseButtonDown(.LEFT) ? 1 : 0
	controls.pending = 0
}

main :: proc() {
	game: lander.State
	lander.init(&game)
	defer lander.destroy(&game)

	brain := agent.create(lander.SENSOR_COUNT, lander.ACTION_COUNT, lander.reward, normalize=lander.normalize, compute_threads=THREAD_COUNT)
	defer agent.destroy(brain)

	human:    bool
	controls: Human_Control
	timestep: utility.Fixed_Timestep

	sensor:  [lander.SENSOR_COUNT]f32
	applied: [lander.ACTION_COUNT]f32

	sim_time: f64
	outcome:  lander.Outcome
	outcome_timer: f32
	attempts:      int
	landings:      int
	reason:        Reason
	reason_value:  f32
	reason_limit:  f32

	rl.SetConfigFlags({.WINDOW_RESIZABLE})
	rl.InitWindow(1280, 720, "Lunar Lander")
	defer rl.CloseWindow()
	rl.SetTargetFPS(240)

	for !rl.WindowShouldClose() {
		defer free_all(context.temp_allocator)

		if rl.IsKeyPressed(.TAB) {
			human = !human
			if human {
				_human_begin(&controls)
			}
			else {
				_human_end()
			}
		}

		if human {
			_human_accumulate(&controls)
		}

		for utility.fixed_timestep(&timestep, lander.FIXED_DELTA) {
			if human {
				_human_consume(&controls, applied[:])
			}
			else {
				agent.act(brain, applied[:])
			}

			done := lander.step(&game, applied[:], lander.FIXED_DELTA)

			lander.observe(game, sensor[:])

			sim_time += f64(lander.FIXED_DELTA)
			agent.observe(brain, sim_time, sensor[:], applied=applied[:])

			outcome_timer = max(outcome_timer - lander.FIXED_DELTA, 0)
			if outcome_timer == 0 {
				outcome = .Flying
				reason  = .None
			}

			if done {
				outcome       = game.outcome
				outcome_timer = OUTCOME_HOLD
				reason, reason_value, reason_limit = _verdict(sensor[:])
				attempts += 1
				if outcome == .Landed {
					landings += 1
				}
				agent.end_episode(brain, sensor[:])
				lander.reset(&game)
			}
		}

		summary := agent.stats(brain)

		rl.BeginDrawing()
		rl.ClearBackground({8, 8, 16, 255})

		_draw_world(game, applied[:], timestep.interpolation)
		_draw_status(game, human, applied[:], outcome, reason, reason_value, reason_limit, landings, attempts, summary.decisions)

		rl.EndDrawing()
	}
}

_box_draw :: proc(box: lander.Box, color: rl.Color, interpolation: f32) {
	position := math.lerp(box.position_, lander.box_position(box), interpolation)
	rotation := utility.lerp_angle(-rl.RAD2DEG * box.rotation_, -rl.RAD2DEG * lander.box_rotation(box), interpolation)
	rl.DrawRectanglePro(
		{position.x, -position.y, box.size.x, box.size.y},
		box.size / 2.0,
		rotation,
		color,
	)
}

_draw_world :: proc(game: lander.State, applied: []f32, interpolation: f32) {
	camera: rl.Camera2D
	camera.offset = {
		f32(rl.GetScreenWidth())  / 2.0,
		f32(rl.GetScreenHeight()) / 2.0,
	}
	camera.target = CAMERA_TARGET
	camera.zoom   = CAMERA_ZOOM
	rl.BeginMode2D(camera)

	_box_draw(game.ground, {60, 60, 70, 255}, interpolation)

	rl.DrawRectanglePro(
		{0, -lander.PAD_SURFACE_Y, lander.PAD_HALF_WIDTH * 2.0, PAD_HEIGHT},
		{lander.PAD_HALF_WIDTH, 0},
		0,
		rl.GREEN,
	)

	thrust := clamp(applied[lander.ACTION_AXIS_Y], 0, 1)
	if thrust > 0.01 {
		angle    := lander.lander_angle(game)
		position := math.lerp(game.lander.position_, lander.box_position(game.lander), interpolation)
		down     := [2]f32{math.sin(angle), -math.cos(angle)}
		root     := position + down * (lander.LANDER_SIZE.y / 2.0)
		tip      := root + down * (FLAME_LENGTH * thrust)
		left     := root + [2]f32{math.cos(angle), math.sin(angle)} * (FLAME_WIDTH / 2.0)
		right    := root - [2]f32{math.cos(angle), math.sin(angle)} * (FLAME_WIDTH / 2.0)

		rl.DrawTriangle(
			{right.x, -right.y},
			{left.x,  -left.y},
			{tip.x,   -tip.y},
			rl.ORANGE,
		)
	}

	_box_draw(game.lander, rl.SKYBLUE, interpolation)

	rl.EndMode2D()
}

_draw_status :: proc(game: lander.State, human: bool, applied: []f32, outcome: lander.Outcome, reason: Reason, reason_value, reason_limit: f32, landings, attempts: int, decisions: int) {
	if human {
		rl.DrawText("Human - hold LEFT MOUSE to thrust, move mouse left/right to steer (TAB to hand back)", 20, 20, 20, rl.WHITE)
	}
	else {
		rl.DrawText(rl.TextFormat("Agent, %d decisions learned (TAB to take over)", decisions), 20, 20, 20, rl.WHITE)
	}

	velocity := lander.lander_velocity(game)
	speed    := math.sqrt(velocity.x * velocity.x + velocity.y * velocity.y) / lander.V_SCALE

	speed_color := speed > lander.LAND_SPEED_NORM ? rl.RED : rl.GREEN
	angle_color := math.cos(lander.lander_angle(game)) < lander.LAND_UPRIGHT_COS ? rl.RED : rl.GREEN

	rl.DrawText(rl.TextFormat("%d FPS", rl.GetFPS()), 20, 44, 20, rl.WHITE)
	rl.DrawText(rl.TextFormat("height   %.0f", lander.lander_height(game)), 20, 72, 20, rl.WHITE)
	rl.DrawText(rl.TextFormat("speed    %.3f  (down %.3f, side %.3f)", speed, abs(velocity.y) / lander.V_SCALE, abs(velocity.x) / lander.V_SCALE), 20, 96, 20, speed_color)
	rl.DrawText(rl.TextFormat("tilt     %.2f", math.cos(lander.lander_angle(game))), 20, 120, 20, angle_color)
	rl.DrawText(rl.TextFormat("steer    %+.2f", applied[lander.ACTION_AXIS_X]), 20, 144, 20, rl.WHITE)
	rl.DrawText(rl.TextFormat("thrust    %.2f", clamp(applied[lander.ACTION_AXIS_Y], 0, 1)), 20, 168, 20, rl.WHITE)

	offset_color := abs(lander.lander_position(game).x) > lander.PAD_HALF_WIDTH ? rl.ORANGE : rl.GREEN
	rl.DrawText(rl.TextFormat("pad off  %+.0f", lander.lander_position(game).x), 20, 196, 20, offset_color)
	rl.DrawText(rl.TextFormat("landed   %d of %d", landings, attempts), 20, 220, 20, rl.WHITE)
	rl.DrawText(rl.TextFormat("time     %.1f", lander.TIME_LIMIT - game.time), 20, 244, 20, rl.WHITE)

	rl.DrawText(rl.TextFormat("to land: speed < %.3f and tilt > %.2f", lander.LAND_SPEED_NORM, lander.LAND_UPRIGHT_COS), 20, 272, 20, rl.GRAY)

	switch outcome {
	case .Flying:
	case .Timeout:
		rl.DrawText("OUT OF TIME", 20, 300, 20, rl.ORANGE)
	case .Missed:
		rl.DrawText("MISSED THE PAD - landed softly, but off target", 20, 300, 20, rl.ORANGE)
	case .Landed:
		rl.DrawText("LANDED", 20, 300, 20, rl.GREEN)
	case .Crashed:
		switch reason {
		case .None:
			rl.DrawText("CRASHED", 20, 300, 20, rl.RED)
		case .Too_Fast:
			rl.DrawText(rl.TextFormat("CRASHED - hit at speed %.3f, limit %.3f", reason_value, reason_limit), 20, 300, 20, rl.RED)
		case .Too_Tilted:
			rl.DrawText(rl.TextFormat("CRASHED - tilt %.2f at touchdown, need %.2f", reason_value, reason_limit), 20, 300, 20, rl.RED)
		case .Off_Map:
			rl.DrawText("CRASHED - flew off the map sideways", 20, 300, 20, rl.RED)
		case .Flew_Away:
			rl.DrawText("CRASHED - flew too high", 20, 300, 20, rl.RED)
		}
	}
}
