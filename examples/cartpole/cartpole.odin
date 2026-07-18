package main

import "core:math"

import rl "vendor:raylib"
import b2 "vendor:box2d"

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

human_action :: proc(previous: Action) -> (res: Action) {
	if !rl.IsKeyDown(.A) && !rl.IsKeyDown(.D) {
		return .None
	}

	res = previous

	if rl.IsKeyPressed(.A) {
		res = .Left
	}
	if rl.IsKeyPressed(.D) {
		res = .Right
	}

	if rl.IsKeyReleased(.A) {
		if rl.IsKeyDown(.D) {
			res = .Right
		}
		else {
			res = .None
		}
	}
	if rl.IsKeyReleased(.D) {
		if rl.IsKeyDown(.A) {
			res = .Left
		}
		else {
			res = .None
		}
	}

	return
}

Category :: enum u64 {
	Normal,
	Pole,
	Mouse,
}

Category_Set :: bit_set[Category; u64]

Box :: struct {
	body:  b2.BodyId,
	shape: b2.ShapeId,
	size:  [2]f32,

	position_: [2]f32, // previous position for visual interpolation
	rotation_: f32,    // previous rotation for visual interpolation
}

box_make :: proc(state: State, type: b2.BodyType, position, size: [2]f32, density: f32, category: Category_Set = {.Normal}, mask: Category_Set = {.Normal}) -> (box: Box) {
	box.size      = size
	box.position_ = position

	body_def         := b2.DefaultBodyDef()
	body_def.type     = type
	body_def.position = position

	box.body = b2.CreateBody(state.world, body_def)

	shape_def                    := b2.DefaultShapeDef()
	shape_def.density             = density
	shape_def.filter.categoryBits = transmute(u64)category
	shape_def.filter.maskBits     = transmute(u64)mask

	box.shape = b2.CreatePolygonShape(box.body, shape_def, b2.MakeBox(size.x / 2.0, size.y / 2.0))

	b2.Shape_SetFriction(box.shape, 0)

	return
}

box_destroy :: proc(box: Box) {
	if box.shape != {} {
		b2.DestroyShape(box.shape, true)
	}
	if box.body  != {} {
		b2.DestroyBody(box.body)
	}
}

box_update :: proc(box: ^Box) {
	box.position_ = b2.Body_GetPosition(box.body)
	box.rotation_ = -rl.RAD2DEG * b2.Rot_GetAngle(b2.Body_GetRotation(box.body))
}

box_draw :: proc(box: Box, color: rl.Color, interpolation: f32) {
	position := math.lerp(box.position_, b2.Body_GetPosition(box.body), interpolation)
	rotation := lerp_angle(box.rotation_, -rl.RAD2DEG * b2.Rot_GetAngle(b2.Body_GetRotation(box.body)), interpolation)
	rl.DrawRectanglePro(
		{position.x, -position.y, box.size.x, box.size.y},
		box.size / 2.0,
		rotation,
		color,
	)
}

TIME_LIMIT :: 30
GRAVITY    :: 2000
CART_LIMIT :: 500
CART_SPEED :: 500
CART_SIZE  :: [2]f32{100,   50}
POLE_SIZE  :: [2]f32{  8,  300}
WALL_SIZE  :: [2]f32{ 10, 1000}

Action :: enum {
	None,
	Left,
	Right,
}

State :: struct {
	high_score: f32,

	time:  f32,
	score: f32,

	world: b2.WorldId,

	cart:            Box,
	pole:            Box,
	left_wall:       Box,
	right_wall:      Box,
	anchor_body:     b2.BodyId,
	revolute_joint:  b2.JointId,
	prismatic_joint: b2.JointId,

	mouse_body:   b2.BodyId,
	mouse_shape:  b2.ShapeId,
	mouse_active: bool,
	mouse_target: [2]f32,
	mouse_position_: [2]f32, // previous position for visual interpolation
}

MOUSE_RADIUS :: 20.0

mouse_destroy :: proc(state: State) {
	if state.mouse_shape != {} {
		b2.DestroyShape(state.mouse_shape, true)
	}
	if state.mouse_body != {} {
		b2.DestroyBody(state.mouse_body)
	}
}

mouse_begin :: proc(state: ^State, world: [2]f32) {
	state.mouse_target = world
	state.mouse_active = true

	b2.Body_SetTransform(state.mouse_body, world, b2.MakeRot(0))
	b2.Body_SetLinearVelocity(state.mouse_body, {0, 0})
	b2.Body_Enable(state.mouse_body)

	state.mouse_position_ = world
}

mouse_end :: proc(state: ^State) {
	if !state.mouse_active {
		return
	}
	state.mouse_active = false
	b2.Body_Disable(state.mouse_body)
}

mouse_apply :: proc(state: ^State, delta: f32) {
	if !state.mouse_active {
		return
	}

	position := b2.Body_GetPosition(state.mouse_body)
	b2.Body_SetLinearVelocity(state.mouse_body, (state.mouse_target - position) / delta)
}

init :: proc(state: ^State) {
	b2.SetLengthUnitsPerMeter(PIXELS_PER_METER)

	world_def          := b2.DefaultWorldDef()
	world_def.gravity.y = -GRAVITY
	state.world         = b2.CreateWorld(world_def)

	reset(state)
}

destroy :: proc(state: ^State) {
	if state.revolute_joint  != {} {
		b2.DestroyJoint(state.revolute_joint)
	}
	if state.prismatic_joint != {} {
		b2.DestroyJoint(state.prismatic_joint)
	}
	if state.anchor_body     != {} {
		b2.DestroyBody(state.anchor_body)
	}
	mouse_destroy(state^)
	box_destroy(state.pole)
	box_destroy(state.cart)
	box_destroy(state.left_wall)
	box_destroy(state.right_wall)
	b2.DestroyWorld(state.world)
}

reset :: proc(state: ^State) {
	if state.revolute_joint  != {} {
		b2.DestroyJoint(state.revolute_joint)
	}
	if state.prismatic_joint != {} {
		b2.DestroyJoint(state.prismatic_joint)
	}
	if state.anchor_body     != {} {
		b2.DestroyBody(state.anchor_body)
	}
	mouse_destroy(state^)
	box_destroy(state.cart)
	box_destroy(state.pole)
	box_destroy(state.left_wall)
	box_destroy(state.right_wall)

	state.time     = 0
	state.score    = 0

	state.mouse_active = false

	anchor_def         := b2.DefaultBodyDef()
	anchor_def.type     = .staticBody
	anchor_def.position = {0, 0}
	state.anchor_body   = b2.CreateBody(state.world, anchor_def)

	state.cart = box_make(state^, .dynamicBody, {0, 0}, CART_SIZE, 5)
	state.pole = box_make(state^, .dynamicBody, {0, -POLE_SIZE.y * 0.5}, POLE_SIZE, 2, category={.Pole}, mask={.Mouse})

	mouse_def         := b2.DefaultBodyDef()
	mouse_def.type     = .kinematicBody
	mouse_def.position = {0, 0}
	state.mouse_body   = b2.CreateBody(state.world, mouse_def)

	mouse_shape_def                    := b2.DefaultShapeDef()
	mouse_shape_def.density             = 1
	mouse_shape_def.filter.categoryBits = transmute(u64)Category_Set{.Mouse}
	mouse_shape_def.filter.maskBits     = transmute(u64)Category_Set{.Pole}

	state.mouse_shape = b2.CreateCircleShape(state.mouse_body, mouse_shape_def, b2.Circle{center = {0, 0}, radius = MOUSE_RADIUS})

	b2.Shape_SetFriction(state.mouse_shape, 0)
	b2.Body_Disable(state.mouse_body)

	state.left_wall  = box_make(state^, .staticBody, {-CART_LIMIT, 0}, WALL_SIZE, 0)
	state.right_wall = box_make(state^, .staticBody, { CART_LIMIT, 0}, WALL_SIZE, 0)

	prismatic_def                 := b2.DefaultPrismaticJointDef()
	prismatic_def.bodyIdA          = state.anchor_body
	prismatic_def.bodyIdB          = state.cart.body
	prismatic_def.localAnchorA     = {0, 0}
	prismatic_def.localAnchorB     = {0, 0}
	prismatic_def.localAxisA       = {1, 0}
	state.prismatic_joint          = b2.CreatePrismaticJoint(state.world, prismatic_def)

	revolute_def             := b2.DefaultRevoluteJointDef()
	revolute_def.bodyIdA      = state.cart.body
	revolute_def.bodyIdB      = state.pole.body
	revolute_def.localAnchorA = {0, 0}
	revolute_def.localAnchorB = {0, POLE_SIZE.y / 2.0}
	state.revolute_joint      = b2.CreateRevoluteJoint(state.world, revolute_def)
}

@(require_results)
cart_position :: proc(state: State) -> f32 {
	return b2.Body_GetPosition(state.cart.body).x
}

@(require_results)
cart_velocity :: proc(state: State) -> f32 {
	return b2.Body_GetLinearVelocity(state.cart.body).x
}

@(require_results)
pole_angle :: proc(state: State) -> f32 {
	return b2.Rot_GetAngle(b2.Body_GetRotation(state.pole.body))
}

@(require_results)
pole_spin :: proc(state: State) -> f32 {
	return b2.Body_GetAngularVelocity(state.pole.body)
}

step :: proc(state: ^State, action: Action, delta: f32) -> (done: bool) {
	state.time += delta

	mouse_apply(state, delta)

	target_speed: f32
	switch action {
	case .None:
	case .Left:  target_speed = -CART_SPEED
	case .Right: target_speed =  CART_SPEED
	}

	speed_diff := target_speed - b2.Body_GetLinearVelocity(state.cart.body).x
	force      := speed_diff * 200000.0

	b2.Body_ApplyForceToCenter(state.cart.body, {force, 0}, true)

	state.mouse_position_ = b2.Body_GetPosition(state.mouse_body)

	box_update(&state.left_wall)
	box_update(&state.right_wall)

	box_update(&state.cart)
	box_update(&state.pole)

	b2.World_Step(state.world, delta, 4)

	pole_angle := b2.Rot_GetAngle(b2.Body_GetRotation(state.pole.body))

	state.score += abs(pole_angle) * delta

	contact_events := b2.World_GetContactEvents(state.world)
	wall_hit       := false

	for i in 0 ..< contact_events.beginCount {
		begin_event := contact_events.beginEvents[i]

		shape_a := begin_event.shapeIdA
		shape_b := begin_event.shapeIdB

		if (shape_a == state.cart.shape && (shape_b == state.left_wall.shape || shape_b == state.right_wall.shape)) ||
		   (shape_b == state.cart.shape && (shape_a == state.left_wall.shape || shape_a == state.right_wall.shape)) {
			wall_hit = true
		}
	}

	if wall_hit || state.time > TIME_LIMIT {
		if state.score > state.high_score {
			state.high_score = state.score
		}
		done = true
	}

	return
}

draw :: proc(state: State, interpolation: f32) {
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
		position := math.lerp(state.mouse_position_, b2.Body_GetPosition(state.mouse_body), interpolation)
		rl.DrawCircleV({position.x, -position.y}, MOUSE_RADIUS, rl.YELLOW)
	}

	position := math.lerp(state.cart.position_, b2.Body_GetPosition(state.cart.body), interpolation)
	draw_text_centered(rl.TextFormat("%.2f", state.score), 10, position.x, position.y + 50, rl.WHITE)

	rl.EndMode2D()

	rl.DrawText(rl.TextFormat("High Score: %.2f", state.high_score),        20, 68, 20, rl.WHITE)
	rl.DrawText(rl.TextFormat("Time: %.2f",       TIME_LIMIT - state.time), 20, 92, 20, rl.WHITE)
}

draw_status :: proc(human: bool, decisions: int, agreement: f32) {
	if human {
		rl.DrawText("Human (TAB to hand back to the agent) - A/D to move", 20, 20, 20, rl.WHITE)
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
