package sim

import "core:math"

import b2 "vendor:box2d"

FIXED_DELTA :: 1.0 / 60.0

PIXELS_PER_METER :: 24

TIME_LIMIT :: 30
GRAVITY    :: 2000
CART_LIMIT :: 500
CART_SPEED :: 2000
CART_SIZE  :: [2]f32{100,   50}
POLE_SIZE  :: [2]f32{  8,  300}
WALL_SIZE  :: [2]f32{ 10, 1000}

MOUSE_RADIUS :: 20.0

OBS_SIZE :: 5

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

Action :: enum {
	None,
	Left,
	Right,
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

	position_: [2]f32,
	rotation_: f32,
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
	mouse_position_: [2]f32,
}

Observation :: [OBS_SIZE]f32

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
	box.rotation_ = b2.Rot_GetAngle(b2.Body_GetRotation(box.body))
}

@(require_results)
box_position :: proc(box: Box) -> [2]f32 {
	return b2.Body_GetPosition(box.body)
}

@(require_results)
box_rotation :: proc(box: Box) -> f32 {
	return b2.Rot_GetAngle(b2.Body_GetRotation(box.body))
}

@(require_results)
mouse_position :: proc(state: State) -> [2]f32 {
	return b2.Body_GetPosition(state.mouse_body)
}

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

_destroy_bodies :: proc(state: ^State) {
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
}

destroy :: proc(state: ^State) {
	_destroy_bodies(state)
	b2.DestroyWorld(state.world)
}

reset :: proc(state: ^State) {
	_destroy_bodies(state)

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

@(require_results)
action_control :: proc(action: Action) -> f32 {
	switch action {
	case .None:  return 0
	case .Left:  return -1
	case .Right: return 1
	}
	return 0
}

@(require_results)
control_action :: proc(control: f32) -> Action {
	DEADZONE :: f32(0.2)
	if control >  DEADZONE {
		return .Right
	}
	if control < -DEADZONE {
		return .Left
	}
	return .None
}

step :: proc(state: ^State, control: f32, delta: f32) -> (done: bool) {
	state.time += delta

	mouse_apply(state, delta)

	target_speed := clamp(control, -1, 1) * CART_SPEED

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

@(require_results)
observe :: proc(state: State) -> (sensor: Observation) {
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
reward :: proc(sensor: Observation) -> (reward: f32, dead: bool) {
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
