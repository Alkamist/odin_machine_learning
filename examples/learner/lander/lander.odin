package lander

import "core:math"

import b2 "vendor:box2d"

FIXED_DELTA :: 1.0 / 60.0

SENSOR_COUNT :: 8
ACTION_COUNT :: 2

SENSOR_X          :: 0
SENSOR_Y          :: 1
SENSOR_VELOCITY_X :: 2
SENSOR_VELOCITY_Y :: 3
SENSOR_ANGLE_SIN  :: 4
SENSOR_ANGLE_COS  :: 5
SENSOR_SPIN       :: 6
SENSOR_CONTACT    :: 7

ACTION_AXIS_X :: 0
ACTION_AXIS_Y :: 1

PIXELS_PER_METER :: 24

TIME_LIMIT :: 30
GRAVITY    :: f32(200)

LANDER_SIZE :: [2]f32{  60, 40}
GROUND_SIZE :: [2]f32{4000, 40}
GROUND_Y    :: f32(-500)

SPAWN_X  :: f32(300)
SPAWN_Y  :: f32(400)
SPAWN_VY :: f32(-60)

GROUND_FRICTION :: f32(0.6)
LANDER_FRICTION :: f32(0.6)

THRUST_ACCEL :: f32(450)
TARGET_SPIN  :: f32(2.0)
ANGULAR_GAIN :: f32(6.0)

X_SCALE :: f32(700)
Y_SCALE :: f32(1000)
V_SCALE :: f32(600)
W_SCALE :: f32(8)

LAND_SPEED_NORM  :: f32(0.12)
LAND_UPRIGHT_COS :: f32(0.9)
X_BOUND_NORM     :: f32(1.0)
H_MAX_NORM       :: f32(1.2)

PAD_HALF_WIDTH :: f32(110)
PAD_HALF_NORM  :: PAD_HALF_WIDTH / X_SCALE

LANDING_BONUS :: f32(40)

POS_WEIGHT  :: f32(3)
VEL_WEIGHT  :: f32(3)
TILT_WEIGHT :: f32(3)
SPIN_WEIGHT :: f32(2)

PAD_SURFACE_Y :: GROUND_Y + GROUND_SIZE.y / 2.0

CONTACT_CAPACITY :: 8

Box :: struct {
	body:  b2.BodyId,
	shape: b2.ShapeId,
	size:  [2]f32,

	position_: [2]f32,
	rotation_: f32,
}

Outcome :: enum {
	Flying,
	Landed,
	Missed,
	Crashed,
	Timeout,
}

State :: struct {
	time:    f32,
	score:   f32,
	outcome: Outcome,

	impact:          bool,
	impact_velocity: [2]f32,

	world: b2.WorldId,

	lander: Box,
	ground: Box,
}

box_make :: proc(state: State, type: b2.BodyType, position, size: [2]f32, density: f32, friction: f32) -> (box: Box) {
	box.size      = size
	box.position_ = position

	body_def         := b2.DefaultBodyDef()
	body_def.type     = type
	body_def.position = position

	box.body = b2.CreateBody(state.world, body_def)

	shape_def        := b2.DefaultShapeDef()
	shape_def.density = density

	box.shape = b2.CreatePolygonShape(box.body, shape_def, b2.MakeBox(size.x / 2.0, size.y / 2.0))

	b2.Shape_SetFriction(box.shape, friction)

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

init :: proc(state: ^State) {
	b2.SetLengthUnitsPerMeter(PIXELS_PER_METER)

	world_def          := b2.DefaultWorldDef()
	world_def.gravity.y = -GRAVITY
	state.world         = b2.CreateWorld(world_def)

	reset(state)
}

_destroy_bodies :: proc(state: ^State) {
	box_destroy(state.lander)
	box_destroy(state.ground)
}

destroy :: proc(state: ^State) {
	_destroy_bodies(state)
	b2.DestroyWorld(state.world)
}

reset :: proc(state: ^State) {
	_destroy_bodies(state)

	state.time            = 0
	state.score           = 0
	state.outcome         = .Flying
	state.impact          = false
	state.impact_velocity = {}

	state.ground = box_make(state^, .staticBody,  {0, GROUND_Y},        GROUND_SIZE, 0, friction=GROUND_FRICTION)
	state.lander = box_make(state^, .dynamicBody, {SPAWN_X, SPAWN_Y},   LANDER_SIZE, 1, friction=LANDER_FRICTION)

	b2.Body_SetLinearVelocity(state.lander.body, {0, SPAWN_VY})
}

@(require_results)
lander_position :: proc(state: State) -> [2]f32 {
	return b2.Body_GetPosition(state.lander.body)
}

@(require_results)
lander_velocity :: proc(state: State) -> [2]f32 {
	return b2.Body_GetLinearVelocity(state.lander.body)
}

@(require_results)
lander_angle :: proc(state: State) -> f32 {
	return b2.Rot_GetAngle(b2.Body_GetRotation(state.lander.body))
}

@(require_results)
lander_spin :: proc(state: State) -> f32 {
	return b2.Body_GetAngularVelocity(state.lander.body)
}

@(require_results)
lander_height :: proc(state: State) -> f32 {
	return b2.Body_GetPosition(state.lander.body).y - PAD_SURFACE_Y - LANDER_SIZE.y / 2.0
}

@(require_results)
lander_contact :: proc(state: State) -> bool {
	contacts: [CONTACT_CAPACITY]b2.ContactData
	return len(b2.Body_GetContactData(state.lander.body, contacts[:])) > 0
}

step :: proc(state: ^State, action: []f32, delta: f32) -> (done: bool) {
	state.time += delta

	steer  := clamp(action[ACTION_AXIS_X], -1, 1)
	thrust := clamp(action[ACTION_AXIS_Y],  0, 1)

	angle := lander_angle(state^)
	up    := [2]f32{-math.sin(angle), math.cos(angle)}

	mass := b2.Body_GetMass(state.lander.body)

	b2.Body_ApplyForceToCenter(state.lander.body, up * (thrust * THRUST_ACCEL * mass), true)

	target_spin := steer * TARGET_SPIN
	spin        := b2.Body_GetAngularVelocity(state.lander.body)
	inertia     := b2.Body_GetRotationalInertia(state.lander.body)

	b2.Body_ApplyTorque(state.lander.body, (target_spin - spin) * ANGULAR_GAIN * inertia, true)

	velocity_before := lander_velocity(state^)
	contact_before  := lander_contact(state^)

	box_update(&state.lander)
	box_update(&state.ground)

	b2.World_Step(state.world, delta, 4)

	state.impact = lander_contact(state^) && !contact_before
	if state.impact {
		state.impact_velocity = velocity_before
	}

	sensor: [SENSOR_COUNT]f32
	observe(state^, sensor[:])

	step_reward, _, _ := reward(sensor[:])
	state.score       += step_reward * delta

	state.outcome = classify(sensor[:])
	if state.outcome == .Flying && state.time > TIME_LIMIT {
		state.outcome = .Timeout
	}

	done = state.outcome != .Flying
	return
}

@(require_results)
classify :: proc(sensor: []f32) -> Outcome {
	offset     := sensor[SENSOR_X]
	height     := sensor[SENSOR_Y]
	velocity_x := sensor[SENSOR_VELOCITY_X]
	velocity_y := sensor[SENSOR_VELOCITY_Y]
	cos_angle  := sensor[SENSOR_ANGLE_COS]

	contact := sensor[SENSOR_CONTACT] > 0.5
	speed   := math.sqrt(velocity_x * velocity_x + velocity_y * velocity_y)

	switch {
	case contact && (speed > LAND_SPEED_NORM || cos_angle < LAND_UPRIGHT_COS):
		return .Crashed
	case contact && abs(offset) > PAD_HALF_NORM:
		return .Missed
	case contact:
		return .Landed
	case abs(offset) > X_BOUND_NORM || height > H_MAX_NORM:
		return .Crashed
	}
	return .Flying
}

observe :: proc(state: State, sensor: []f32) {
	position := lander_position(state)
	velocity := state.impact ? state.impact_velocity : lander_velocity(state)
	angle    := lander_angle(state)
	spin     := lander_spin(state)

	sensor[SENSOR_X]          = position.x / X_SCALE
	sensor[SENSOR_Y]          = lander_height(state) / Y_SCALE
	sensor[SENSOR_VELOCITY_X] = velocity.x / V_SCALE
	sensor[SENSOR_VELOCITY_Y] = velocity.y / V_SCALE
	sensor[SENSOR_ANGLE_SIN]  = math.sin(angle)
	sensor[SENSOR_ANGLE_COS]  = math.cos(angle)
	sensor[SENSOR_SPIN]       = spin / W_SCALE
	sensor[SENSOR_CONTACT]    = lander_contact(state) ? 1 : 0
}

normalize :: proc(sensor: []f32) {
	sin    := sensor[SENSOR_ANGLE_SIN]
	cos    := sensor[SENSOR_ANGLE_COS]
	length := math.sqrt(sin * sin + cos * cos)
	if length > 1e-4 {
		sensor[SENSOR_ANGLE_SIN] = sin / length
		sensor[SENSOR_ANGLE_COS] = cos / length
	}
}

@(require_results)
reward :: proc(sensor: []f32) -> (reward: f32, done: bool, failed: bool) {
	offset     := sensor[SENSOR_X]
	height     := sensor[SENSOR_Y]
	velocity_x := sensor[SENSOR_VELOCITY_X]
	velocity_y := sensor[SENSOR_VELOCITY_Y]
	cos_angle  := sensor[SENSOR_ANGLE_COS]
	spin       := sensor[SENSOR_SPIN]

	reward  = -POS_WEIGHT * (offset * offset + height * height)
	reward -=  VEL_WEIGHT * (velocity_x * velocity_x + velocity_y * velocity_y)
	reward -=  TILT_WEIGHT * (1 - cos_angle)
	reward -=  SPIN_WEIGHT * spin * spin

	switch classify(sensor) {
	case .Landed:
		reward += LANDING_BONUS
		done    = true
	case .Missed, .Crashed:
		done   = true
		failed = true
	case .Flying, .Timeout:
	}
	return
}
