package world

SENSOR_SIZE :: 8

SENSOR_X          :: 0
SENSOR_Y          :: 1
SENSOR_VELOCITY_X :: 2
SENSOR_VELOCITY_Y :: 3
SENSOR_ANGLE_SIN  :: 4
SENSOR_ANGLE_COS  :: 5
SENSOR_SPIN       :: 6
SENSOR_CONTACT    :: 7

BINARY_COUNT :: 0
ANALOG_COUNT :: 2
ACTION_DIM   :: BINARY_COUNT + ANALOG_COUNT

ACTION_AXIS_X :: 0
ACTION_AXIS_Y :: 1

Sensor :: [SENSOR_SIZE]f32
Action :: [ACTION_DIM]f32

Angle_Pair :: struct {
	sin: int,
	cos: int,
}

ANGLE_PAIRS :: []Angle_Pair{{sin=SENSOR_ANGLE_SIN, cos=SENSOR_ANGLE_COS}}

Reward_Proc :: proc(sensor: Sensor) -> (reward: f32, dead: bool)
