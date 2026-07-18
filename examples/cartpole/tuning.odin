package main

import "core:fmt"
import "core:os"
import "core:strconv"
import "core:strings"

// Temporary sweep harness: lets the headless benchmark vary the agent's
// hyperparameters from the command line so they can be compared over a set of
// seeds without a rebuild per variant. The defaults here are the real values.
Tuning :: struct {
	plan_samples:    int,
	plan_elites:     int,
	plan_iters:      int,
	plan_horizon:    int,
	train_steps:     int,
	learning_rate:   f32,
	warmup:          int,
	upright_weight:  f32,
	energy_weight:   f32,
	center_weight:   f32,
	barrier_onset:   f32,
	barrier_weight:  f32,
	spin_weight:     f32,
	discount:        f32,
}

tuning := Tuning {
	plan_samples   = 64,
	plan_elites    = PLAN_ELITES,
	plan_iters     = PLAN_ITERS,
	plan_horizon   = 20,
	train_steps    = TRAIN_STEPS,
	learning_rate  = LEARNING_RATE,
	warmup         = WARMUP_DECISIONS,
	upright_weight = 3,
	energy_weight  = 3,
	center_weight  = 3,
	barrier_onset  = 0.5,
	barrier_weight = 20,
	spin_weight    = 2,
	discount       = 0.98,
}

tuning_parse :: proc(arguments: []string) {
	for argument in arguments {
		if !strings.has_prefix(argument, "-") || !strings.contains(argument, "=") {
			continue
		}

		parts := strings.split(argument[1:], "=", context.temp_allocator)
		key   := parts[0]
		value := parts[1]

		number, ok := strconv.parse_f64(value)
		if !ok {
			fmt.eprintln("bad tuning value:", argument)
			os.exit(1)
		}

		switch key {
		case "plan_samples":   tuning.plan_samples   = int(number)
		case "plan_elites":    tuning.plan_elites    = int(number)
		case "plan_iters":     tuning.plan_iters     = int(number)
		case "plan_horizon":   tuning.plan_horizon   = int(number)
		case "train_steps":    tuning.train_steps    = int(number)
		case "learning_rate":  tuning.learning_rate  = f32(number)
		case "warmup":         tuning.warmup         = int(number)
		case "upright_weight": tuning.upright_weight = f32(number)
		case "energy_weight":  tuning.energy_weight  = f32(number)
		case "center_weight":  tuning.center_weight  = f32(number)
		case "barrier_onset":  tuning.barrier_onset  = f32(number)
		case "barrier_weight": tuning.barrier_weight = f32(number)
		case "spin_weight":    tuning.spin_weight    = f32(number)
		case "discount":       tuning.discount       = f32(number)
		case:
			fmt.eprintln("unknown tuning key:", key)
			os.exit(1)
		}
	}

	assert(tuning.plan_samples <= PLAN_SAMPLES, "plan_samples exceeds its compiled maximum")
	assert(tuning.plan_horizon <= PLAN_HORIZON, "plan_horizon exceeds its compiled maximum")
	assert(tuning.plan_elites  <= tuning.plan_samples)
}
