package cartpole_tests

import "core:fmt"
import "core:math/rand"
import "core:testing"

import ml  "../../../"
import cpu "../../../backends/cpu"

import "../sim"
import "../../agent"
import "../../world"

THREAD_COUNT :: 4
CONTEXT_SIZE :: 1024 * 1024 * 256

Run_Result :: struct {
	best_score:   f32,
	best_upright: f32,
	episodes:     int,
	value_fit:    f32,
	fit_samples:  int,
}

@(require_results)
_run_learning :: proc(seed: u64, sim_seconds: f64) -> (result: Run_Result) {
	random_state := rand.create(seed)
	context.random_generator = rand.default_random_generator(&random_state)

	cpu.set_thread_count(THREAD_COUNT)

	ctx := cpu.context_create(CONTEXT_SIZE)
	defer cpu.context_destroy(ctx)
	ml.context_scope(ctx)

	game: sim.State
	sim.init(&game)
	defer sim.destroy(&game)

	brain := agent.make(sim.reward)
	agent.boot(brain)
	defer agent.destroy(brain)
	defer agent.shutdown(brain)

	sim_time: f64
	episode:  u64 = 1

	for sim_time < sim_seconds {
		action  := agent.act(brain)
		control := action[world.ACTION_AXIS_X]
		done    := sim.step(&game, control, sim.FIXED_DELTA)

		sim_time += f64(sim.FIXED_DELTA)
		agent.drive(brain, sim_time, sim.observe(game), action, episode)

		if done {
			upright := game.upright_time / max(game.time, 1e-9)

			if game.score > result.best_score {
				result.best_score = game.score
			}
			if upright > result.best_upright {
				result.best_upright = upright
			}
			result.episodes += 1
			sim.reset(&game)
			episode += 1
		}
	}

	result.value_fit, result.fit_samples = agent.value_fit(brain)
	return
}

@(test)
test_cartpole_learns_fast :: proc(t: ^testing.T) {
	SIM_SECONDS       :: f64(150)
	UPRIGHT_THRESHOLD :: f32(0.5)
	FIT_THRESHOLD     :: f32(0.5)
	FIT_MINIMUM       :: 64

	for seed in u64(1) ..= 2 {
		result := _run_learning(seed, SIM_SECONDS)

		testing.expectf(
			t,
			result.best_upright >= UPRIGHT_THRESHOLD,
			"seed %d: best episode held the pole inverted %.0f%% of the time, need %.0f%% (a freely spinning pole already reads ~14%%)",
			seed, result.best_upright * 100, UPRIGHT_THRESHOLD * 100,
		)

		testing.expectf(
			t,
			result.fit_samples >= FIT_MINIMUM,
			"seed %d: only %d value-fit samples, need %d (buffer chains too short to judge Q)",
			seed, result.fit_samples, FIT_MINIMUM,
		)

		testing.expectf(
			t,
			result.value_fit >= FIT_THRESHOLD,
			"seed %d: value fit %.2f below %.2f over %d samples (Q does not track observed discounted return)",
			seed, result.value_fit, FIT_THRESHOLD, result.fit_samples,
		)

		fmt.printfln(
			"seed %d | best score %.2f | best upright %.0f%% | episodes %d | value fit %.2f (%d samples)",
			seed, result.best_score, result.best_upright * 100, result.episodes, result.value_fit, result.fit_samples,
		)
	}
}
