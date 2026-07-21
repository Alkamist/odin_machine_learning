package cartpole_tests

import "core:fmt"
import "core:math/rand"
import "core:testing"

import ml  "../../../"
import cpu "../../../backends/cpu"

import "../sim"
import "../agent"

THREAD_COUNT :: 4
CONTEXT_SIZE :: 1024 * 1024 * 256

@(require_results)
_run_learning :: proc(seed: u64, sim_seconds: f64) -> (best_score: f32, episodes: int) {
	rand.reset(seed)
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
		control := action[0]
		done    := sim.step(&game, control, sim.FIXED_DELTA)

		sim_time += f64(sim.FIXED_DELTA)
		agent.drive(brain, sim_time, sim.observe(game), action, episode)

		if done {
			if game.score > best_score {
				best_score = game.score
			}
			episodes += 1
			sim.reset(&game)
			episode += 1
		}
	}

	return
}

@(test)
test_cartpole_learns_fast :: proc(t: ^testing.T) {
	SIM_SECONDS :: f64(90)
	THRESHOLD   :: f32(45)

	for seed in u64(1) ..= 2 {
		best_score, episodes := _run_learning(seed, SIM_SECONDS)

		testing.expectf(
			t,
			best_score >= THRESHOLD,
			"seed %d: best score %.2f below %.0f after %d episodes (agent is not learning swing-up)",
			seed, best_score, THRESHOLD, episodes,
		)

		fmt.printfln("seed %d | best score %.2f | episodes %d", seed, best_score, episodes)
	}
}
