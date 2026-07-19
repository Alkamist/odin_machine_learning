package main

import "core:fmt"
import "core:math/rand"
import "core:os"
import "core:strconv"
import "core:time"

import ml  "../../../"
import cpu "../../../backends/cpu"

import "../sim"
import "../agent"

THREAD_COUNT :: 4

#assert(len(sim.Action) == agent.ACTION_COUNT)

main :: proc() {
	minutes := f64(10)
	seed    := u64(1)

	if len(os.args) > 1 {
		if value, ok := strconv.parse_f64(os.args[1]); ok {
			minutes = value
		}
	}
	if len(os.args) > 2 {
		if value, ok := strconv.parse_u64(os.args[2]); ok {
			seed = value
		}
	}

	rand.reset(seed)

	cpu.set_thread_count(THREAD_COUNT)

	ctx := cpu.context_create(1024 * 1024 * 256)
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

	best_score:   f32
	scores:       [dynamic]f32
	defer delete(scores)

	start_tick := time.tick_now()

	for sim_time < minutes * 60 {
		action := sim.Action(agent.act(brain))
		done   := sim.step(&game, action, sim.FIXED_DELTA)

		sim_time += f64(sim.FIXED_DELTA)
		agent.drive(brain, sim_time, sim.observe(game), int(action), episode)

		if done {
			duration    := game.time
			score       := game.score
			wall_elapsed := time.duration_seconds(time.tick_diff(start_tick, time.tick_now()))

			append(&scores, score)
			if score > best_score {
				best_score = score
			}

			fmt.printfln(
				"episode %d | score %.2f | sim %.1fs | decisions %d | agreement %.0f%% | wall %.1fs | speedup %.1fx",
				episode,
				score,
				duration,
				agent.decisions(brain),
				agent.agreement(brain) * 100,
				wall_elapsed,
				sim_time / max(wall_elapsed, 1e-9),
			)

			sim.reset(&game)
			episode += 1
		}
	}

	wall_elapsed := time.duration_seconds(time.tick_diff(start_tick, time.tick_now()))

	recent_count := min(len(scores), 10)
	recent_mean:  f32
	for i in len(scores) - recent_count ..< len(scores) {
		recent_mean += scores[i]
	}
	if recent_count > 0 {
		recent_mean /= f32(recent_count)
	}

	fmt.printfln("--- summary ---")
	fmt.printfln("episodes         %d", len(scores))
	fmt.printfln("best score       %.2f", best_score)
	fmt.printfln("mean last 10     %.2f", recent_mean)
	fmt.printfln("total decisions  %d", agent.decisions(brain))
	fmt.printfln("final agreement  %.0f%%", agent.agreement(brain) * 100)
	fmt.printfln("overall speedup  %.1fx", sim_time / max(wall_elapsed, 1e-9))
}
