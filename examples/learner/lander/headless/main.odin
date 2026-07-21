package main

import "core:fmt"
import "core:math"
import "core:math/rand"
import "core:os"
import "core:strconv"
import "core:time"

import ml  "../../../../"
import cpu "../../../../backends/cpu"

import        "../../agent"
import lander ".."

THREAD_COUNT :: 1

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

	random_state := rand.create(seed)
	context.random_generator = rand.default_random_generator(&random_state)

	cpu.set_thread_count(THREAD_COUNT)

	ctx := cpu.context_create(1024 * 1024 * 256)
	defer cpu.context_destroy(ctx)

	ml.context_scope(ctx)

	game: lander.State
	lander.init(&game)
	defer lander.destroy(&game)

	brain := agent.make(lander.reward)
	agent.boot(brain)
	defer agent.destroy(brain)
	defer agent.shutdown(brain)

	sim_time: f64
	episode:  u64 = 1

	best_rate := math.inf_f32(-1)
	counts:      [lander.Outcome]int
	scores:      [dynamic]f32
	successes:   [dynamic]f32
	defer delete(scores)
	defer delete(successes)

	start_tick := time.tick_now()

	for sim_time < minutes * 60 {
		action := agent.act(brain)
		done   := lander.step(&game, action, lander.FIXED_DELTA)

		sim_time += f64(lander.FIXED_DELTA)
		agent.drive(brain, sim_time, lander.observe(game), action, episode)

		if done {
			duration     := game.time
			rate         := game.score / max(game.time, 1e-9)
			wall_elapsed := time.duration_seconds(time.tick_diff(start_tick, time.tick_now()))

			outcome := game.outcome

			counts[outcome] += 1
			append(&scores, rate)
			append(&successes, outcome == .Landed ? f32(1) : 0)
			if rate > best_rate {
				best_rate = rate
			}

			fmt.printfln(
				"episode %d | %-7v | reward/s %.2f | rest x %+.0f | sim %.1fs | decisions %d | policy match %.0f%% | wall %.1fs | speedup %.1fx",
				episode,
				outcome,
				rate,
				lander.lander_position(game).x,
				duration,
				agent.decisions(brain),
				agent.policy_match(brain) * 100,
				wall_elapsed,
				sim_time / max(wall_elapsed, 1e-9),
			)

			lander.reset(&game)
			episode += 1
		}
	}

	wall_elapsed := time.duration_seconds(time.tick_diff(start_tick, time.tick_now()))

	recent_count := min(len(scores), 10)
	recent_mean:    f32
	recent_success: f32
	for i in len(scores) - recent_count ..< len(scores) {
		recent_mean    += scores[i]
		recent_success += successes[i]
	}
	if recent_count > 0 {
		recent_mean    /= f32(recent_count)
		recent_success /= f32(recent_count)
	}

	fmt.printfln("--- summary ---")
	fmt.printfln("episodes        %d", len(scores))
	fmt.printfln("best reward/s   %.2f", best_rate)

	fmt.printfln("mean reward/s   %.2f", recent_mean)
	fmt.printfln("landed          %d of %d (%.0f%%)", counts[.Landed], len(scores), 100 * f32(counts[.Landed]) / max(f32(len(scores)), 1))
	fmt.printfln("success last 10 %.0f%%", recent_success * 100)
	fmt.printfln("missed pad      %d", counts[.Missed])
	fmt.printfln("crashed         %d", counts[.Crashed])
	fmt.printfln("timed out       %d", counts[.Timeout])
	fmt.printfln("total decisions %d", agent.decisions(brain))
	fmt.printfln("final match     %.0f%%", agent.policy_match(brain) * 100)
	fmt.printfln("overall speedup %.1fx", sim_time / max(wall_elapsed, 1e-9))
}
