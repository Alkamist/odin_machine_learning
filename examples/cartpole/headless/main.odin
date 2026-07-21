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
import "../world"

THREAD_COUNT :: 1

MASTERY_UPRIGHT :: f32(0.85)
DEGRADE_FLOOR   :: f32(0.70)

main :: proc() {
	minutes      := f64(10)
	seed         := u64(1)
	freeze_after := f64(0)
	freeze:       agent.Frozen_Set

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
	if len(os.args) > 3 {
		switch os.args[3] {
		case "none":
		case "models": freeze = {.Models}
		case "policy": freeze = {.Policy}
		case "both":   freeze = {.Models, .Policy}
		case:
			fmt.eprintfln("unknown freeze mode %q, expected none, models, policy, or both", os.args[3])
			os.exit(1)
		}
	}
	if len(os.args) > 4 {
		if value, ok := strconv.parse_f64(os.args[4]); ok {
			freeze_after = value
		}
	}

	random_state := rand.create(seed)
	context.random_generator = rand.default_random_generator(&random_state)

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

	best_score:    f32
	best_upright:  f32
	mastered:      bool
	mastered_at:   f64
	degraded:      bool
	degraded_at:   f64
	scores:        [dynamic]f32
	uprights:      [dynamic]f32
	defer delete(scores)
	defer delete(uprights)

	start_tick := time.tick_now()

	for sim_time < minutes * 60 {
		action  := agent.act(brain)
		control := action[world.ACTION_AXIS_X]
		done    := sim.step(&game, control, sim.FIXED_DELTA)

		sim_time += f64(sim.FIXED_DELTA)
		agent.drive(brain, sim_time, sim.observe(game), action, episode)

		if done {
			duration       := game.time
			score          := game.score
			wall_elapsed   := time.duration_seconds(time.tick_diff(start_tick, time.tick_now()))
			value_fit, _   := agent.value_fit(brain)

			upright := game.upright_time / max(game.time, 1e-9)

			append(&scores, score)
			append(&uprights, upright)
			if score > best_score {
				best_score = score
			}
			if upright > best_upright {
				best_upright = upright
			}

			if !mastered && upright >= MASTERY_UPRIGHT {
				mastered    = true
				mastered_at = sim_time
			}
			if mastered && sim_time >= freeze_after {
				brain.frozen = freeze
			}
			if mastered && !degraded && upright < DEGRADE_FLOOR {
				degraded    = true
				degraded_at = sim_time
			}

			fmt.printfln(
				"episode %d | score %.2f | upright %.0f%% | sim %.1fs | decisions %d | policy match %.0f%% | value fit %.2f | wall %.1fs | speedup %.1fx",
				episode,
				score,
				upright * 100,
				duration,
				agent.decisions(brain),
				agent.policy_match(brain) * 100,
				value_fit,
				wall_elapsed,
				sim_time / max(wall_elapsed, 1e-9),
			)

			sim.reset(&game)
			episode += 1
		}
	}

	wall_elapsed := time.duration_seconds(time.tick_diff(start_tick, time.tick_now()))

	recent_count := min(len(scores), 10)
	recent_mean:    f32
	recent_upright: f32
	for i in len(scores) - recent_count ..< len(scores) {
		recent_mean    += scores[i]
		recent_upright += uprights[i]
	}
	if recent_count > 0 {
		recent_mean    /= f32(recent_count)
		recent_upright /= f32(recent_count)
	}

	fmt.printfln("--- summary ---")
	fmt.printfln("episodes         %d", len(scores))
	fmt.printfln("best score       %.2f", best_score)
	fmt.printfln("best upright     %.0f%%", best_upright * 100)
	fmt.printfln("mean last 10     %.2f", recent_mean)
	fmt.printfln("upright last 10  %.0f%%", recent_upright * 100)

	final_fit, fit_samples := agent.value_fit(brain)

	fmt.printfln("total decisions  %d", agent.decisions(brain))
	fmt.printfln("final match      %.0f%%", agent.policy_match(brain) * 100)
	fmt.printfln("value fit        %.2f (%d samples)", final_fit, fit_samples)
	fmt.printfln("overall speedup  %.1fx", sim_time / max(wall_elapsed, 1e-9))

	fmt.printfln(
		"result seed=%d freeze=%v freeze_after=%.0f mastered=%v mastered_at=%.0f degraded=%v degraded_at=%.0f upright_last10=%.4f episodes=%d",
		seed, freeze, freeze_after, mastered, mastered_at, degraded, degraded_at, recent_upright, len(scores),
	)
}
