package main

import "core:fmt"
import "core:math/rand"
import "core:os"
import "core:strconv"
import "core:time"

import          "../../agent"
import cartpole ".."

THREAD_COUNT :: 1

MASTERY_UPRIGHT :: f32(0.85)
DEGRADE_FLOOR   :: f32(0.70)

main :: proc() {
	minutes      := f64(10)
	seed         := u64(1)
	freeze_after := f64(0)
	frozen:       agent.Frozen_Set

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
		case "models": frozen = {.Models}
		case "policy": frozen = {.Policy}
		case "both":   frozen = {.Models, .Policy}
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

	game: cartpole.State
	cartpole.init(&game)
	defer cartpole.destroy(&game)

	brain := agent.create(cartpole.SENSOR_COUNT, cartpole.ACTION_COUNT, cartpole.reward, normalize=cartpole.normalize, compute_threads=THREAD_COUNT)
	defer agent.destroy(brain)

	sensor: [cartpole.SENSOR_COUNT]f32
	action: [cartpole.ACTION_COUNT]f32

	sim_time: f64
	episodes: int

	best_score:   f32
	best_upright: f32
	mastered:     bool
	mastered_at:  f64
	degraded:     bool
	degraded_at:  f64
	scores:       [dynamic]f32
	uprights:     [dynamic]f32
	defer delete(scores)
	defer delete(uprights)

	start_tick := time.tick_now()

	for sim_time < minutes * 60 {
		agent.act(brain, action[:])
		done := cartpole.step(&game, action[:], cartpole.FIXED_DELTA)

		sim_time += f64(cartpole.FIXED_DELTA)
		cartpole.observe(game, sensor[:])
		agent.observe(brain, sim_time, sensor[:], applied=action[:])
		agent.catch_up(brain, sim_time)

		if done {
			duration     := game.time
			score        := game.score
			wall_elapsed := time.duration_seconds(time.tick_diff(start_tick, time.tick_now()))
			summary      := agent.stats(brain)

			upright := game.upright_time / max(game.time, 1e-9)

			episodes += 1
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
				agent.freeze(brain, frozen)
			}
			if mastered && !degraded && upright < DEGRADE_FLOOR {
				degraded    = true
				degraded_at = sim_time
			}

			fmt.printfln(
				"episode %d | score %.2f | upright %.0f%% | sim %.1fs | decisions %d | policy match %.0f%% | value fit %.2f | wall %.1fs | speedup %.1fx",
				episodes,
				score,
				upright * 100,
				duration,
				summary.decisions,
				summary.policy_match * 100,
				summary.value_fit,
				wall_elapsed,
				sim_time / max(wall_elapsed, 1e-9),
			)

			cartpole.reset(&game)
			agent.end_episode(brain)
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

	summary := agent.stats(brain)

	fmt.printfln("--- summary ---")
	fmt.printfln("episodes         %d", len(scores))
	fmt.printfln("best score       %.2f", best_score)
	fmt.printfln("best upright     %.0f%%", best_upright * 100)
	fmt.printfln("mean last 10     %.2f", recent_mean)
	fmt.printfln("upright last 10  %.0f%%", recent_upright * 100)
	fmt.printfln("total decisions  %d", summary.decisions)
	fmt.printfln("final match      %.0f%%", summary.policy_match * 100)
	fmt.printfln("value fit        %.2f (%d samples)", summary.value_fit, summary.fit_samples)
	fmt.printfln("overall speedup  %.1fx", sim_time / max(wall_elapsed, 1e-9))

	fmt.printfln(
		"result seed=%d freeze=%v freeze_after=%.0f mastered=%v mastered_at=%.0f degraded=%v degraded_at=%.0f upright_last10=%.4f episodes=%d",
		seed, frozen, freeze_after, mastered, mastered_at, degraded, degraded_at, recent_upright, len(scores),
	)
}
