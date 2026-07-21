package cartpole_tests

import "core:fmt"
import "core:math/rand"
import "core:sync"
import "core:testing"
import "core:thread"

import          "../../agent"
import cartpole ".."

SEEDS            :: 8
REQUIRED_PASSES  :: 7
ALLOWED_FAILURES :: SEEDS - REQUIRED_PASSES

MASTERY_UPRIGHT :: f32(0.85)
DEGRADE_FLOOR   :: f32(0.70)
DEGRADE_STREAK  :: 3

LEARN_DEADLINE :: f64(150)
HOLD_DEADLINE  :: f64(15 * 60)

Verdict :: enum {
	Stable,
	Slow,
	Collapsed,
	Aborted,
}

Sweep :: struct {
	failures: int,
	abort:    bool,
}

Seed_Run :: struct {
	seed:  u64,
	sweep: ^Sweep,

	verdict:        Verdict,
	peak_upright:   f32,
	last_upright:   f32,
	mastered_at:    f64,
	stopped_at:     f64,
	episodes:       int,
	low_episodes:   int,
	max_low_streak: int,
	value_fit:      f32,
	fit_samples:    int,
}

_fail :: proc(run: ^Seed_Run, verdict: Verdict) {
	run.verdict = verdict
	if sync.atomic_add(&run.sweep.failures, 1) + 1 > ALLOWED_FAILURES {
		sync.atomic_store(&run.sweep.abort, true)
	}
}

_run_seed :: proc(run: ^Seed_Run) {
	random_state := rand.create(run.seed)
	context.random_generator = rand.default_random_generator(&random_state)

	game: cartpole.State
	cartpole.init(&game)
	defer cartpole.destroy(&game)

	brain := agent.create(cartpole.SENSOR_COUNT, cartpole.ACTION_COUNT, cartpole.reward, normalize=cartpole.normalize)
	defer agent.destroy(brain)

	sensor: [cartpole.SENSOR_COUNT]f32
	action: [cartpole.ACTION_COUNT]f32

	sim_time:   f64
	mastered:   bool
	low_streak: int

	for {
		agent.act(brain, action[:])
		done := cartpole.step(&game, action[:], cartpole.FIXED_DELTA)

		sim_time += f64(cartpole.FIXED_DELTA)
		cartpole.observe(game, sensor[:])
		agent.observe(brain, sim_time, sensor[:], applied=action[:])
		agent.catch_up(brain, sim_time)

		if !done {
			continue
		}

		upright := game.upright_time / max(game.time, 1e-9)

		run.episodes    += 1
		run.last_upright = upright
		run.stopped_at   = sim_time
		if upright > run.peak_upright {
			run.peak_upright = upright
		}
		if !mastered && upright >= MASTERY_UPRIGHT {
			mastered        = true
			run.mastered_at = sim_time
		}

		if mastered {
			if upright < DEGRADE_FLOOR {
				low_streak        += 1
				run.low_episodes  += 1
				if low_streak > run.max_low_streak {
					run.max_low_streak = low_streak
				}
				if low_streak >= DEGRADE_STREAK {
					_fail(run, .Collapsed)
					break
				}
			}
			else {
				low_streak = 0
			}
		}
		if !mastered && sim_time >= LEARN_DEADLINE {
			_fail(run, .Slow)
			break
		}
		if sim_time >= HOLD_DEADLINE {
			run.verdict = .Stable
			break
		}
		if sync.atomic_load(&run.sweep.abort) {
			run.verdict = .Aborted
			break
		}

		cartpole.reset(&game)
		agent.end_episode(brain)
	}

	summary        := agent.stats(brain)
	run.value_fit   = summary.value_fit
	run.fit_samples = summary.fit_samples
}

@(test)
test_cartpole_learns_fast_and_holds :: proc(t: ^testing.T) {
	sweep:   Sweep
	runs:    [SEEDS]Seed_Run
	workers: [SEEDS]^thread.Thread

	for &run, index in runs {
		run = {seed=u64(index + 1), sweep=&sweep}
		workers[index] = thread.create_and_start_with_poly_data(&run, _run_seed)
	}
	for worker in workers {
		thread.join(worker)
		thread.destroy(worker)
	}

	passes: int
	counts: [Verdict]int

	for run in runs {
		counts[run.verdict] += 1
		if run.verdict == .Stable {
			passes += 1
		}

		fmt.printfln(
			"seed %d | %v | peak upright %.0f%% | last %.0f%% | mastered at %.0fs | ran %.0fs over %d episodes | dips %d (max streak %d) | value fit %.2f (%d samples)",
			run.seed, run.verdict, run.peak_upright * 100, run.last_upright * 100,
			run.mastered_at, run.stopped_at, run.episodes, run.low_episodes, run.max_low_streak,
			run.value_fit, run.fit_samples,
		)
	}

	testing.expectf(
		t,
		passes >= REQUIRED_PASSES,
		"%d of %d seeds held %.0f%% upright out to %.0f sim-seconds, need %d (%d never reached %.0f%% within %.0fs, %d collapsed for %d+ episodes below %.0f%% without recovering, %d abandoned once the verdict was decided)",
		passes, SEEDS, DEGRADE_FLOOR * 100, HOLD_DEADLINE, REQUIRED_PASSES,
		counts[.Slow], MASTERY_UPRIGHT * 100, LEARN_DEADLINE,
		counts[.Collapsed], DEGRADE_STREAK, DEGRADE_FLOOR * 100,
		counts[.Aborted],
	)
}
