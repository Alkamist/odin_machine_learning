package cartpole_tests

import "core:fmt"
import "core:math/rand"
import "core:sync"
import "core:testing"
import "core:thread"

import ml  "../../../"
import cpu "../../../backends/cpu"

import "../agent"
import "../sim"
import "../world"

CONTEXT_SIZE :: 1024 * 1024 * 256

SEEDS            :: 8
REQUIRED_PASSES  :: 7
ALLOWED_FAILURES :: SEEDS - REQUIRED_PASSES

MASTERY_UPRIGHT :: f32(0.85)
DEGRADE_FLOOR   :: f32(0.70)

LEARN_DEADLINE :: f64(150)
HOLD_DEADLINE  :: f64(15 * 60)

Verdict :: enum {
	Stable,
	Slow,
	Degraded,
	Aborted,
}

Sweep :: struct {
	failures: int,
	abort:    bool,
}

Seed_Run :: struct {
	seed:  u64,
	sweep: ^Sweep,

	verdict:      Verdict,
	peak_upright: f32,
	last_upright: f32,
	mastered_at:  f64,
	stopped_at:   f64,
	episodes:     int,
	value_fit:    f32,
	fit_samples:  int,
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
	mastered: bool

	for {
		action  := agent.act(brain)
		control := action[world.ACTION_AXIS_X]
		done    := sim.step(&game, control, sim.FIXED_DELTA)

		sim_time += f64(sim.FIXED_DELTA)
		agent.drive(brain, sim_time, sim.observe(game), action, episode)

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

		if mastered && upright < DEGRADE_FLOOR {
			_fail(run, .Degraded)
			break
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

		sim.reset(&game)
		episode += 1
	}

	run.value_fit, run.fit_samples = agent.value_fit(brain)
}

@(test)
test_cartpole_learns_fast_and_holds :: proc(t: ^testing.T) {
	cpu.set_thread_count(1)

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
			"seed %d | %v | peak upright %.0f%% | last %.0f%% | mastered at %.0fs | ran %.0fs over %d episodes | value fit %.2f (%d samples)",
			run.seed, run.verdict, run.peak_upright * 100, run.last_upright * 100,
			run.mastered_at, run.stopped_at, run.episodes, run.value_fit, run.fit_samples,
		)
	}

	testing.expectf(
		t,
		passes >= REQUIRED_PASSES,
		"%d of %d seeds held %.0f%% upright out to %.0f sim-seconds, need %d (%d never reached %.0f%% within %.0fs, %d degraded below %.0f%% after mastering, %d abandoned once the verdict was decided)",
		passes, SEEDS, DEGRADE_FLOOR * 100, HOLD_DEADLINE, REQUIRED_PASSES,
		counts[.Slow], MASTERY_UPRIGHT * 100, LEARN_DEADLINE,
		counts[.Degraded], DEGRADE_FLOOR * 100,
		counts[.Aborted],
	)
}
