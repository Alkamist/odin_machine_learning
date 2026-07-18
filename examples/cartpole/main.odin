package main

import "core:fmt"
import "core:math"
import "core:math/rand"
import "core:os"
import "core:time"

import ml  "../../"
import cpu "../../backends/cpu"

FIXED_DELTA :: 1.0 / 60.0

PIXELS_PER_METER :: 24

main :: proc() {
	tuning_parse(os.args[1:])

	cpu.set_thread_count(tuning.threads)

	ctx := cpu.context_create(1024 * 1024 * 256)
	defer cpu.context_destroy(ctx)

	ml.context_scope(ctx)

	for argument in os.args[1:] {
		if argument == "-headless" {
			headless()
			return
		}
		if argument == "-profile" {
			profile()
			return
		}
		if argument == "-bench" {
			bench()
			return
		}
	}

	interactive()
}

interactive :: proc() {
	game_state: State
	init(&game_state)
	defer destroy(&game_state)

	agent := agent_make()
	defer agent_destroy(agent)

	human:    bool
	action:   Action
	timestep: Fixed_Timestep

	window_open()
	defer window_close()

	for !window_should_close() {
		defer free_all(context.temp_allocator)

		frame_begin()

		if toggle_pressed() {
			human = !human
		}

		if mouse_pressed() {
			mouse_begin(&game_state, mouse_position())
		}
		if mouse_held() {
			game_state.mouse_target = mouse_position()
		}
		else {
			mouse_end(&game_state)
		}

		agent_spent: time.Duration
		steps:       f32

		for fixed_timestep(&timestep, FIXED_DELTA) {
			if human {
				action = human_action(action)
			}
			else {
				start  := time.tick_now()
				action  = agent_step(&agent, game_state)
				agent_spent += time.tick_since(start)
			}
			steps += 1

			if step(&game_state, action, FIXED_DELTA) {
				reset(&game_state)
				agent_forget_episode(&agent)
			}
		}

		draw(game_state, timestep.interpolation)
		draw_status(human, agent.decisions, f32(time.duration_milliseconds(agent_spent)), steps)

		frame_end()
	}
}

// Where the frame time actually goes. Runs only a few hundred frames on one
// seed so it stays cheap to run.
profile :: proc() {
	rand.reset(0)

	game_state: State
	init(&game_state)
	defer destroy(&game_state)

	agent := agent_make()
	defer agent_destroy(agent)

	FRAMES :: 400

	worst: time.Duration
	total: time.Duration

	for _ in 1 ..= FRAMES {
		defer free_all(context.temp_allocator)

		start  := time.tick_now()
		action := agent_step(&agent, game_state)
		spent  := time.tick_since(start)

		total += spent
		worst  = max(worst, spent)

		if step(&game_state, action, FIXED_DELTA) {
			reset(&game_state)
			agent_forget_episode(&agent)
		}
	}

	budget := time.Duration(FIXED_DELTA * f32(time.Second))

	fmt.printfln("mean  %8.3f ms/frame  (%.0f%% of a 60Hz frame)", time.duration_milliseconds(total) / FRAMES, 100 * f64(total / FRAMES) / f64(budget))
	fmt.printfln("worst %8.3f ms/frame  (%.0f%% of a 60Hz frame)", time.duration_milliseconds(worst),          100 * f64(worst)         / f64(budget))
	fmt.printfln("agent train %6.3f ms  plan %6.3f ms  per decision", time.duration_milliseconds(agent_train_time) / (FRAMES / ACTION_REPEAT), time.duration_milliseconds(agent_plan_time) / (FRAMES / ACTION_REPEAT))
}

HEADLESS_SEEDS  :: 6
HEADLESS_FRAMES :: 1200
REPORT_EVERY    :: 100

// Runs the agent with no window and no realtime pacing so its learning curve
// can be read off directly. Learning this fast is stochastic enough that a
// single run says very little, so this averages several seeds.
headless :: proc() {
	uprights: [HEADLESS_FRAMES / REPORT_EVERY]f32
	solved:   [HEADLESS_SEEDS]int

	for seed in 0 ..< HEADLESS_SEEDS {
		rand.reset(u64(seed))
		solved[seed] = -1

		game_state: State
		init(&game_state)
		defer destroy(&game_state)

		agent := agent_make()
		defer agent_destroy(agent)

		previous_score: f32

		for frame in 1 ..= HEADLESS_FRAMES {
			defer free_all(context.temp_allocator)

			action := agent_step(&agent, game_state)

			if step(&game_state, action, FIXED_DELTA) {
				reset(&game_state)
				agent_forget_episode(&agent)
				previous_score = 0
			}

			if frame % REPORT_EVERY == 0 {
				// Score is an integral, so its gain over the window is the mean
				// |angle| held: pi is a perfect upright hold.
				gain    := (game_state.score - previous_score) / (REPORT_EVERY * FIXED_DELTA)
				upright := 100 * gain / math.PI

				uprights[frame / REPORT_EVERY - 1] += upright / HEADLESS_SEEDS

				// "Solved" means it never fell again after this point.
				if upright < 90 {
					solved[seed] = -1
				}
				else if solved[seed] < 0 {
					solved[seed] = frame
				}

				previous_score = game_state.score
			}
		}
	}

	fmt.println("mean upright %, averaged over", HEADLESS_SEEDS, "seeds:")
	for upright, i in uprights {
		fmt.printfln("  frame %4d  %3.0f%%", (i + 1) * REPORT_EVERY, upright)
	}
	fmt.println("frame first held to the end, per seed:", solved)

	// One number to compare variants by: how much of the run after the point it
	// ought to have solved the task was actually spent upright.
	late:  f32
	count: int
	for upright, i in uprights {
		if (i + 1) * REPORT_EVERY >= 400 {
			late  += upright
			count += 1
		}
	}
	held := 0
	for frame in solved {
		if frame >= 0 {
			held += 1
		}
	}
	fmt.printfln("SUMMARY upright_after_400=%.1f held=%d/%d", late / f32(count), held, HEADLESS_SEEDS)
}
