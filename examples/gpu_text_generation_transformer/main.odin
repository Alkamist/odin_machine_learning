// GPU text generation: trains a Transformer end-to-end on the GPU and
// generates text from it. Single ml.Context on the GPU backend — model
// parameters, activations, optimizer state, gradients all live on
// device.
//
// Build: odin build examples/gpu_text_generation_transformer -o:speed -no-bounds-check -microarch:native -out:examples/gpu_text_generation_transformer/gpu_text_generation_transformer.exe

package main

import "core:os"
import "core:fmt"
import "core:math/rand"
import "core:time"

import "../utility"
import ml  "../../"
import gpu "../../gpu"
import tfm "../../transformer"

FILE_NAME :: "../data/stories_short.txt"

STEPS         :: 10000
LEARNING_RATE :: 0.001
PERIOD        :: 12

LAYERS          :: 4
ATTENTION_HEADS :: 4
EMBEDDING_SIZE  :: 128
VOCABULARY      :: 256
SEQUENCE_LENGTH :: 64

main :: proc() {
	defer fmt.println("Finished")

	ctx := ml.context_create(256 * 1024 * 1024, gpu.backend())
	defer ml.context_destroy(ctx)
	ml.context_scope(ctx)

	model := make_model()
	defer destroy_model(model)

	text, text_err := os.read_entire_file(FILE_NAME, context.allocator)
	if text_err != nil {
		fmt.eprintfln("Failed to read %v", FILE_NAME)
	}
	defer delete(text)

	training_split := int(0.9 * f32(len(text)))

	training_text   := text[:training_split]
	validation_text := text[training_split:]

	step_t0 := time.tick_now()
	steps_in_window: int

	for {
		defer free_all(context.temp_allocator)

		input, target := random_sample(training_text, SEQUENCE_LENGTH)
		if learn(&model, input, target) {
			steps_in_window += 1

			if model.opt.iteration % 100 == 0 {
				vinput, vtarget := random_sample(validation_text, SEQUENCE_LENGTH)
				loss := evaluate(model, vinput, vtarget)

				dt := time.tick_since(step_t0)
				steps_per_sec := f64(steps_in_window) * 1e9 / f64(dt)
				fmt.printfln("step %v  val_loss=%.4f  %.1f steps/sec",
					model.opt.iteration, loss, steps_per_sec)
				step_t0 = time.tick_now()
				steps_in_window = 0
			}

			if model.opt.iteration % 500 == 0 {
				speak(model, 1024)
			}
		}
	}
}

Model :: struct {
	transformer: tfm.Transformer,
	opt:         ml.Optimizer,
}

make_model :: proc(allocator := context.allocator) -> (res: Model) {
	res.transformer = tfm.make(LAYERS, ATTENTION_HEADS, EMBEDDING_SIZE, VOCABULARY)
	return
}

destroy_model :: proc(model: Model) {
	tfm.destroy(model.transformer)
}

forward :: proc(model: Model, text: []byte) -> ml.Tensor {
	tokens := make([]int, len(text), context.temp_allocator)
	for i in 0 ..< len(text) {
		tokens[i] = int(text[i])
	}
	return tfm.forward(model.transformer, tokens)
}

evaluate :: proc(model: Model, text: []byte, target: byte) -> f32 {
	targets := make([]int, len(text), context.temp_allocator)
	for i in 0 ..< len(text) - 1 {
		targets[i] = int(text[i + 1])
	}
	targets[len(targets) - 1] = int(target)

	ml.clear()

	logits := forward(model, text)

	loss := ml.cross_entropy(logits, targets)
	loss  = ml.mean(loss)

	// `get_data` flushes the auto-batch and reads the scalar back.
	scalar: [1]f32
	loss_t := loss
	loss_t.backend.get_data(&loss_t, scalar[:])
	return scalar[0]
}

learn :: proc(model: ^Model, text: []byte, target: byte) -> bool {
	targets := make([]int, len(text), context.temp_allocator)
	for i in 0 ..< len(text) - 1 {
		targets[i] = int(text[i + 1])
	}
	targets[len(targets) - 1] = int(target)

	ml.clear()

	logits := forward(model^, text)

	loss := ml.cross_entropy(logits, targets)
	loss  = ml.mean(loss)

	ml.backward()

	lr := utility.linear_learning_rate(LEARNING_RATE, 0, int(model.opt.iteration), STEPS)

	stepped := false
	if ml.optimize(&model.opt, period=PERIOD, learning_rate=lr) {
		tfm.update(model.opt, model.transformer)
		stepped = true
	}
	return stepped
}

// Autoregressive generation. Forward + sample for each token, all on
// GPU. The trailing-token logits row is folded into the batch download
// so each token costs one queue submit + one wait.
speak :: proc(model: Model, token_count: int) {
	fmt.print("==============================================================\n\n")

	text: [SEQUENCE_LENGTH]byte
	vocab_logits := make([]f32, VOCABULARY, context.temp_allocator)
	full_logits  := make([]f32, SEQUENCE_LENGTH * VOCABULARY, context.temp_allocator)

	t0 := time.tick_now()
	for i in 0 ..< token_count {
		ml.clear()

		logits := forward(model, text[:])

		logits_t := logits
		logits_t.backend.get_data(&logits_t, full_logits)

		row := min(i, SEQUENCE_LENGTH - 1)
		for k in 0 ..< VOCABULARY do vocab_logits[k] = full_logits[row * VOCABULARY + k]

		output := utility.sample_top_p(vocab_logits, 0.9, 1)
		fmt.print(rune(output))

		if i + 1 < SEQUENCE_LENGTH {
			text[i + 1] = byte(output)
		} else {
			copy(text[:], text[1:])
			text[len(text) - 1] = byte(output)
		}
	}
	dt := time.tick_since(t0)

	fmt.print("\n\n==============================================================\n")
	fmt.printfln("(GPU inference: %v tokens in %.2f ms = %.1f tokens/sec)",
		token_count, f64(dt) / 1e6, f64(token_count) * 1e9 / f64(dt))
}

random_sample :: proc(text: []byte, length: int) -> (input: []byte, target: byte) {
	i := rand.int_max(len(text) - length - 1)
	return text[i:][:length], text[i + length]
}
