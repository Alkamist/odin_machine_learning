// In this example, a Gated Recurrent Unit looks at random
// snippets of text from the file and learns how to predict
// the next byte. You can then predict the next byte and feed
// the result back into the network to generate text.

package main

import "core:os"
import "core:fmt"
import "core:math/rand"
import "../utility"
import ml "../../"
import cpu "../../backend_cpu"
import "../../mlp"
import "../../gru"

FILE_NAME :: "../data/stories_short.txt"

STEPS         :: 1000
LEARNING_RATE :: 0.001
PERIOD        :: 128

HIDDEN_SIZE :: 128
MEMORY_SIZE :: 128
VOCABULARY  :: 256

SEQUENCE_LENGTH :: 1024 * 16

main :: proc() {
	defer fmt.println("Finished")

	ctx := cpu.context_create(1024 * 1024)
	defer ml.context_destroy(ctx)
	ml.context_scope(ctx)

	model := model_make()
	defer model_destroy(model)

	text, text_err := os.read_entire_file(FILE_NAME, context.allocator)
	if text_err != nil {
		fmt.eprintfln("Failed to read %v", FILE_NAME)
	}
	defer delete(text)

	training_split := int(0.9 * f32(len(text)))

	training_text   := text[:training_split]
	validation_text := text[training_split:]

	for _ in 0 ..< STEPS {
		defer free_all(context.temp_allocator)

		model_learn(&model, random_sample(training_text, SEQUENCE_LENGTH))

		loss := model_evaluate(model, random_sample(validation_text, SEQUENCE_LENGTH))
		fmt.printfln("%v, Validation Loss: %v", model.step, loss)

		if model.step % 10 == 0 {
			model_speak(model, 1024)
		}
	}
}

Model :: struct {
	gru: gru.Gru,
	mlp: mlp.Mlp,

	step: int,

	opt: ml.Optimizer,
}

model_make :: proc(allocator := context.allocator) -> (model: Model) {
	model.gru = gru.make(VOCABULARY,               MEMORY_SIZE,             allocator=allocator)
	model.mlp = mlp.make(VOCABULARY + MEMORY_SIZE, HIDDEN_SIZE, VOCABULARY, allocator=allocator)
	return
}

model_destroy :: proc(model: Model) {
	gru.destroy(model.gru)
	mlp.destroy(model.mlp)
}

model_forward :: proc(model: Model, character: byte) -> ml.Tensor {
	one_hot := make([]f32, VOCABULARY, allocator=context.temp_allocator)
	one_hot[character] = 1

	input     := ml.tensor(one_hot)
	state     := gru.forward(model.gru, input)
	mlp_input := ml.concat(input, state)

	return mlp.forward(model.mlp, mlp_input)
}

model_evaluate :: proc(model: Model, text: []byte) -> (loss: f32) {
	gru.reset_state(model.gru)

	for i in 0 ..< len(text) - 1 {
		ml.clear()

		character := text[i]
		target    := text[i + 1]

		logits := model_forward(model, character)

		sample_loss: [1]f32
		ml.get_data(ml.cross_entropy(logits, {int(target)}), sample_loss[:])
		loss += sample_loss[0]
	}

	loss /= f32(len(text) - 1)

	return
}

model_learn :: proc(model: ^Model, text: []byte) {
	lr := utility.linear_learning_rate(LEARNING_RATE, 0, model.step, STEPS)

	gru.reset_state(model.gru)

	for i in 0 ..< len(text) - 1 {
		ml.clear()

		character := text[i]
		target    := text[i + 1]

		logits := model_forward(model^, character)

		_ = ml.cross_entropy(logits, {int(target)})

		ml.backward()

		if ml.optimize(&model.opt, period=PERIOD, learning_rate=lr) {
			gru.update(model.opt, model.gru)
			mlp.update(model.opt, model.mlp)
		}
	}

	model.step += 1
}

model_speak :: proc(model: Model, count: int) {
	gru.reset_state(model.gru)

	fmt.print("==============================================================\n\n")

	logits_data := make([]f32, VOCABULARY, allocator=context.temp_allocator)

	output: byte

	for i in 0 ..< count {
		ml.clear()

		logits := model_forward(model, output)
		ml.get_data(logits, logits_data)

		output = byte(utility.sample_top_p(logits_data, 0.9, 1))

		fmt.print(rune(output))
	}

	fmt.print("\n\n==============================================================\n")
}

random_sample :: proc(text: []byte, length: int) -> []byte {
	i := rand.int_max(len(text) - length)
	return text[i:][:length]
}