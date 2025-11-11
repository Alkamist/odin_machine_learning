package main

import "core:os"
import "core:fmt"
import "core:math/rand"
import "../utility"
import ml "../../"
import "../../mlp"
import "../../gru"


// In this example, a Gated Recurrent Unit looks at random
// snippets of text from the file and learns how to predict
// the next byte. You can then predict the next byte and feed
// the result back into the network to generate text.


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

	ml.init(1024 * 1024)

	model := make_model()
	defer destroy_model(model)

	text, text_ok := os.read_entire_file(FILE_NAME)
	if !text_ok {
		fmt.eprintfln("Failed to read %v", FILE_NAME)
	}

	training_split := int(0.9 * f32(len(text)))

	training_text   := text[:training_split]
	validation_text := text[training_split:]

	for _ in 0 ..< STEPS {
		defer free_all(context.temp_allocator)

		learn(&model, random_sample(training_text, SEQUENCE_LENGTH))

		loss := evaluate(model, random_sample(validation_text, SEQUENCE_LENGTH))
		fmt.printfln("%v, Validation Loss: %v", model.step, loss)

		if model.step % 10 == 0 {
			speak(model, 1024)
		}
	}
}

Model :: struct {
	gru: gru.Gru,
	mlp: mlp.Mlp,
	
	step: int,

	opt: ml.Optimizer,
}

make_model :: proc(allocator := context.allocator) -> (res: Model) {
	res.gru = gru.make(VOCABULARY,               MEMORY_SIZE,             allocator=allocator)
	res.mlp = mlp.make(VOCABULARY + MEMORY_SIZE, HIDDEN_SIZE, VOCABULARY, allocator=allocator)
    return
}

destroy_model :: proc(model: Model) {
	gru.destroy(model.gru)
	mlp.destroy(model.mlp)
}

forward :: proc(model: Model, character: byte) -> ml.Array {
	input := ml.zeros(256)
	input.data[character] = 1

	state     := gru.forward(model.gru, input)
	mlp_input := ml.concat(input, state)
	
	return mlp.forward(model.mlp, mlp_input)
}

evaluate :: proc(model: Model, text: []byte) -> (loss: f32) {
	gru.reset_state(model.gru)

	for i in 0 ..< len(text) - 1 {
		ml.clear()

		character := text[i]
		target    := text[i + 1]

		logits := forward(model, character)
		
		loss += ml.cross_entropy(logits, {int(target)}).data[0]
	}

	loss /= f32(len(text) - 1)

	return
}

learn :: proc(model: ^Model, text: []byte) {
	lr := utility.linear_learning_rate(LEARNING_RATE, 0, model.step, STEPS)

	gru.reset_state(model.gru)

	for i in 0 ..< len(text) - 1 {
		ml.clear()

		character := text[i]
		target    := text[i + 1]

		logits := forward(model^, character)

		_ = ml.cross_entropy(logits, {int(target)})

		ml.backward()

		if ml.optimize(&model.opt, period=PERIOD, learning_rate=lr) {
			gru.update(model.opt, model.gru)
			mlp.update(model.opt, model.mlp)
		}
	}

	model.step += 1
}

speak :: proc(model: Model, count: int) {
	gru.reset_state(model.gru)

	fmt.print("==============================================================\n\n")

	output: byte

	for i in 0 ..< count {
		ml.clear()

		logits := forward(model, output)
		output  = byte(utility.sample_top_p(logits.data, 0.9, 1))

		fmt.print(rune(output))
	}

	fmt.print("\n\n==============================================================\n")
}

random_sample :: proc(text: []byte, length: int) -> (res: []byte) {
	i := rand.int_max(len(text) - length)
	return text[i:][:length]
}