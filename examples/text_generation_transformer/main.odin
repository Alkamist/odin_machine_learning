package main

import "core:os"
import "core:fmt"
import "core:math"
import "core:math/rand"
import "core:slice"
import "../utility"
import "../../ml"
import tfm "../../ml/transformer"


// In this example, a Transformer looks at random snippets of 
// text from the file and learns how to predict the next byte. 
// You can then predict the next byte and feed the result back 
// into the network to generate text.


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

	ml.init(1024 * 1024 * 16)
	ml.set_thread_count(24)
	
	model := make_model()
	defer destroy_model(model)

	text, text_ok := os.read_entire_file(FILE_NAME)
	if !text_ok {
		fmt.eprintfln("Failed to read %v", FILE_NAME)
	}

	training_split := int(0.9 * f32(len(text)))

	training_text   := text[:training_split]
	validation_text := text[training_split:]

	for {
		defer free_all(context.temp_allocator)

		if learn(&model, random_sample(training_text, SEQUENCE_LENGTH)) {
			if model.opt.iteration % 100 == 0 {
				loss := evaluate(model, random_sample(validation_text, SEQUENCE_LENGTH))
				fmt.printfln("%v, Validation Loss: %v", model.opt.iteration, loss)
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

text_to_tokens :: proc(text: []byte) -> (res: ml.Array) {
	res = ml.zeros(len(text))
	for i in 0 ..< len(text) {
		res.data[i] = f32(text[i]) / 255.0
	}
	return
}

forward :: proc(model: Model, text: []byte) -> ml.Array {
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

	return loss.data[0]
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

	if ml.optimize(&model.opt, period=PERIOD, learning_rate=lr) {
		tfm.update(model.opt, model.transformer)
		return true
	}

	return false
}

speak :: proc(model: Model, token_count: int) {
	fmt.print("==============================================================\n\n")

	text: [SEQUENCE_LENGTH]byte

	for i in 0 ..< token_count {
		ml.clear()

		logits       := forward(model, text[:])
		logits_data  := logits.data
		token_logits := logits_data[min(i, SEQUENCE_LENGTH - 1) * model.transformer.vocabulary_size:][:model.transformer.vocabulary_size]
		output       := utility.sample_top_p(token_logits, 0.9, 1)

		fmt.print(rune(output))

		if i + 1 < SEQUENCE_LENGTH {
			text[i + 1] = byte(output)
		} else {
			copy(text[:], text[1:])
			text[len(text) - 1] = byte(output)
		}
	}

	fmt.print("\n\n==============================================================\n")
}

random_sample :: proc(text: []byte, length: int) -> (input: []byte, target: byte) {
	i := rand.int_max(len(text) - length - 1)
	return text[i:][:length], text[i + length]
}