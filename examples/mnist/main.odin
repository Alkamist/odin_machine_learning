// In this example, a Multilayer Perceptron will learn how
// to look at the raw data of hand-drawn digits from the
// MNIST dataset and predict which digits they are.
//
// You will need to unzip the dataset in the data folder first.

package main

import "core:os"
import "core:fmt"
import "core:math/rand"
import "core:slice"
import "core:strconv"
import "core:encoding/csv"

import ml "../../"
import cpu "../../backends/cpu"
import gpu "../../backends/vulkan"
import "../../networks/mlp"

BATCH_SIZE :: 1000

main :: proc() {
	defer fmt.println("Finished")

	cpu.set_thread_count(24)

	ctx := cpu.context_create(1024 * 1024 * 256)
	defer cpu.context_destroy(ctx)

	// ctx := gpu.context_create()
	// defer gpu.context_destroy(ctx)

	ml.context_scope(ctx)

	model := model_make()
	defer model_destroy(model)

	training_set := mnist_load("examples/data/mnist_train.csv", 60000)
	defer mnist_destroy(training_set)

	validation_set := mnist_load("examples/data/mnist_test.csv", 10000)
	defer mnist_destroy(validation_set)

	// Create an order to shuffle so that we can train on
	// every datapoint every epoch but in random order.
	order := make([]int, training_set.samples / BATCH_SIZE)
	defer delete(order)
	for i in 0 ..< len(order) {
		order[i] = i
	}

	for epoch in 0 ..< 50 {
		defer free_all(context.temp_allocator)

		rand.shuffle(order)

		for b in 0 ..< training_set.samples / BATCH_SIZE {
			model_learn(&model, mnist_sample(training_set, order[b], BATCH_SIZE))
		}

		score := 0
		for b in 0 ..< validation_set.samples / BATCH_SIZE {
			inputs, targets := mnist_sample(validation_set, b, BATCH_SIZE)

			predictions: [BATCH_SIZE]int
			model_predict(model, inputs, predictions[:])

			for i in 0 ..< BATCH_SIZE {
				if predictions[i] == targets[i] {
					score += 1
				}
			}
		}
		fmt.printfln("%v, Validation Set Accuracy: %.2f%%", epoch, 100.0 * f32(score) / f32(validation_set.samples))
	}
}

Model :: struct {
	mlp: mlp.Mlp,
	opt: ml.Optimizer,
}

model_make :: proc(allocator := context.allocator) -> (model: Model) {
	model.mlp = mlp.make(MNIST_IMAGE_SIZE, 128, MNIST_CLASS_COUNT, allocator=allocator)
	return
}

model_destroy :: proc(model: Model) {
	mlp.destroy(model.mlp)
}

model_forward :: proc(model: Model, input: []f32, batch_size: int) -> ml.Tensor {
	x := ml.reshape(ml.tensor(input), {batch_size, MNIST_IMAGE_SIZE})
	return mlp.forward(model.mlp, x)
}

model_predict :: proc(model: Model, input: []f32, predictions: []int) {
	count := len(predictions)

	ml.clear({.No_Gradients})

	logits             := model_forward(model, input, count)
	probabilities      := ml.softmax(logits)

	probabilities_data := make([]f32, ml.len(probabilities), allocator=context.temp_allocator)
	ml.get_data(probabilities, probabilities_data)

	class_size := len(probabilities_data) / count

	for i in 0 ..< count {
		predictions[i] = slice.max_index(probabilities_data[i * class_size:][:class_size])
	}
}

model_learn :: proc(model: ^Model, input: []f32, targets: []int) {
	ml.clear()

	logits := model_forward(model^, input, len(targets))
	loss   := ml.cross_entropy(logits, targets)

	ml.backward()

	if ml.optimize(&model.opt, period=1) {
		mlp.update(model.opt, model.mlp)
	}
}

MNIST_IMAGE_SIZE  :: 784
MNIST_CLASS_COUNT :: 10

Mnist :: struct {
	samples: int,
	inputs:  []f32,
	targets: []int,
}

mnist_load :: proc(file_name: string, samples: int, allocator := context.allocator) -> (mnist: Mnist) {
	file_data, err := os.read_entire_file(file_name, context.temp_allocator)
	if err != nil {
		fmt.eprintfln("Failed to load mnist data from ", file_name)
		return
	}

	csv_reader: csv.Reader
	csv.reader_init_with_string(&csv_reader, cast(string)file_data, context.temp_allocator)
	defer csv.reader_destroy(&csv_reader)

	_, _ = csv.read(&csv_reader, context.temp_allocator)

	mnist.inputs  = make([]f32, samples * MNIST_IMAGE_SIZE,  allocator)
	mnist.targets = make([]int, samples, allocator)

	for i in 0 ..< samples {
		values_str, err := csv.read(&csv_reader, context.temp_allocator)
		if err != nil {
			break
		}

		y_int, _ := strconv.parse_i64(values_str[0])
		mnist.targets[i] = int(y_int)

		for j in 0 ..< MNIST_IMAGE_SIZE {
			value_int, _ := strconv.parse_i64(values_str[j + 1])
			mnist.inputs[i * MNIST_IMAGE_SIZE + j] = f32(value_int) / 255.0
		}
	}

	mnist.samples = samples
	return
}

mnist_destroy :: proc(mnist: Mnist) {
	delete(mnist.inputs)
	delete(mnist.targets)
}

mnist_sample :: proc(mnist: Mnist, i, batch_size: int) -> (inputs: []f32, targets: []int) {
	inputs  = mnist.inputs[i * MNIST_IMAGE_SIZE * batch_size:][:MNIST_IMAGE_SIZE * batch_size]
	targets = mnist.targets[i * batch_size:][:batch_size]
	return
}