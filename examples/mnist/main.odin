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
import "../../mlp"

BATCH_SIZE :: 100

main :: proc() {
	defer fmt.println("Finished")

	ctx := ml.context_create(1024 * 1024)
	defer ml.context_destroy(ctx)
	ml.context_scope(ctx)
	ml.set_thread_count(24)

	model := make_model()
	defer destroy_model(model)

	training_set := load_mnist("../data/mnist_train.csv", 60000)
	defer destroy_mnist(training_set)

	validation_set := load_mnist("../data/mnist_test.csv", 10000)
	defer destroy_mnist(validation_set)

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
			learn(&model, sample_mnist(training_set, order[b], BATCH_SIZE))
		}

		score := 0
		for b in 0 ..< validation_set.samples / BATCH_SIZE {
			inputs, targets := sample_mnist(validation_set, b, BATCH_SIZE)

			predictions: [BATCH_SIZE]int
			predict(model, inputs, predictions[:])

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

make_model :: proc(allocator := context.allocator) -> (model: Model) {
	model.mlp = mlp.make(MNIST_IMAGE_SIZE, 128, MNIST_CLASS_COUNT, allocator=allocator)
	return
}

destroy_model :: proc(model: Model) {
	mlp.destroy(model.mlp)
}

forward :: proc(model: Model, input: []f32, batch_size: int) -> ml.Tensor {
	x := ml.reshape(ml.tensor(input), batch_size, MNIST_IMAGE_SIZE)
	return mlp.forward(model.mlp, x)
}

predict :: proc(model: Model, input: []f32, predictions: []int) {
	count := len(predictions)

	ml.clear()

	logits             := forward(model, input, count)
	probabilities      := ml.softmax(logits)
	probabilities_data := probabilities.data

	class_size := len(probabilities_data) / count

	for i in 0 ..< count {
		predictions[i] = slice.max_index(probabilities_data[i * class_size:][:class_size])
	}
}

learn :: proc(model: ^Model, input: []f32, targets: []int) {
	ml.clear()

	logits := forward(model^, input, len(targets))
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

load_mnist :: proc(file_name: string, samples: int, allocator := context.allocator) -> (mnist: Mnist) {
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

destroy_mnist :: proc(mnist: Mnist) {
	delete(mnist.inputs)
	delete(mnist.targets)
}

sample_mnist :: proc(mnist: Mnist, i, batch_size: int) -> (inputs: []f32, targets: []int) {
	inputs  = mnist.inputs[i * MNIST_IMAGE_SIZE * batch_size:][:MNIST_IMAGE_SIZE * batch_size]
	targets = mnist.targets[i * batch_size:][:batch_size]
	return
}