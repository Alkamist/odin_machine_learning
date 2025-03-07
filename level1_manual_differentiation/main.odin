package level1

import "base:runtime"
import "core:os"
import "core:fmt"
import "core:math"
import "core:math/rand"
import "core:slice"
import "core:strconv"
import "core:encoding/csv"

INPUT_SIZE  :: 784
HIDDEN_SIZE :: 100
OUTPUT_SIZE :: 10

main :: proc() {
	defer free_all(context.temp_allocator)
	defer fmt.println("Finished")
	
	training_set := load_mnist("../mnist_train.csv", 60000)
	defer destroy(training_set)

	validation_set := load_mnist("../mnist_test.csv", 10000)
	defer destroy(validation_set)

	// The model is too big for the stack
	model := new(Model)
	defer free(model)
	init_model(model)

	// The order of the training set
	order := make([]int, len(training_set.output))
	defer delete(order)
	for i in 0 ..< len(order) {
		order[i] = i
	}

	for epoch in 0 ..< 50 {
		// Shuffle the training set
		rand.shuffle(order)

		// Train on the training set
		for s in 0 ..< len(training_set.output) {
			learn(model, training_set.input[s], training_set.output[s], batch_size=100)
		}

		// Validate with the validation set
		score := 0
		for s in 0 ..< len(validation_set.output) {
			output := forward(model^, validation_set.input[s])
			if slice.max_index(output[:]) == slice.max_index(validation_set.output[s][:]) {
				score += 1
			}
		}
		fmt.printfln("%v Validation set accuracy: %.2f%%", epoch, 100.0 * f32(score) / f32(len(validation_set.output)))
	}
}

sigmoid_value :: proc(x: f32) -> f32 {
	return 1.0 / (1.0 + math.exp(-x))
}
sigmoid :: proc(x: [$N]f32) -> (res: [N]f32) {
	for i in 0 ..< len(x) {
		res[i] = sigmoid_value(x[i])
	}
	return
}
sigmoid_derivative :: proc(x: f32) -> f32 {
	return sigmoid_value(x) * (1.0 - sigmoid_value(x))
}
softmax :: proc(x: [$N]f32) -> (res: [N]f32) {
	max_value := min(f32)
	for i in 0 ..< len(x) {
		max_value = max(max_value, x[i])
	}
	sum: f32 = 0
	for i in 0 ..< len(x) {
		res[i] = math.exp(x[i] - max_value)
		sum += res[i]
	}
	for i in 0 ..< len(res) {
		res[i] /= sum
	}
	return
}

forward :: proc {
	_forward_linear,
	_forward_model,
}

// Parameters need 3 extra floating points for training
Parameter :: struct {
	value:    f32,
	gradient: f32,
	adam_m:   f32,
	adam_v:   f32,
}
update :: proc(parameter: ^Parameter, timestep, batch_size: int, learning_rate, beta1, beta2, epsilon: f32) {
	gradient := parameter.gradient / f32(batch_size)
	parameter.adam_m = beta1 * parameter.adam_m + (1 - beta1) * gradient
	parameter.adam_v = beta2 * parameter.adam_v + (1 - beta2) * gradient * gradient
	m_corrected := parameter.adam_m / (1 - math.pow(beta1, f32(timestep)))
	v_corrected := parameter.adam_v / (1 - math.pow(beta2, f32(timestep)))
	parameter.value -= learning_rate * m_corrected / (math.sqrt(v_corrected) + epsilon)
	parameter.gradient = 0
}

Linear :: struct($I, $O: int) {
	weights: [O][I]Parameter,
	biases:  [O]Parameter,
}
init_linear :: proc(linear: ^Linear($I, $O)) {
	// He initialization
	scale := math.sqrt(2.0 / f32(I))
	for o in 0 ..< O {
		for i in 0 ..< I {
			linear.weights[o][i].value = rand.float32_normal(0, 1) * scale
		}
		linear.biases[o].value = rand.float32_normal(0, 1)
	}
}
_forward_linear :: proc(linear: Linear($I, $O), input: [I]f32) -> (res: [O]f32) {
	for o in 0 ..< O {
		res[o] = linear.biases[o].value
		for i in 0 ..< I {
			res[o] += linear.weights[o][i].value * input[i]
		}
	}
	return
}

Model :: struct {
	timestep:      int,
	batch_counter: int,
	layer0:        Linear(INPUT_SIZE,  HIDDEN_SIZE),
	layer1:        Linear(HIDDEN_SIZE, OUTPUT_SIZE),
}
init_model :: proc(model: ^Model) {
	init_linear(&model.layer0)
	init_linear(&model.layer1)
	return
}
_forward_model :: proc(model: Model, input: [INPUT_SIZE]f32) -> (res: [OUTPUT_SIZE]f32) {
	hidden := sigmoid(forward(model.layer0, input))
	res     = softmax(forward(model.layer1, hidden))
	return
}
learn :: proc(
	model:         ^Model,
	input:         [INPUT_SIZE]f32,
	target:        [OUTPUT_SIZE]f32,
	batch_size:    int = 100,
	learning_rate: f32 = 0.001,
	beta1:         f32 = 0.9,
	beta2:         f32 = 0.999,
	epsilon:       f32 = 1e-8,
) {
	// Forward
	hidden           := forward(model.layer0, input)
	hidden_activated := sigmoid(hidden)
	output           := softmax(forward(model.layer1, hidden_activated))

	// Backward (gradient accumulation)
	deltas: [OUTPUT_SIZE]f32
	for o in 0 ..< OUTPUT_SIZE {
		deltas[o] = (output[o] - target[o]) * sigmoid_derivative(output[o])
		for h in 0 ..< HIDDEN_SIZE {
			model.layer1.weights[o][h].gradient += deltas[o] * hidden_activated[h]
		}
		model.layer1.biases[o].gradient += deltas[o]
	}
	for h in 0 ..< HIDDEN_SIZE {
		d_cost_o: f32
		for o in 0 ..< OUTPUT_SIZE {
			d_cost_o += deltas[o] * model.layer1.weights[o][h].value
		}
		delta := d_cost_o * sigmoid_derivative(hidden[h])
		for i in 0 ..< INPUT_SIZE {
			model.layer0.weights[h][i].gradient += delta * input[i]
		}
		model.layer0.biases[h].gradient += delta
	}

	// Update parameters with accumulated gradients in batches
	model.batch_counter += 1
	if model.batch_counter > batch_size {
		model.timestep += 1
		for h in 0 ..< HIDDEN_SIZE {
			for i in 0 ..< INPUT_SIZE {
				update(&model.layer0.weights[h][i], model.timestep, batch_size, learning_rate, beta1, beta2, epsilon)
			}
			update(&model.layer0.biases[h], model.timestep, batch_size, learning_rate, beta1, beta2, epsilon)
		}
		for o in 0 ..< OUTPUT_SIZE {
			for h in 0 ..< HIDDEN_SIZE {
				update(&model.layer1.weights[o][h], model.timestep, batch_size, learning_rate, beta1, beta2, epsilon)
			}
			update(&model.layer1.biases[o], model.timestep, batch_size, learning_rate, beta1, beta2, epsilon)
		}
		model.batch_counter = 0
	}
}

Mnist :: struct {
	input:  [dynamic][INPUT_SIZE]f32,
	output: [dynamic][OUTPUT_SIZE]f32,
}
load_mnist :: proc(file_name: string, size: int, allocator := context.allocator) -> (mnist: Mnist) {
	runtime.DEFAULT_TEMP_ALLOCATOR_TEMP_GUARD()
	
	file_data, success := os.read_entire_file(file_name, context.temp_allocator)
	if !success {
		fmt.eprintln("Failed to load mnist data from ", file_name)
		return
	}

	csv_reader: csv.Reader
	csv.reader_init_with_string(&csv_reader, cast(string)file_data, context.temp_allocator)

	_, _ = csv.read(&csv_reader)

	mnist.input  = make([dynamic][INPUT_SIZE]f32,  size, allocator)
	mnist.output = make([dynamic][OUTPUT_SIZE]f32, size, allocator)

	for n in 0 ..< size {
		values_str, err := csv.read(&csv_reader)
		if err != nil {
			break
		}

		y_int, _ := strconv.parse_i64(values_str[0])
		mnist.output[n][int(y_int)] = 1

		for i in 0 ..< INPUT_SIZE {
			value_int, _ := strconv.parse_i64(values_str[i + 1])
			mnist.input[n][i] = f32(value_int) / 255.0
		}
	}

	return
}
destroy :: proc(mnist: Mnist) {
	delete(mnist.input)
	delete(mnist.output)
}