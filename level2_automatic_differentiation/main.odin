package level2

import "base:runtime"
import "core:os"
import "core:fmt"
import "core:math"
import "core:math/rand"
import "core:slice"
import "core:strconv"
import "core:encoding/csv"

INPUT_SIZE    :: 784
OUTPUT_SIZE   :: 10
BATCH_SIZE    :: 100

main :: proc() {
	defer free_all(context.temp_allocator)
	defer fmt.println("Finished")

	training_set := load_mnist("../mnist_train.csv", 60000)
	defer destroy(training_set)

	validation_set := load_mnist("../mnist_test.csv", 10000)
	defer destroy(validation_set)

	g: Graph
	defer delete(g)

	linear := init_linear(&g, INPUT_SIZE, OUTPUT_SIZE)
	randomize(&g, linear)

	// The order of the training set
	order := make([dynamic]int, len(training_set.input))
	defer delete(order)
	for i in 0 ..< len(order) {
		order[i] = i
	}

	for e in 0 ..< 50 {
		score         := 0
		batch_counter := 0
		adam_timestep := 0

		// Shuffle the training set
		rand.shuffle(order[:])

		// Train on the training set
		for s in order {
			sample_input  := training_set.input[s][:]
			sample_output := training_set.output[s][:]

			g_temp := make(Graph, 0, 64000)
			defer delete(g_temp)
			linear_temp := clone(&g_temp, &g, linear)

			input  := array(&g_temp, sample_input)
			output := forward(&g_temp, linear_temp, input, nil)
			target := array(&g_temp, sample_output)

			reverse(&g_temp, mean_squared_error(&g_temp, output, target))

			// Accumulate gradients
			for i in 0 ..< linear.weights.len {
				slice_gradient(&g, linear.weights)[i] += slice_gradient(&g_temp, linear_temp.weights)[i]
			}
			for i in 0 ..< linear.biases.len {
				slice_gradient(&g, linear.biases)[i] += slice_gradient(&g_temp, linear_temp.biases)[i]
			}

			// Apply gradients in batches
			batch_counter += 1
			if batch_counter >= BATCH_SIZE {
				batch_counter = 0
				adam_timestep += 1
				step_adam(&g, linear.weights, adam_timestep, BATCH_SIZE)
				step_adam(&g, linear.biases, adam_timestep,  BATCH_SIZE)
			}

			if slice.max_index(slice_values(&g_temp, output)[:]) == slice.max_index(sample_output) {
				score += 1
			}
		}

		fmt.printfln("%v, Training set accuracy: %.2f%%", e, 100.0 * f32(score) / f32(len(training_set.input)))
	}

	// Validate with the validation set
	score := 0
	for s in 0 ..< len(validation_set.input) {
		sample_input  := validation_set.input[s][:]
		sample_output := validation_set.output[s][:]

		g_temp := make(Graph, 0, 64000)
		defer delete(g_temp)
		linear_temp := clone(&g_temp, &g, linear)

		input  := array(&g_temp, sample_input)
		output := forward(&g_temp, linear_temp, input, nil)

		if slice.max_index(slice_values(&g_temp, output)[:]) == slice.max_index(sample_output) {
			score += 1
		}
	}
	fmt.printfln("Validation set accuracy: %.2f%%", 100.0 * f32(score) / f32(len(validation_set.input)))
}

destroy :: proc {
	_destroy_mnist,
}
clone :: proc {
	_clone_graph,
	_clone_linear,
}
array :: proc {
	_array_from_len,
	_array_from_slice,
}

Op :: enum u8 {
	Value,
	Add,
	Sub,
	Mul,
	Pow,
	Sigmoid,
}
Node :: struct {
	kind:     Op,
	value:    f32,
	gradient: f32,
	adam_m:   f32,
	adam_v:   f32,
	left:     int,
	right:    int,
}
Graph :: #soa[dynamic]Node

value :: proc(g: ^Graph, value: f32) -> int {
	append(g, Node{
		kind  = .Value,
		value = value,
	})
	return len(g) - 1
}
add :: proc(g: ^Graph, left, right: int) -> int {
	append(g, Node{
		kind  = .Add,
		value = g[left].value + g[right].value,
		left  = left,
		right = right,
	})
	return len(g) - 1
}
sub :: proc(g: ^Graph, left, right: int) -> int {
	append(g, Node{
		kind  = .Sub,
		value = g[left].value - g[right].value,
		left  = left,
		right = right,
	})
	return len(g) - 1
}
mul :: proc(g: ^Graph, left, right: int) -> int {
	append(g, Node{
		kind  = .Mul,
		value = g[left].value * g[right].value,
		left  = left,
		right = right,
	})
	return len(g) - 1
}
pow :: proc(g: ^Graph, left, right: int) -> int {
	append(g, Node{
		kind  = .Pow,
		value = math.pow(g[left].value, g[right].value),
		left  = left,
		right = right,
	})
	return len(g) - 1
}
sigmoid :: proc(g: ^Graph, index: int) -> int {
	append(g, Node{
		kind  = .Sigmoid,
		value = 1.0 / (1.0 + math.exp(-g[index].value)),
		left  = index,
		right = 0,
	})
	return len(g) - 1
}
_reverse :: proc(g: ^Graph, index: int) {
	node := g[index]
	switch node.kind {
	case .Value:
	case .Add:
		g[node.left].gradient += node.gradient
		g[node.right].gradient += node.gradient
	case .Sub:
		g[node.left].gradient += node.gradient
		g[node.right].gradient -= node.gradient
	case .Mul:
		g[node.left].gradient += node.gradient * g[node.right].value
		g[node.right].gradient += node.gradient * g[node.left].value
	case .Pow:
		left := g[node.left].value
		right := g[node.right].value
		g[node.left].gradient += node.gradient * right * math.pow(left, right - 1)
		g[node.right].gradient += node.gradient * math.ln(left) * math.pow(left, right)
	case .Sigmoid:
		g[node.left].gradient += node.gradient * node.value * (1.0 - node.value)
	}
}
reverse :: proc(g: ^Graph, index: int) {
	g[index].gradient = 1
	for i := len(g) - 1; i >= 1; i -= 1 {
		_reverse(g, i)
	}
}
softmax :: proc(g: ^Graph, arr: Array) {
	max_value := min(f32)
	for value in slice_values(g, arr) {
		if value > max_value {
			max_value = value
		}
	}
	sum: f32
	for &value in slice_values(g, arr) {
		value = math.exp(value - max_value)
		sum += value
	}
	for &value in slice_values(g, arr) {
		value /= sum
	}
	return
}
mean_squared_error :: proc(g: ^Graph, output, target: Array) -> (loss: int) {
	loss = value(g, 0)
	for o in 0 ..< output.len {
		loss = add(g,
			loss,
			pow(g,
				sub(g, output.offset + o, target.offset + o),
				value(g, 2),
			)
		)
	}
	loss = mul(g,
		loss,
		value(g, 1.0 / f32(output.len))
	)
	return
}


Array :: struct {
	offset: int,
	len:    int,
}
_array_from_len :: proc(g: ^Graph, len: int) -> (arr: Array) {
	arr.offset = value(g, 0)
	arr.len    = len
	for _ in 1 ..< len {
		value(g, 0)
	}
	return
}
_array_from_slice :: #force_inline proc(g: ^Graph, s: []f32) -> (arr: Array) {
	arr = _array_from_len(g, len(s))
	copy(slice_values(g, arr), s)
	return
}
slice_values :: #force_inline proc(g: ^Graph, arr: Array) -> []f32 {
	return ([^]f32)(&g[arr.offset].value)[:arr.len]
}
slice_gradient :: #force_inline proc(g: ^Graph, arr: Array) -> []f32 {
	return ([^]f32)(&g[arr.offset].gradient)[:arr.len]
}
_clone_graph :: #force_inline proc(g_to, g_from: ^Graph, arr_from: Array) -> (res: Array) {
	res = array(g_to, arr_from.len)
	copy(slice_values(g_to, res), slice_values(g_from, arr_from))
	return
}
step_adam :: proc(
	g:             ^Graph, 
	arr:           Array,
	timestep:      int,
	batch_size:    int = 128,
    learning_rate: f32 = 0.001,
    beta1:         f32 = 0.9,
    beta2:         f32 = 0.999,
    epsilon:       f32 = 1e-8,
) {
	for i in 0 ..< arr.len {
		index := arr.offset + i
		gradient := g[index].gradient / f32(batch_size)
		g[index].adam_m = beta1 * g[index].adam_m + (1 - beta1) * gradient
		g[index].adam_v = beta2 * g[index].adam_v + (1 - beta2) * gradient * gradient
		m_corrected := g[index].adam_m / (1 - math.pow(beta1, f32(timestep)))
		v_corrected := g[index].adam_v / (1 - math.pow(beta2, f32(timestep)))
		g[index].value -= learning_rate * m_corrected / (math.sqrt(v_corrected) + epsilon)
	}
	slice.zero(slice_gradient(g, arr))
}

Linear :: struct {
	input_size:  int,
	output_size: int,
	weights:     Array,
	biases:      Array,
}
init_linear :: proc(g: ^Graph, input_size, output_size: int) -> Linear {
	return {
		input_size  = input_size,
		output_size = output_size,
		weights     = array(g, output_size * input_size),
		biases      = array(g, output_size),
	}
}
_clone_linear :: proc(g_to, g_from: ^Graph, linear: Linear) -> (res: Linear) {
	res.input_size  = linear.input_size
	res.output_size = linear.output_size
	res.weights     = clone(g_to, g_from, linear.weights)
	res.biases      = clone(g_to, g_from, linear.weights)
	return
}
randomize :: proc(g: ^Graph, linear: Linear) {
	// He initialization
	scale := math.sqrt(2.0 / f32(linear.input_size))
	for &value in slice_values(g, linear.weights) {
		value = rand.float32_normal(0, 1) * scale
	}
	for &value in slice_values(g, linear.biases) {
		value = rand.float32_normal(0, 1)
	}
}
forward :: proc(
	g:          ^Graph, 
	linear:     Linear, 
	input:      Array, 
	activation: proc(g: ^Graph, index: int) -> int = nil,
) -> (output: Array) {
	runtime.DEFAULT_TEMP_ALLOCATOR_TEMP_GUARD()

	outs := make([dynamic]Node, linear.output_size, context.temp_allocator)

	for o in 0 ..< linear.output_size {
		out := value(g, 0)
		for i in 0 ..< input.len {
			out = add(g,
				out,
				mul(g,
					linear.weights.offset + o * linear.input_size + i,
					input.offset + i,
				)
			)
		}
		out = add(g, out, linear.biases.offset + o)
		if activation != nil {
			out = activation(g, out)
		}
		outs[o] = g[out]
	}

	output = array(g, linear.output_size)
	for o in 0 ..< output.len {
		g[output.offset + o] = outs[o]
	}
	softmax(g, output)

	return
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
_destroy_mnist :: proc(mnist: Mnist) {
	delete(mnist.input)
	delete(mnist.output)
}