package level3

import "base:runtime"
import "core:os"
import "core:fmt"
import "core:math"
import "core:math/rand"
import "core:strconv"
import "core:encoding/csv"
import "ml"

INPUT_SIZE  :: 784
OUTPUT_SIZE :: 10

main :: proc() {
	defer free_all(context.temp_allocator)
	defer fmt.println("Finished")

	// Uncomment to see what devices you have available
	// ml.print_devices()

	// CPU must be last
	model := init_model({"CUDA0", "CPU"}, INPUT_SIZE, 100, OUTPUT_SIZE)
	defer destroy(model)

	training_set := load_mnist("../mnist_train.csv", 60000)
	defer destroy(training_set)

	validation_set := load_mnist("../mnist_test.csv", 10000)
	defer destroy(validation_set)

	init_optimizer(&model, batch_len=1000)
	for e in 0 ..< 55 {
		epoch(&model, training_set.input, training_set.target)
		accuracy := validate(model, validation_set.input, validation_set.target)
		fmt.printfln("%v, Validation Set Accuracy: %.2f%%", e, accuracy * 100.0)
	}
}

destroy :: proc {
	_destroy_model,
	_destroy_mnist,
}

Model :: struct {
	devices: []ml.Device,

	ctx_input:  ml.Context,
	ctx_static: ml.Context,

	layers: [2]struct {
		weight: ml.Tensor,
		bias:   ml.Tensor,
	},

	optimizer: ml.Adamw_Optimizer,
}
init_model :: proc(devices: []ml.Device, input_size, hidden_size, output_size: int) -> (model: Model) {
	model.devices = make([]ml.Device, len(devices))
	copy(model.devices, devices)

	model.ctx_static = ml.new_context(4 * ml.tensor_overhead())

	model.layers[0].weight = ml.tensor(model.ctx_static, .F32, input_size, hidden_size)
	model.layers[0].bias   = ml.tensor(model.ctx_static, .F32, hidden_size)
	model.layers[1].weight = ml.tensor(model.ctx_static, .F32, hidden_size, output_size)
	model.layers[1].bias   = ml.tensor(model.ctx_static, .F32, output_size)

	ml.set_name(model.layers[0].weight, "layer0.weight")
	ml.set_name(model.layers[0].bias,   "layer0.bias")
	ml.set_name(model.layers[1].weight, "layer1.weight")
	ml.set_name(model.layers[1].bias,   "layer1.bias")

	ml.alloc_tensors(&model.ctx_static, model.devices[0])

	// He initialization
	for layer in model.layers {
		runtime.DEFAULT_TEMP_ALLOCATOR_TEMP_GUARD()

		weight_temp := make([]f32, ml.len(layer.weight), context.temp_allocator)
		bias_temp   := make([]f32, ml.len(layer.bias),   context.temp_allocator)

		scale := math.sqrt(2.0 / f32(ml.len(layer.weight, 0)))
		for &value in weight_temp {
			value = rand.float32_normal(0, 1) * scale
		}
		for &value in bias_temp {
			value = rand.float32_normal(0, 1)
		}

		ml.set_data(layer.weight, weight_temp[:])
		ml.set_data(layer.bias,   bias_temp[:])
	}

	return
}
_destroy_model :: proc(model: Model) {
	ml.free(model.ctx_input)
	ml.free(model.ctx_static)
	ml.free(model.optimizer)
}
build :: proc(ctx: ml.Context, model: rawptr, input: ml.Tensor) -> (output: ml.Tensor) {
	model := cast(^Model)model

	for layer in model.layers {
		ml.set_parameter(layer.weight)
		ml.set_parameter(layer.bias)
	}

	output = input

	for i in 0 ..< len(model.layers) - 1 {
		layer := model.layers[i]
		output = ml.relu(ctx,
			ml.add(ctx,
				ml.matmul(ctx,
					layer.weight,
					output,
				),
				layer.bias,
			)
		)
	}

	final_layer := model.layers[len(model.layers) - 1]

	output = ml.add(ctx,
		ml.matmul(ctx,
			final_layer.weight,
			output,
		),
		final_layer.bias,
	)

	ml.set_name(output, "output")
	ml.set_output(output)

	return
}
predict :: proc(model: Model, input: []f32) -> int {
	model := model

	scheduler := ml.new_scheduler(model.devices)
	defer ml.free(scheduler)

	ctx := ml.new_context(ml.DEFAULT_GRAPH_SIZE * ml.tensor_overhead() + ml.graph_overhead())
	defer ml.free(ctx)

	input_tensor := ml.tensor(ctx, .F32, len(input))
	ml.set_input(input_tensor)

	ml.alloc_tensors(&ctx, model.devices[0])
	ml.set_data(input_tensor, input)

	answer := ml.argmax(ctx, build(ctx, &model, input_tensor))

	graph := ml.graph(ctx)
	ml.forward(graph, answer)
	ml.compute(scheduler, graph)

	return int(ml.get_data(answer, i32)[0])
}
init_optimizer :: proc(model: ^Model, batch_len: int) {
	model := model

	ml.clear(&model.ctx_input)
	model.ctx_input = ml.new_context(ml.tensor_overhead())
	input := ml.tensor(model.ctx_input, .F32, ml.len(model.layers[0].weight, 0), batch_len)
	ml.set_name(input, "input")
	ml.alloc_tensors(&model.ctx_input, model.devices[0])

	model.optimizer = ml.new_adamw_optimizer(model.devices, model, input, build, batch_len)
}
epoch :: proc(model: ^Model, input, target: ml.Tensor) {
	ml.epoch(&model.optimizer, input, target)
}
validate :: proc(model: Model, input, target: ml.Tensor) -> (accuracy: f32) {
	model := model

	scheduler := ml.new_scheduler(model.devices)
	defer ml.free(scheduler)

	ml.set_input(input)
	ml.set_output(target)

	ctx := ml.new_context(ml.DEFAULT_GRAPH_SIZE * ml.tensor_overhead() + ml.graph_overhead())
	defer ml.free(ctx)

	score := ml.count_equal(ctx, ml.argmax(ctx, build(ctx, &model, input)), ml.argmax(ctx, target))

	graph := ml.graph(ctx)
	ml.forward(graph, score)
	ml.compute(scheduler, graph)

	runtime.DEFAULT_TEMP_ALLOCATOR_TEMP_GUARD()

	return f32(ml.get_data(score, i64, context.temp_allocator)[0]) / f32(ml.len(target, 1))
}

Mnist :: struct {
	ctx:    ml.Context,
	input:  ml.Tensor,
	target: ml.Tensor,
}
load_mnist :: proc(file_name: string, count: int) -> (mnist: Mnist) {
	runtime.DEFAULT_TEMP_ALLOCATOR_TEMP_GUARD()

	file_data, success := os.read_entire_file(file_name, context.temp_allocator)
	if !success {
		fmt.eprintln("Failed to load mnist data from ", file_name)
		return
	}

	csv_reader: csv.Reader
	csv.reader_init_with_string(&csv_reader, cast(string)file_data, context.temp_allocator)

	_, _ = csv.read(&csv_reader)

	mnist.ctx = ml.new_context(2 * ml.tensor_overhead())

	mnist.input  = ml.tensor(mnist.ctx, .F32, INPUT_SIZE,  count)
	mnist.target = ml.tensor(mnist.ctx, .F32, OUTPUT_SIZE, count)

	ml.set_name(mnist.input,  "mnist_input")
	ml.set_name(mnist.target, "mnist_target")

	ml.alloc_tensors(&mnist.ctx)

	input  := make([]f32, count * INPUT_SIZE,  context.temp_allocator)
	target := make([]f32, count * OUTPUT_SIZE, context.temp_allocator)

	for i in 0 ..< count {
		values_str, err := csv.read(&csv_reader)
		if err != nil {
			break
		}

		y_int, _ := strconv.parse_i64(values_str[0])
		target[i * OUTPUT_SIZE + int(y_int)] = 1

		for j in 0 ..< INPUT_SIZE {
			value_int, _ := strconv.parse_i64(values_str[j + 1])
			input[i * INPUT_SIZE + j] = f32(value_int) / 255.0
		}
	}

	ml.set_data(mnist.input,  input[:])
	ml.set_data(mnist.target, target[:])

	return
}
_destroy_mnist :: proc(mnist: Mnist) {
	ml.free(mnist.ctx)
}