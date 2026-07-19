package main

import "base:runtime"

import "core:bytes"
import "core:compress/gzip"
import "core:fmt"
import "core:log"

import ml  "../../"
import cpu "../../backends/cpu"
import     "../../dataset"
import     "../../networks/mlp"
import     "../fetch"

BATCH_SIZE :: 1000

main :: proc() {
	defer fmt.println("Finished")

	cpu.set_thread_count(24)

	ctx := cpu.context_create(1024 * 1024 * 256)
	defer cpu.context_destroy(ctx)

	ml.context_scope(ctx)

	model := model_make()
	defer model_destroy(model)

	if !fetch.ensure_assets(MNIST_ASSETS, "MNIST") {
		return
	}

	training_set, training_ok := mnist_load(MNIST_ASSETS[0].dest, MNIST_ASSETS[1].dest)
	if !training_ok {
		return
	}
	defer mnist_destroy(training_set)

	validation_set, validation_ok := mnist_load(MNIST_ASSETS[2].dest, MNIST_ASSETS[3].dest)
	if !validation_ok {
		return
	}
	defer mnist_destroy(validation_set)

	training_batcher := dataset.batcher_make(training_set.samples, BATCH_SIZE)
	defer dataset.batcher_destroy(&training_batcher)

	validation_batcher := dataset.batcher_make(validation_set.samples, BATCH_SIZE, shuffle=false)
	defer dataset.batcher_destroy(&validation_batcher)

	batch_inputs := make([]f32, BATCH_SIZE * MNIST_IMAGE_SIZE)
	defer delete(batch_inputs)

	batch_targets := make([]int, BATCH_SIZE)
	defer delete(batch_targets)

	for epoch in 0 ..< 50 {
		defer free_all(context.temp_allocator)

		dataset.batcher_reset(&training_batcher)
		for batch in dataset.batcher_next(&training_batcher) {
			dataset.gather(batch_inputs, training_set.inputs, batch, stride=MNIST_IMAGE_SIZE)
			dataset.gather(batch_targets, training_set.targets, batch)
			model_learn(&model, batch_inputs, batch_targets)
		}

		score := 0
		dataset.batcher_reset(&validation_batcher)
		for batch in dataset.batcher_next(&validation_batcher) {
			dataset.gather(batch_inputs, validation_set.inputs, batch, stride=MNIST_IMAGE_SIZE)
			dataset.gather(batch_targets, validation_set.targets, batch)

			predictions: [BATCH_SIZE]int
			model_predict(model, batch_inputs, predictions[:])

			for i in 0 ..< BATCH_SIZE {
				if predictions[i] == batch_targets[i] {
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
	model.opt = ml.optimizer_make()
	return
}

model_destroy :: proc(model: Model) {
	model := model
	ml.optimizer_destroy(&model.opt)
	mlp.destroy(model.mlp)
}

model_forward :: proc(model: Model, input: []f32, batch_size: int) -> ml.Tensor {
	x := ml.tensor(input, []int{batch_size, MNIST_IMAGE_SIZE})
	return mlp.forward(model.mlp, x)
}

model_predict :: proc(model: Model, input: []f32, predictions: []int) {
	count := len(predictions)

	ml.clear()

	logits := model_forward(model, input, count)
	ml.argmax(logits, predictions)
}

model_learn :: proc(model: ^Model, input: []f32, targets: []int) {
	ml.clear(training=true)

	logits := model_forward(model^, input, len(targets))
	loss   := ml.mean(ml.cross_entropy(logits, targets))

	ml.backward(loss)

	if ml.optimizer_step(&model.opt) {
		mlp.update(&model.opt, model.mlp)
	}
}

MNIST_IMAGE_SIZE  :: 784
MNIST_CLASS_COUNT :: 10

MNIST_DATA_DIR :: #directory + "data"

// The canonical idx-ubyte files, from the S3 mirror the PyTorch project set up
// after yann.lecun.com stopped serving them reliably. Sizes are the gzipped
// sizes on the wire, which is what ensure_assets checks on disk.
MNIST_ASSETS := []fetch.Asset {
	{
		url  = "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz",
		dest = MNIST_DATA_DIR + "/train-images-idx3-ubyte.gz",
		size = 9_912_422,
	},
	{
		url  = "https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz",
		dest = MNIST_DATA_DIR + "/train-labels-idx1-ubyte.gz",
		size = 28_881,
	},
	{
		url  = "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz",
		dest = MNIST_DATA_DIR + "/t10k-images-idx3-ubyte.gz",
		size = 1_648_877,
	},
	{
		url  = "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz",
		dest = MNIST_DATA_DIR + "/t10k-labels-idx1-ubyte.gz",
		size = 4_542,
	},
}

Mnist :: struct {
	samples: int,
	inputs:  []f32,
	targets: []int,
}

// Decompresses an idx file and returns its payload along with the header-declared
// item count. idx headers are big endian: a magic number whose low byte is the
// number of dimensions, followed by one i32be length per dimension.
@(require_results)
_idx_read :: proc(file_name: string, expect_magic: u32, allocator := context.temp_allocator) -> (payload: []u8, count: int, ok: bool) {
	buffer: bytes.Buffer
	if err := gzip.load(file_name, &buffer, allocator = allocator); err != nil {
		log.errorf("could not decompress %v: %v", file_name, err)
		return
	}

	data := buffer.buf[:]

	dims := int(expect_magic & 0xff)
	header := 4 * (1 + dims)
	if len(data) < header {
		log.errorf("%v is truncated: %d bytes", file_name, len(data))
		return
	}

	be_u32 :: proc(b: []u8) -> u32 {
		return u32(b[0]) << 24 | u32(b[1]) << 16 | u32(b[2]) << 8 | u32(b[3])
	}

	if magic := be_u32(data); magic != expect_magic {
		log.errorf("%v has magic %08x, expected %08x", file_name, magic, expect_magic)
		return
	}

	count = int(be_u32(data[4:]))

	// Every dimension past the first is per-item, so their product is the item
	// stride. For labels that product is empty and the stride is 1 byte.
	stride := 1
	for d in 1 ..< dims {
		stride *= int(be_u32(data[4 * (1 + d):]))
	}

	if len(data) - header < count * stride {
		log.errorf("%v holds %d bytes, expected %d", file_name, len(data) - header, count * stride)
		return
	}
	return data[header:][:count * stride], count, true
}

@(require_results)
mnist_load :: proc(images_file, labels_file: string, allocator := context.allocator) -> (mnist: Mnist, ok: bool) {
	// The decompressed training images are ~47 MB of scratch; release them as
	// soon as they have been converted rather than holding them until whatever
	// the caller's next free_all happens to be.
	runtime.DEFAULT_TEMP_ALLOCATOR_TEMP_GUARD()

	pixels, image_count := _idx_read(images_file, 0x0000_0803) or_return
	labels, label_count := _idx_read(labels_file, 0x0000_0801) or_return

	if image_count != label_count {
		log.errorf("%v has %d images but %v has %d labels", images_file, image_count, labels_file, label_count)
		return
	}

	mnist.samples = image_count
	mnist.inputs  = make([]f32, image_count * MNIST_IMAGE_SIZE, allocator)
	mnist.targets = make([]int, image_count, allocator)

	for value, i in pixels {
		mnist.inputs[i] = f32(value) / 255.0
	}
	for value, i in labels {
		mnist.targets[i] = int(value)
	}
	return mnist, true
}

mnist_destroy :: proc(mnist: Mnist) {
	delete(mnist.inputs)
	delete(mnist.targets)
}
