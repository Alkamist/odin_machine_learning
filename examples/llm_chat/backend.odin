package example_chat

import ml   "../../"
import cpu  "../../backends/cpu"
import cuda "../../backends/cuda"

ML_BACKEND :: #config(ML_BACKEND, "cpu")
#assert(ML_BACKEND == "cpu" || ML_BACKEND == "cuda", "ML_BACKEND must be \"cpu\" or \"cuda\"")

@(require_results)
backend_create :: proc(options: Options) -> ^ml.Context {
	when ML_BACKEND == "cpu" {
		cpu.set_thread_count(options.threads)
		return cpu.context_create(options.cpu_arena)
	} else {
		return cuda.context_create(decode_graph=true)
	}
}

backend_destroy :: proc(ctx: ^ml.Context) {
	when ML_BACKEND == "cpu" {
		cpu.context_destroy(ctx)
	} else {
		cuda.context_destroy(ctx)
		cuda.device_destroy()
	}
}
