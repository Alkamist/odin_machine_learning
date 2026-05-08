package machine_learning_backend_cuda

// Auto-graph capture/replay.
//
// `enable_decode_graph(true)` flips `Context.auto_graph_enabled`. After that:
//
//   clear()       -> begin stream capture (if not already)
//   <forward ops> -> kernel/memcpy launches are recorded into a Graph
//   buffer_get()  -> _auto_graph_finish: end capture, update-or-instantiate
//                    GraphExec, launch on stream. Then the existing
//                    StreamSynchronize + MemcpyDtoH read out the result.
//
// We rely on `cuGraphExecUpdate` to swap kernel-arg / memcpy-pointer values
// into the existing GraphExec when the topology is unchanged across forwards
// (the steady-state decode case). On topology change (e.g. prefill chunk
// of 64 tokens followed by decode chunks of 1) the update fails and we
// destroy + re-instantiate. That's a single instantiation per topology
// transition, which is cheap relative to a generation.

import "base:runtime"

import "bindings/cuda"

// End an in-flight capture, fold the captured Graph into `auto_exec`
// (updating in place if possible, re-instantiating otherwise), launch the
// resulting GraphExec on the stream, and clear the capture flag.
//
// Caller must hold `_gpu_mutex` and have already checked `gctx.auto_capturing`.
_auto_graph_finish :: proc(gctx: ^Context, loc: runtime.Source_Code_Location) {
	graph: cuda.Graph
	cuda.check(cuda.StreamEndCapture(gctx.stream, &graph), loc=loc)
	gctx.auto_capturing = false

	defer cuda.GraphDestroy(graph)

	if gctx.auto_exec != nil {
		info: cuda.GraphExecUpdateResultInfo
		r := cuda.GraphExecUpdate(gctx.auto_exec, graph, &info)
		if r != .SUCCESS || info.result != .SUCCESS {
			// Topology / function / parameter change we can't patch in place.
			// Throw away the old exec and rebuild from this graph.
			cuda.GraphExecDestroy(gctx.auto_exec)
			gctx.auto_exec = nil
		}
	}

	if gctx.auto_exec == nil {
		cuda.check(cuda.GraphInstantiateWithFlags(&gctx.auto_exec, graph, cuda.GRAPH_INSTANTIATE_DEFAULT), loc=loc)
	}

	cuda.check(cuda.GraphLaunch(gctx.auto_exec, gctx.stream), loc=loc)
}
