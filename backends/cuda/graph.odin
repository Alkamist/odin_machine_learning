package machine_learning_backend_cuda

import "base:runtime"

import "bindings/cuda"

_auto_graph_finish :: proc(gctx: ^Context, loc: runtime.Source_Code_Location) {
	graph: cuda.Graph
	cuda.check(cuda.StreamEndCapture(gctx.stream, &graph), loc=loc)
	gctx.auto_capturing = false

	defer cuda.GraphDestroy(graph)

	if gctx.auto_exec != nil {
		info: cuda.GraphExecUpdateResultInfo
		r := cuda.GraphExecUpdate(gctx.auto_exec, graph, &info)
		if r != .SUCCESS || info.result != .SUCCESS {
			cuda.GraphExecDestroy(gctx.auto_exec)
			gctx.auto_exec = nil
		}
	}

	if gctx.auto_exec == nil {
		cuda.check(cuda.GraphInstantiateWithFlags(&gctx.auto_exec, graph, cuda.GRAPH_INSTANTIATE_DEFAULT), loc=loc)
	}

	cuda.check(cuda.GraphLaunch(gctx.auto_exec, gctx.stream), loc=loc)
}
