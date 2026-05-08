package cuda_bindings_smoke

// Smoke test for the CUDA driver + NVRTC bindings.
// Compiles a trivial vector_add kernel at runtime, runs it, and verifies the
// output. Proves the bindings link and the driver/NVRTC dlls are reachable
// before we build any backend abstractions on top.

import "core:fmt"
import "core:os"

import "../../backends/cuda/bindings/cuda"
import "../../backends/cuda/bindings/nvrtc"

KERNEL_SRC: cstring : `
extern "C" __global__ void vector_add(const float* a, const float* b, float* c, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) c[i] = a[i] + b[i];
}
`

main :: proc() {
	fail :: proc(msg: string) -> ! {
		fmt.eprintln("FAIL:", msg)
		os.exit(1)
	}

	cuda.check(cuda.Init(0))

	driver_version: i32
	cuda.check(cuda.DriverGetVersion(&driver_version))
	fmt.printfln("CUDA driver version: %d", driver_version)

	device_count: i32
	cuda.check(cuda.DeviceGetCount(&device_count))
	if device_count <= 0 { fail("no CUDA devices found") }

	dev: cuda.Device
	cuda.check(cuda.DeviceGet(&dev, 0))

	name_buf: [128]u8
	cuda.check(cuda.DeviceGetName(raw_data(name_buf[:]), i32(len(name_buf)), dev))
	cc_major, cc_minor, sm_count: i32
	cuda.check(cuda.DeviceGetAttribute(&cc_major, .COMPUTE_CAPABILITY_MAJOR, dev))
	cuda.check(cuda.DeviceGetAttribute(&cc_minor, .COMPUTE_CAPABILITY_MINOR, dev))
	cuda.check(cuda.DeviceGetAttribute(&sm_count, .MULTIPROCESSOR_COUNT, dev))
	fmt.printfln("device: %s  cc=%d.%d  SMs=%d",
		cstring(raw_data(name_buf[:])), cc_major, cc_minor, sm_count)

	ctx: cuda.Context
	cuda.check(cuda.CtxCreate(&ctx, 0, dev))
	defer cuda.CtxDestroy(ctx)

	// NVRTC: compile vector_add to cubin for the actual device's compute
	// capability. cubin avoids ptxjit at module load time, which we want for
	// fast startup.
	prog: nvrtc.Program
	nvrtc.check(nvrtc.CreateProgram(&prog, KERNEL_SRC, "vector_add.cu", 0, nil, nil))
	defer nvrtc.DestroyProgram(&prog)

	arch_buf: [32]u8
	arch_str := fmt.bprintf(arch_buf[:], "--gpu-architecture=sm_%d%d", cc_major, cc_minor)
	arch_buf[len(arch_str)] = 0
	options := [?]cstring{
		cstring(raw_data(arch_buf[:])),
		"--use_fast_math",
		"-default-device",
	}

	if r := nvrtc.CompileProgram(prog, i32(len(options)), raw_data(options[:])); r != .SUCCESS {
		log_size: uint
		nvrtc.check(nvrtc.GetProgramLogSize(prog, &log_size))
		log := make([]u8, log_size)
		nvrtc.check(nvrtc.GetProgramLog(prog, raw_data(log)))
		fmt.eprintln(string(log))
		fail("nvrtc compile failed")
	}

	cubin_size: uint
	nvrtc.check(nvrtc.GetCUBINSize(prog, &cubin_size))
	cubin := make([]u8, cubin_size); defer delete(cubin)
	nvrtc.check(nvrtc.GetCUBIN(prog, raw_data(cubin)))
	fmt.printfln("cubin size: %d bytes", cubin_size)

	module: cuda.Module
	cuda.check(cuda.ModuleLoadData(&module, raw_data(cubin)))
	defer cuda.ModuleUnload(module)

	kernel: cuda.Function
	cuda.check(cuda.ModuleGetFunction(&kernel, module, "vector_add"))

	N :: 1 << 20
	bytes := uint(N * size_of(f32))

	host_a := make([]f32, N); defer delete(host_a)
	host_b := make([]f32, N); defer delete(host_b)
	host_c := make([]f32, N); defer delete(host_c)
	for i in 0..<N {
		host_a[i] = f32(i)
		host_b[i] = f32(2 * i)
	}

	d_a, d_b, d_c: cuda.DevicePtr
	cuda.check(cuda.MemAlloc(&d_a, bytes)); defer cuda.MemFree(d_a)
	cuda.check(cuda.MemAlloc(&d_b, bytes)); defer cuda.MemFree(d_b)
	cuda.check(cuda.MemAlloc(&d_c, bytes)); defer cuda.MemFree(d_c)

	cuda.check(cuda.MemcpyHtoD(d_a, raw_data(host_a), bytes))
	cuda.check(cuda.MemcpyHtoD(d_b, raw_data(host_b), bytes))

	BLOCK :: 256
	grid := u32((N + BLOCK - 1) / BLOCK)
	n_arg: i32 = N
	args := [?]rawptr{ &d_a, &d_b, &d_c, &n_arg }

	cuda.check(cuda.LaunchKernel(
		kernel,
		grid, 1, 1,
		BLOCK, 1, 1,
		0, nil,
		raw_data(args[:]), nil,
	))
	cuda.check(cuda.CtxSynchronize())

	cuda.check(cuda.MemcpyDtoH(raw_data(host_c), d_c, bytes))

	mismatches := 0
	for i in 0..<N {
		expected := f32(3 * i)
		if host_c[i] != expected {
			mismatches += 1
			if mismatches <= 4 {
				fmt.eprintfln("  mismatch at %d: got %v want %v", i, host_c[i], expected)
			}
		}
	}
	if mismatches != 0 {
		fail(fmt.tprintf("%d mismatches out of %d", mismatches, N))
	}

	fmt.printfln("OK  vector_add of %d floats produced expected sums", N)
}
