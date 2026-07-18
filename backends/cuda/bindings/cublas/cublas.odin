// cuBLAS bindings, scoped to what we need for BF16 / F32 GEMM and the
// dispatch helpers used in the linear forward/backward path. Mirrors the
// llama.cpp/ggml-cuda use of `cublasGemmEx` rather than the more verbose
// cublasLt matmul setup.
//
// cublas.lib is vendored under ../lib/. The corresponding cublas64_12.dll
// must be on PATH (it ships in `<CUDA_PATH>\bin` for CUDA 12.x).
//
// Naming follows the Odin vendor convention: PascalCase mirroring the C
// names with the `cublas` prefix stripped via `link_prefix`.
package cublas

import "core:fmt"

import cu "../cuda"

when ODIN_OS == .Windows {
	foreign import lib "../lib/cublas.lib"
} else {
	foreign import lib "system:cublas" // libcublas.so; -L supplies its location
}

Handle :: distinct rawptr

Status :: enum i32 {
	SUCCESS          = 0,
	NOT_INITIALIZED  = 1,
	ALLOC_FAILED     = 3,
	INVALID_VALUE    = 7,
	ARCH_MISMATCH    = 8,
	MAPPING_ERROR    = 11,
	EXECUTION_FAILED = 13,
	INTERNAL_ERROR   = 14,
	NOT_SUPPORTED    = 15,
	LICENSE_ERROR    = 16,
}

// Matrix transpose options (`cublasOperation_t`).
Operation :: enum i32 {
	N = 0,  // no transpose
	T = 1,  // transpose
	C = 2,  // conjugate transpose (ignored for real types)
}

// Element data type. Defined by `cudaDataType_t` in library_types.h; we
// inline the few we use here.
DataType :: enum i32 {
	R_32F  = 0,
	R_64F  = 1,
	R_16F  = 2,
	R_16BF = 14,
	R_8I   = 3,
	R_8U   = 8,
	R_32I  = 10,
}

// Compute precision (`cublasComputeType_t`). `_32F` is the default we use:
// inputs are BF16, accumulation runs in FP32 to keep training-grade accuracy.
ComputeType :: enum i32 {
	_16F           = 64,
	_16F_PEDANTIC  = 65,
	_32F           = 68,
	_32F_PEDANTIC  = 69,
	_32F_FAST_16F  = 74,
	_32F_FAST_16BF = 75,
	_32F_FAST_TF32 = 77,
	_64F           = 70,
	_64F_PEDANTIC  = 71,
	_32I           = 72,
	_32I_PEDANTIC  = 73,
}

// Algorithm selector. Only DEFAULT is needed; the heuristics inside cuBLAS
// pick a tensor-core-using algo automatically when inputs allow it.
GemmAlgo :: enum i32 {
	DEFAULT = -1,
}

// Pointer mode (`cublasPointerMode_t`). HOST means alpha/beta are read from
// the calling thread; we always pass them by host pointer.
PointerMode :: enum i32 {
	HOST   = 0,
	DEVICE = 1,
}

@(default_calling_convention="c", link_prefix="cublas")
foreign lib {
	Create_v2          :: proc(handle: ^Handle) -> Status ---
	Destroy_v2         :: proc(handle: Handle) -> Status ---
	GetVersion_v2      :: proc(handle: Handle, version: ^i32) -> Status ---
	SetStream_v2       :: proc(handle: Handle, stream: cu.Stream) -> Status ---
	GetStream_v2       :: proc(handle: Handle, stream: ^cu.Stream) -> Status ---
	GetStatusName      :: proc(status: Status) -> cstring ---
	GetStatusString    :: proc(status: Status) -> cstring ---
	SetPointerMode_v2  :: proc(handle: Handle, mode: PointerMode) -> Status ---

	// Mixed-precision GEMM. With A_type = B_type = R_16BF and compute_type =
	// _32F you get Ampere BF16 tensor cores; C_type can be R_16BF (in-place
	// downcast) or R_32F.
	GemmEx :: proc(
		handle:        Handle,
		transa, transb: Operation,
		m, n, k:       i32,
		alpha:         rawptr,
		A:             rawptr, A_type: DataType, lda: i32,
		B:             rawptr, B_type: DataType, ldb: i32,
		beta:          rawptr,
		C:             rawptr, C_type: DataType, ldc: i32,
		compute_type:  ComputeType,
		algo:          GemmAlgo,
	) -> Status ---

	// Strided batched GEMM. `stride_*` is the element stride between
	// consecutive matrices in the batch. Used for batched_matmul / attention.
	GemmStridedBatchedEx :: proc(
		handle:         Handle,
		transa, transb: Operation,
		m, n, k:        i32,
		alpha:          rawptr,
		A:              rawptr, A_type: DataType, lda: i32, stride_A: i64,
		B:              rawptr, B_type: DataType, ldb: i32, stride_B: i64,
		beta:           rawptr,
		C:              rawptr, C_type: DataType, ldc: i32, stride_C: i64,
		batch_count:    i32,
		compute_type:   ComputeType,
		algo:           GemmAlgo,
	) -> Status ---
}

check :: proc(s: Status, loc := #caller_location) {
	if s == .SUCCESS { return }
	name := GetStatusName(s)
	desc := GetStatusString(s)
	n := name != nil ? string(name) : "?"
	d := desc != nil ? string(desc) : "?"
	fmt.panicf("cuBLAS error: %s (%d): %s", n, i32(s), d, loc=loc)
}
