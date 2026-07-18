// NVRTC bindings (runtime CUDA compilation).
//
// Compiles `.cu` source to PTX or cubin in-process so kernels can be embedded
// with `#load` and built at startup, the same shape the vulkan backend uses
// for SPIR-V.
//
// nvrtc.lib is vendored under ../lib/. The corresponding nvrtc64_120_0.dll
// must be on PATH (it ships in `<CUDA_PATH>\bin` for CUDA 12.x).
//
// Naming follows the Odin vendor convention: PascalCase mirroring the C
// names with the `nvrtc` prefix stripped via `link_prefix`.
package nvrtc

import "core:fmt"

when ODIN_OS == .Windows {
	foreign import lib "../lib/nvrtc.lib"
} else {
	foreign import lib "system:nvrtc" // libnvrtc.so; -L supplies its location
}

Program :: distinct rawptr

Result :: enum i32 {
	SUCCESS                           = 0,
	OUT_OF_MEMORY                     = 1,
	PROGRAM_CREATION_FAILURE          = 2,
	INVALID_INPUT                     = 3,
	INVALID_PROGRAM                   = 4,
	INVALID_OPTION                    = 5,
	COMPILATION                       = 6,
	BUILTIN_OPERATION_FAILURE         = 7,
	NO_NAME_EXPRESSIONS_AFTER_COMPILE = 8,
	NO_LOWERED_NAMES_BEFORE_COMPILE   = 9,
	NAME_EXPRESSION_NOT_VALID         = 10,
	INTERNAL_ERROR                    = 11,
	TIME_FILE_WRITE_FAILED            = 12,
}

@(default_calling_convention="c", link_prefix="nvrtc")
foreign lib {
	Version        :: proc(major, minor: ^i32) -> Result ---
	GetErrorString :: proc(result: Result) -> cstring ---

	CreateProgram :: proc(
		prog: ^Program,
		src:  cstring,
		name: cstring,
		num_headers: i32,
		headers: [^]cstring,
		include_names: [^]cstring,
	) -> Result ---

	DestroyProgram :: proc(prog: ^Program) -> Result ---

	CompileProgram :: proc(
		prog: Program,
		num_options: i32,
		options: [^]cstring,
	) -> Result ---

	GetPTXSize   :: proc(prog: Program, size: ^uint) -> Result ---
	GetPTX       :: proc(prog: Program, out: [^]u8) -> Result ---
	GetCUBINSize :: proc(prog: Program, size: ^uint) -> Result ---
	GetCUBIN     :: proc(prog: Program, out: [^]u8) -> Result ---

	GetProgramLogSize :: proc(prog: Program, size: ^uint) -> Result ---
	GetProgramLog     :: proc(prog: Program, out: [^]u8) -> Result ---
}

check :: proc(r: Result, loc := #caller_location) {
	if r == .SUCCESS { return }
	msg := GetErrorString(r)
	s := msg != nil ? string(msg) : "?"
	fmt.panicf("NVRTC error (%d): %s", i32(r), s, loc=loc)
}
