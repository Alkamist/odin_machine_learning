# Stdlib map — where to Grep before guessing

Run `odin root` to get `<root>`. Packages live under `<root>/base`, `<root>/core`, `<root>/vendor`. Always Grep the directory below to confirm a symbol's name and signature before writing it.

## `base/` — implicit, no import needed
- `builtin` — `len`, `cap`, `make`, `new`, `delete`, `append`, `copy`, `Maybe`, basic types.
- `intrinsics` — compiler intrinsics (atomics, SIMD primitives, type queries).
- `runtime` — `Context`, `Allocator`, `Source_Code_Location`, panic/assert plumbing.
- `sanitizer` — ASAN/TSAN/UBSAN hooks.

## `core/` — import as `"core:<name>"`

**Formatting & I/O**
- `fmt` — `printf`, `println`, `aprintf`, `tprintf`, `eprintln`. NOT Go's `fmt`.
- `io` — `Reader`, `Writer`, `Stream` interfaces.
- `bufio` — buffered readers/writers.
- `os` — files, env, args, exit. (Note: `os` is being modernized; check what's actually there.)

**Strings & text**
- `strings` — builders, split, trim, replace, clone.
- `strconv` — parse/format numbers.
- `unicode`, `unicode/utf8`, `unicode/utf16` — codepoint handling.
- `text/...` — table, scanner, edit, regex, i18n.

**Collections & algorithms**
- `slice` — sort, search, reverse, map operations on `[]T`.
- `sort` — comparator-based sorts.
- `container/...` — `queue`, `priority_queue`, `bit_array`, `small_array`, `intrusive`, `topological_sort`, `lru`.

**Memory**
- `mem` — `Arena`, `Tracking_Allocator`, `Scratch_Allocator`, `Dynamic_Pool`, alignment helpers.
- `mem/virtual` — virtual memory, page allocators.

**Math**
- `math` — scalar math, constants.
- `math/linalg` — vectors, matrices, quaternions (use this, not hand-rolled).
- `math/rand`, `math/big`, `math/noise`, `math/bits`, `math/cmplx`, `math/ease`, `math/fixed`.

**Concurrency**
- `sync` — mutexes, atomics, once, waitgroup-equivalents.
- `thread` — OS threads, thread pools.

**Encoding & data**
- `encoding/json`, `encoding/xml`, `encoding/csv`, `encoding/cbor`, `encoding/base64`, `encoding/hex`, `encoding/varint`, `encoding/ansi`.
- `compress` — `gzip`, `zlib`, `shoco`.
- `hash` — non-crypto hashes (xxhash, fnv, crc).
- `crypto` — crypto primitives (curves, hashes, AEAD).

**System & networking**
- `net` — sockets, TCP/UDP, DNS.
- `nbio` — non-blocking I/O event loop.
- `time` — clocks, durations, formatting.
- `sys/...` — raw OS bindings (windows, linux, darwin, posix). Prefer higher-level wrappers when available.
- `dynlib` — dynamic library loading.

**Tooling & introspection**
- `reflect` — runtime type info.
- `log` — leveled logging via `context.logger`.
- `flags` — CLI flag parsing.
- `testing` — `@(test)` procedures, run with `odin test`.
- `debug/...`, `prof/...` — debug info, profiling helpers.
- `odin/...` — Odin's own parser/tokenizer/ast/checker (for tooling).

**Misc**
- `image/...` — PNG, BMP, QOI, NetPBM, TGA decoders/encoders.
- `simd`, `simd/x86` — SIMD types and ops.
- `relative` — relative pointers.
- `terminal` — ANSI, terminal capabilities.
- `path/filepath`, `path/slashpath` — path manipulation.

## `vendor/` — third-party bindings, import as `"vendor:<name>"`

Graphics & windowing: `OpenGL`, `vulkan`, `wgpu`, `directx`, `glfw`, `sdl2`, `sdl3`, `egl`, `wasm`.
Game/media: `raylib`, `box2d`, `miniaudio`, `portmidi`, `ggpo`, `fontstash`, `nanovg`, `microui`, `kb_text_shape`.
Assets: `stb`, `cgltf`, `OpenEXRCore`, `compress`.
Net & data: `ENet`, `curl`, `commonmark`.
Scripting: `lua`.
Platform: `darwin`, `libc`, `libc-shim`.

When the user wants a binding not in `vendor/`, they will need to write one (Odin has `foreign` blocks for C interop) — don't invent imports.
