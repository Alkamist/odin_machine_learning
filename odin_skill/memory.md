# Memory management

If anything here contradicts the live install, the install wins. Source of truth:
- `<root>/base/runtime/core.odin` — `Allocator`, `Allocator_Mode`, `Allocator_Error`, `Context`.
- `<root>/core/mem/` — concrete allocators.
- `<root>/core/mem/virtual/` — page-backed arenas.

## Mental model: an allocator is a vtable

```odin
Allocator :: struct {
    procedure: Allocator_Proc,    // single proc that handles all modes
    data:      rawptr,            // allocator state (the Arena, the Tracking_Allocator, …)
}
```

Every allocator is just two pointers. The `procedure` dispatches on `Allocator_Mode`:

```
Alloc, Alloc_Non_Zeroed, Free, Free_All, Resize, Resize_Non_Zeroed,
Query_Features, Query_Info
```

Most allocators don't implement every mode — many arenas, for example, refuse `Free` of a single allocation but support `Free_All`. Errors come back as `Allocator_Error`:

```
None, Out_Of_Memory, Invalid_Pointer, Invalid_Argument, Mode_Not_Implemented
```

You almost never call the procedure directly — you call `make`, `new`, `delete`, `free`, `mem.alloc`, `mem.resize`, etc., which dispatch through the allocator.

## The implicit `context`

Every Odin call (with the default Odin calling convention) passes a `Context` struct. The two important fields are:

```odin
context.allocator       // used by `new`, `make`, `delete`, `free`, `append`, ...
context.temp_allocator  // used when you explicitly opt in (see below)
```

(There's also `context.logger`, `context.assertion_failure_proc`, `context.user_ptr`, `context.user_index` — the same plumbing.)

`context` is **lexically scoped**. A nested call inherits the current scope's context by value:

```odin
do_work :: proc() {
    context.allocator = my_arena_allocator   // affects this scope and its callees
    thing := make([]int, 100)                // uses my_arena_allocator
    helper()                                 // helper sees my_arena_allocator too
}                                            // context restored on scope exit

helper :: proc() { other := new(Foo) }       // uses the inherited allocator
```

To swap a single field for a single call, the idiom is:

```odin
{
    context.allocator = arena_alloc
    blob := make([]u8, 1<<20)
}   // back to whatever was there before
```

To copy the current context, mutate, and use:

```odin
c := context
c.allocator = my_alloc
context = c
```

`proc "c"` and other non-Odin conventions **do not carry context**. If you call back into Odin from C, you must do `context = runtime.default_context()` first, or `fmt`, `make`, etc. will dereference garbage.

## `new` / `make` / `delete` / `free` recap

Allocator-aware builtins. The allocator is the **last** parameter, defaulting to `context.allocator`.

```odin
// Single object:
p := new(Foo)                    // ^Foo, zeroed
p := new(Foo, my_allocator)
free(p)                          // pairs with new
free(p, my_allocator)            // explicit allocator

// Group of objects:
s := make([]int, 100)            // []int, zeroed
m := make(map[string]int)
xs := make([dynamic]u8, 0, 1024)
delete(s)                        // pairs with make for slice/map/[dynamic]/string
delete(m)
delete(xs)
```

Naming rule (from `core/mem/doc.odin`):

- **`new` ↔ `free`** — single object via pointer.
- **`make` ↔ `delete`** — group of objects (slice, map, dynamic array, string).

Don't cross the streams. `delete(p)` on a `^Foo` is wrong; `free(s)` on a `[]Foo` is wrong.

### Zero by default

Newly allocated memory is zeroed. For large buffers where you'll overwrite immediately, use the `*_non_zeroed` modes via `mem.alloc_non_zeroed`, or pass through allocators that support `Alloc_Non_Zeroed`. Skipping zeroing is a measurable win for multi-MB buffers.

### Size constants

`base:runtime` exports `Byte`, `Kilobyte`, `Megabyte`, `Gigabyte`, `Terabyte`, `Petabyte`, `Exabyte` (powers of 1024). Use them — `1 * mem.Megabyte` reads better than `1024*1024`.

## The temp allocator pattern

`context.temp_allocator` is an additional allocator slot meant for short-lived allocations that get freed in bulk. The default temp allocator is a per-thread arena.

You **opt in** explicitly — none of the standard builtins reach for the temp allocator unless told:

```odin
buf := make([]u8, 1024, context.temp_allocator)

// `fmt` has dedicated `tprintf` / `tprint` / `taprintf` variants that allocate into
// the temp allocator and return strings:
msg := fmt.tprintf("frame %d at %v", frame, time.now())   // freed with temp_allocator
```

At a logical boundary (end of frame, end of request, end of test), free the lot:

```odin
free_all(context.temp_allocator)
```

This is the cheapest possible cleanup: O(1), no individual frees, no fragmentation. Don't `delete` individual temp allocations — let the `free_all` do it.

## Allocators in `core:mem`

Pattern is consistent: every allocator type has a `*_init` proc, a `*_destroy` proc, and an `<name>_allocator(&x) -> Allocator` accessor that returns the vtable to plug into `context.allocator`.

| Type | When to use |
|---|---|
| `Arena` | Linear bump allocator over a fixed buffer. No individual `Free`; `Free_All` resets. The classic per-frame allocator. |
| `Scratch` | Ring of arenas. Allocations older than the ring are recycled. Good for ephemeral working buffers without explicit `Free_All`. |
| `Stack` / `Small_Stack` | LIFO bump allocator with markers. Free in reverse order, like a real stack. |
| `Dynamic_Arena` | Arena over a chain of growable backing buffers. Useful when you want arena semantics but don't know the size upfront and don't have access to virtual memory. |
| `Buddy_Allocator` | General-purpose, supports per-allocation `Free`. Higher overhead than an arena. |
| `Mutex_Allocator` | Wraps another allocator with a mutex for thread-safe sharing. |
| `Compat_Allocator` (a.k.a. rollback stack) | General-purpose allocator that supports rollback to a saved point. See `rollback_stack_allocator.odin`. |
| `Tracking_Allocator` | Wraps another allocator and records every `Alloc`/`Free`. Used for leak detection. See section below. |
| `nil_allocator` | All operations return `Mode_Not_Implemented`. Use it to assert that a region of code must not allocate. |
| `panic_allocator` | All operations panic. Same intent as `nil_allocator` but louder. |

For each, look up the actual struct fields and init signature in `core:mem` before using — Grep for `*_init` on the type.

## Page-backed arenas: `core:mem/virtual`

`core:mem/virtual` uses OS virtual memory (reserve / commit / decommit) to provide arenas that can be huge without paying for memory you don't use. Three variants of `virtual.Arena`:

- **`.Growing`** — chained memory blocks; the arena grows by reserving more virtual address space when needed.
- **`.Static`** — single block, fixed reservation (default 1 GiB on 64-bit). Commits pages on demand. Best when you have a known upper bound and want a single contiguous region.
- **`.Buffer`** — single block backed by a user-provided `[]byte`. No virtual memory at all; fits where you'd otherwise use `mem.Arena`.

Init with the matching proc: `virtual.arena_init_growing`, `arena_init_static`, `arena_init_buffer`. Wrap as an `Allocator` with `virtual.arena_allocator(&arena)`. Reset with `virtual.arena_free_all(&arena)`. Tear down with `virtual.arena_destroy(&arena)`.

For most production game/server code, **`virtual.Arena` (`.Growing` or `.Static`) is the right default** for the per-frame / per-request scope.

## Tracking allocator — leak detection

Standard pattern in main during development:

```odin
import "core:fmt"
import "core:mem"
import "core:os"

main :: proc() {
    when ODIN_DEBUG {
        track: mem.Tracking_Allocator
        mem.tracking_allocator_init(&track, context.allocator)
        defer mem.tracking_allocator_destroy(&track)
        context.allocator = mem.tracking_allocator(&track)

        defer {
            if len(track.allocation_map) > 0 {
                fmt.eprintfln("=== %v leaked allocations ===", len(track.allocation_map))
                for _, entry in track.allocation_map {
                    fmt.eprintfln("  %v bytes at %v", entry.size, entry.location)
                }
            }
            if len(track.bad_free_array) > 0 {
                fmt.eprintfln("=== %v bad frees ===", len(track.bad_free_array))
                for entry in track.bad_free_array {
                    fmt.eprintfln("  %p at %v", entry.memory, entry.location)
                }
            }
        }
    }

    run()
}
```

This catches leaks (allocations never freed) and bad frees (freeing pointers the allocator doesn't own, double frees) at program exit. Wrap **each thread's** allocator the same way; `Tracking_Allocator` has its own mutex.

## Ownership conventions

Odin has no borrow checker. By convention:

- **Allocator (`make`/`new`) → caller of the proc that returned the value owns it** unless a comment says otherwise. Pair the call with `defer delete(...)` at the *same scope* where ownership begins.
- **Procedures that take a slice or pointer typically borrow it** — they read/write through the borrow but do not free it. If a proc takes ownership (will free or store the pointer), document it (`// takes ownership of `data``).
- **For long-lived structures, store the allocator inside the struct**, like `Tracking_Allocator` and `[dynamic]T` do. The destroy proc uses that allocator to free internal state.
- **An allocator passed in is borrowed for the lifetime of the call**, unless the proc explicitly stores it.

When a struct holds heap-owned fields, mirror the lifecycle:

```odin
World :: struct {
    entities: [dynamic]Entity,
    name:     string,                // owned (cloned)
    allocator: mem.Allocator,
}

world_init :: proc(w: ^World, name: string, allocator := context.allocator) {
    w.allocator = allocator
    w.entities  = make([dynamic]Entity, 0, 64, allocator)
    w.name      = strings.clone(name, allocator)
}

world_destroy :: proc(w: ^World) {
    delete(w.entities, w.allocator)
    delete(w.name,     w.allocator)
    w^ = {}
}
```

## Strings and allocation

| Form | Allocated? | Free with |
|---|---|---|
| `"literal"` | No (static data) | — never `delete` |
| `string(byte_slice)` | No (view) | depends on the slice |
| `strings.clone(s)` | Yes | `delete(cloned)` |
| `fmt.aprintf(...)` | Yes (`context.allocator`) | `delete(result)` |
| `fmt.tprintf(...)` | Yes (`context.temp_allocator`) | `free_all(temp_allocator)` |
| `strings.concatenate({...})` | Yes | `delete(result)` |
| `strings.Builder` | Yes (internal `[dynamic]u8`) | `strings.builder_destroy(&b)` |

**Never `delete` a string literal.** It will not crash on most allocators, but it's a logical bug — and a `Tracking_Allocator` will flag it.

## Common pitfalls (Go-isms and others)

- `append(&xs, x)` not `xs = append(xs, x)`. `xs` is a `[dynamic]T`, mutated in place.
- `delete_key(&m, k)` not `delete(m, k)`. `delete(m)` frees the **whole map**.
- `defer delete(...)` belongs at the same scope as the allocation, not at the end of `main`. Match acquisition and release.
- A procedure that calls `make`/`new` internally and returns the result is transferring ownership — the caller now owes a `delete`/`free`. Document this.
- `context.allocator = X` mutates a *copy* of `Context` for the current scope; setting it inside a block does not leak to the caller.
- Tracking allocator's internal map is allocated using `internals_allocator` (defaults to `context.allocator` at init time). Make sure that allocator is something **outside** the tracked one — otherwise the tracker tracks itself.
- `proc "c"` callbacks have no `context`. Set one up with `context = runtime.default_context()` before any allocation, logging, or `fmt`.
- `temp_allocator` is per-thread by default. Don't share temp-allocated values across threads, and don't keep them past the next `free_all`.
- `make([dynamic]T, 0, cap)` — second arg is **length**, third is capacity. Got that backwards once and it cost an afternoon.
