# Errors

Odin has **no exceptions**. Errors are values returned from procedures, like Go — but the syntax for propagating them is more compact (`or_return`), and the convention for typing them is richer (enums and tagged unions, not stringly-typed).

If something here contradicts the live install, the install wins. Source of truth:
- `<root>/base/runtime/core.odin` — `Allocator_Error`.
- `<root>/core/io/io.odin:17` — canonical `Error :: enum i32 { None, EOF, ... }` shape.
- `<root>/core/os/errors.odin:39` — canonical `Error :: union #shared_nil { ... }` shape.
- `<root>/examples/demo/demo.odin:2049-2237` — `or_else`, `or_return`, `or_break`, `or_continue` examples.

## The basics: errors are positional return values

```odin
read_line :: proc(r: ^Reader) -> (line: string, err: io.Error) {
    ...
}

line, err := read_line(r)
if err != nil {
    // handle
}
```

Multiple returns are **positional, not a tuple type**. You can't pass them around as a single value, and you can't destructure further than the call site.

By convention, **the error is the last return value**. `or_return` and `or_else` rely on this — they pop the last value off and inspect it.

## Naming the error type

Two patterns dominate.

**(1) An enum.** Cheap, exhaustive in `switch`, fits in a register.

```odin
// From core:io
Error :: enum i32 {
    None = 0,
    EOF,
    Unexpected_EOF,
    Short_Write,
    Invalid_Write,
    Short_Buffer,
    No_Progress,
    Invalid_Whence,
    ...
}
```

`None = 0` is essential — it makes the zero value mean "no error" so a fresh-zeroed return slot is already correct. **Always** put `None = 0` (explicit `= 0`) at the top of an error enum.

**(2) A union of enums** for procs that can fail in multiple categories.

```odin
// From core:os
Error :: union #shared_nil {
    General_Error,             // an enum
    io.Error,                  // another enum
    runtime.Allocator_Error,   // another enum
    Platform_Error,            // a distinct integer wrapper
}
```

`#shared_nil` is the key directive: it makes `nil` mean "no error" regardless of which variant was most recently set. Without it, an `Error` holding a `General_Error.None` would still be a non-nil union. With it, `err == nil` works the way you want.

Returning any of the variant types from a proc that returns `Error` triggers an automatic widening conversion — you can `return io.Error.EOF` from a proc returning `os.Error`.

## `or_return` — propagate

The bread-and-butter operator. It pops the last value off a multi-valued expression, checks it for `nil` / `false`, and if non-nil/non-false, returns from the current procedure with that value placed in the last return slot.

```odin
// Common idiom:
n0, err := read_thing()
if err != nil { return err }

// With or_return:
n0 := read_thing() or_return
```

Rules:

- The last return value of the **current** procedure must be assignable from the popped value.
- If the current procedure returns 2+ values, **all returns must be named** so that `or_return` can do a bare `return` after assigning to the last named return.
- For a one-valued expression returning just an error, `or_return` does a simple early return.

Worked examples (verbatim from `demo.odin:2107-2169`):

```odin
caller_1 :: proc() -> Error                  { return .None }
caller_2 :: proc() -> (int, Error)           { return 123, .None }
caller_3 :: proc() -> (int, int, Error)      { return 123, 345, .None }

// Single-error return — no need for named returns.
foo_1 :: proc() -> Error {
    n1 := caller_2() or_return        // returns from foo_1 if caller_2's err != nil
    caller_1() or_return               // 1-valued, early return on non-nil
    n0, n1 := caller_3() or_return     // last value popped, first two assigned
    return .None
}

// Multi-valued return — names required for `or_return` to work with a bare return.
foo_2 :: proc() -> (n: int, err: Error) {
    y := caller_2() or_return          // assigns err and returns on failure
    caller_1() or_return

    // You can still mix in explicit returns when needed:
    if z, zerr := caller_2(); zerr != nil {
        return -345 * z, zerr
    }

    // `defer if err != nil { ... }` is a clean way to log / clean up on error paths.
    defer if err != nil {
        fmt.println("Error in", #procedure, ":", err)
    }

    n = 123
    return
}
```

## `or_else` — fallback value

For expressions that return `(value, ok)` or an optional. Provides a default when the operation says "no value".

```odin
m: map[string]int
i := m["hellope"] or_else 123          // map lookup miss → 123

v: union { int, f64 }
i := v.(int) or_else 123               // type assertion miss → 123
i  = v.? or_else 123                   // type-inferred form

m: Maybe(int)
i  = m.? or_else 456                   // nil Maybe → 456
```

`or_else` works with anything that has the `(T, ok: bool)` shape: map indexing, type assertions, `Maybe` unwrapping, `pop_safe`, `strconv.parse_*`, etc.

## `or_break` / `or_continue` — for loops

Like `or_return`, but exits the loop instead of the procedure. The popped value is **discarded** — these are for cases where the error doesn't propagate further.

```odin
for {
    caller_1() or_break                 // break on non-nil
}

for {
    n := caller_2() or_break            // value bound, error breaks
    _ = n
}

loop: for {
    n := caller_2() or_break loop       // labelled break — exits the named loop
    _ = n
}

for {
    x, y := caller_3() or_continue      // skip iteration on error
    _, _ = x, y
    break
}
```

Use these when an error in one iteration shouldn't kill the procedure but should abort/skip the loop. If you want to propagate, use `or_return` instead.

## `Maybe(T)` is not for errors

`Maybe(T)` (or `?T`) is the optional-value type. Use it when "no value" is a valid, non-exceptional state — a cache miss, an unset config field, an `i32 -> Maybe(Color)` parser that legitimately accepts "absent".

Don't use `Maybe(T)` to communicate failure that the caller might want to inspect. Errors carry information; `Maybe(T)` carries none beyond "present / absent". For failure, return `(T, Error)`.

A common combo: an internal helper returns `Maybe(T)` for absence; the public proc wraps it as `(T, Error)` when absence is exceptional.

## `panic`, `assert`, `unreachable` — programmer errors only

These abort the program. Use them for **invariant violations** and **bugs**, never for conditions a caller might recover from.

```odin
panic("invariant broken: refcount went negative")     // unconditional
assert(cap(buf) >= n)                                  // condition or panic
assert(cap(buf) >= n, "buffer too small for write")    // with message

unreachable()      // marker for control-flow positions the compiler can't prove dead
```

`assert` runs in all builds by default. To strip in release, build with `-no-bounds-check -disable-assert` or guard with `when ODIN_DEBUG`.

For a condition the caller chose — bad input, missing file, network failure — return an error. Reserve `panic`/`assert` for "this should be impossible".

## Don't swallow errors

```odin
_, _ = thing_that_can_fail()            // bad — silent failure
```

If you genuinely don't care, comment why:

```odin
// We deliberately ignore the close error; we already wrote everything we needed.
_ = os.close(f)
```

The static analyzer (`odin check`, `-vet`) will warn about discarded errors when it can prove the value carries error semantics.

## Quick reference

| Want | Write |
|---|---|
| Propagate error early | `x := foo() or_return` |
| Propagate, single-valued expr | `foo() or_return` |
| Provide default value | `x := lookup() or_else default` |
| Break loop on error | `x := foo() or_break` (optionally `or_break label`) |
| Skip iteration on error | `x := foo() or_continue` |
| Unwrap `Maybe(T)` (panic on nil) | `x := m.?` |
| Unwrap `Maybe(T)` safely | `x, ok := m.?` |
| Type-assert (panic on miss) | `x := v.(T)` |
| Type-assert safely | `x, ok := v.(T)` |
| Test invariant (kill on fail) | `assert(cond)` |
| Mark dead branch | `unreachable()` |
| Define an error enum | `Error :: enum { None = 0, ... }` |
| Define a multi-source error | `Error :: union #shared_nil { ... }` |
