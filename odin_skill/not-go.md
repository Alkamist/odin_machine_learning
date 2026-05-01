# Odin is not Go

If you find yourself writing something here that contradicts the live Odin install, the install wins — Grep `<odin-root>/core` to confirm.

## Imports & packages

Odin uses **library collections**, not URL-style import paths.

```odin
package main

import "core:fmt"            // standard library
import "core:math/linalg"    // sub-package
import "base:runtime"        // implicit base library (rarely needed explicitly)
import "vendor:raylib"       // third-party bindings shipped with the compiler
import "shared:my_lib"       // custom collection, mapped via -collection:shared=path
import "./util"              // sibling directory, relative to the importing file
```

The local name defaults to the last path segment (`fmt`, `linalg`, `raylib`). Alias with `import alias "...":`

```odin
import dt "core:time/datetime"
import win "core:sys/windows"
```

**Package rules — different from Go:**

- A package is a *directory*, not a file. Every `.odin` file in the directory must start with the same `package <name>` declaration. Mixing package names in one directory is an error.
- The package name does **not** have to match the directory name — though by convention it usually does. The import path uses the directory; the local identifier comes from the `package` declaration (or the alias).
- Executables must contain `package main` and a `main :: proc() { ... }`.
- There is no `go.mod` equivalent. Dependency management is filesystem + the `-collection:NAME=PATH` compiler flag.
- Imported names are **always** referenced through the package qualifier (`fmt.println`). There is no dot-import / unqualified import.
## Procedures

Declaration uses the `name :: proc(...) -> Ret { ... }` constant-binding form, not Go's `func`.

```odin
add :: proc(a, b: int) -> int { return a + b }

// Multiple return values — positional, no tuple type:
divmod :: proc(a, b: int) -> (q, r: int) {
    return a / b, a % b
}

// Named returns + naked `return`:
sum :: proc(xs: []int) -> (total: int) {
    for x in xs { total += x }
    return
}
```

**No methods.** Odin has no receiver syntax. There is no `func (r *Receiver) Foo()`. Just write `foo :: proc(self: ^Thing, ...)` and call it as `foo(&thing, ...)`. By convention, prefix the procedure with the type name: `entity_update`, `entity_destroy`.

**Default args, named args, variadics, spread** — verified from `examples/demo/demo.odin:472-488`:

```odin
sum :: proc(nums: ..int, init_value := 0) -> (result: int) {
    result = init_value
    for n in nums { result += n }
    return
}

sum()                          // 0
sum(1, 2, 3)                   // 6
sum(1, 2, init_value = 5)      // 8       (named arg)
odds := []int{1, 3, 5}
sum(..odds)                    // 9       (spread a slice as varargs)
sum(..odds, init_value = 5)    // 14
```

**Procedures are first-class values.** They can be stored, passed, and nested inside other procedures. Procedure *types* are written `proc(int, int) -> int`.

**Overloading is explicit, via overload groups** — Odin does not auto-overload by signature. You bundle distinct procedures into one name:

```odin
add :: proc{add_ints, add_floats, add_numbers}
```

**Calling conventions** matter — Odin's default convention implicitly threads `context` (for allocators, logger, user data). Other conventions don't.

```odin
proc(...)              // default Odin convention, carries `context`
proc "contextless"     // Odin ABI, no context — use for hot inner loops
proc "c"               // C ABI — required for callbacks passed to C libraries
proc "stdcall"         // Win32 callbacks
proc "naked"           // no prologue/epilogue
```

If you call a `proc "c"` from Odin code, you cannot use `context.allocator`, `fmt.println`, etc. inside it — they assume the implicit context exists. Wrap with `context = runtime.default_context()` if you need to.

**Polymorphic (generic) procs** use `$` to introduce a type parameter:

```odin
print_value :: proc(value: $T)            { /* T inferred */ }
add         :: proc(p, q: $T) -> T        { return p + q }
alloc_type  :: proc($T: typeid) -> ^T     { /* T passed explicitly */ }
copy_slice  :: proc(dst, src: []$T) -> int { /* constrains both args to same []T */ }
```

Type specialization constrains the parameter:

```odin
make_slice :: proc($T: typeid/[]$E, len: int) -> T { return make(T, len) }
allocate   :: proc(table: ^$T/Table, capacity: int) { ... }
```

(Generics are covered in more depth in the type-system section below.)
## Control flow

**`for` is the only loop keyword.** No `while`, no `do-while`, no `loop`. All four shapes use `for`:

```odin
for i := 0; i < 10; i += 1 { ... }   // C-style
for i < 10 { ... }                    // while-style (drop the semicolons)
for { break }                         // infinite
for x in some_collection { ... }      // range-over
```

No parens around the header. Body braces or `do <stmt>` are required — there is no implicit single-statement form like C's `if (c) x++;`.

**Range syntax — two operators, both work in `for` and `switch`:**

```odin
for j in 0..<10 { ... }   // half-open: 0, 1, ..., 9
for j in 0..=9  { ... }   // closed:    0, 1, ..., 9
```

**Range-over works on string, [N]T, []T, [dynamic]T, map, and custom iterators.** Strings iterate by **UTF-8 codepoint** (rune), not byte:

```odin
for character in "Hello, 世界" { ... }   // character: rune

for value in slice            { ... }
for value, index in slice     { ... }   // 2-value form: value + index
for key in some_map           { ... }
for key, value in some_map    { ... }
```

**Iterated values are copies.** To mutate in place, prefix the binding with `&`:

```odin
for &pixel in pixels {     // pixel is ^T, dereference auto-applied
    pixel.r = 0
}
```

(The older idiom `for _, idx in xs { xs[idx] = ... }` is still common in core code.)

**`if` and `switch` accept an init statement** (like Go), separated by `;`:

```odin
if y := compute(); y < 0 { ... } else if y == 0 { ... } else { ... }

switch arch := ODIN_ARCH; arch {
case .i386:  ...
case .amd64: ...
case:        ...   // default — empty case header, NOT `default:`
}
```

**`switch` differences from C/Go:**

- Cases do **not** fall through by default (same as Go). Use `fallthrough` to opt in.
- Cases accept **multiple values and ranges**: `case 'A'..='Z', 'a'..='z', '0'..='9':`.
- A switch with no expression is `switch true` — used as a clean if/else chain.
- For enums and unions, the compiler enforces exhaustiveness. Prefix with `#partial` to opt out: `#partial switch arch { ... }`.
- Type switch on a `union` or `any`: `switch v in val { case int: ...; case bool: ... }`.

**`defer` runs at end of *scope*** (not function), in reverse declaration order. Defers entire blocks too:

```odin
defer fmt.println("1")
defer fmt.println("2")
defer fmt.println("3")   // prints 3, 2, 1 at scope exit

defer {
    cleanup_a()
    cleanup_b()
}

defer if cond { bar() }
```

Pair every owning allocation with a `defer delete(...)` or `defer free(...)` at the same scope where ownership begins, unless ownership is transferred out.

**`when` is compile-time `if`.** Conditions must be constant expressions; only the chosen branch is type-checked. Allowed at file scope. This is Odin's `#ifdef`.

```odin
when ODIN_OS == .Windows {
    import win "core:sys/windows"
    // ... Windows-only code
} else when ODIN_OS == .Linux {
    // ... Linux-only code
}
```

`when` does not introduce a new scope, and does not allow an init statement.

**Labelled break/continue** for nested loops/switches:

```odin
loop: for cond1 {
    for cond2 {
        break loop      // exits both
    }
}
```

`break` inside a `switch` exits the switch, not the surrounding loop — same as C/Go.
## Arrays, slices, dynamic arrays, maps, strings

Odin has **four distinct array kinds** plus maps. Don't conflate them.

| Type             | Layout                                  | Allocated? | Mutable len? |
|------------------|-----------------------------------------|------------|--------------|
| `[N]T`           | `N * sizeof(T)` inline                  | No (value) | No           |
| `[]T`            | `{data: ^T, len: int}` view             | No (view)  | No           |
| `[dynamic]T`     | `{data: ^T, len, cap: int, allocator}`  | Heap       | Yes          |
| `[dynamic; N]T`  | `{data: [N]T, len: int}` inline         | No (value) | Yes (≤ N)    |
| `map[K]V`        | hash map with allocator                 | Heap       | Yes          |

### Fixed arrays `[N]T` — value type with array programming

This is the biggest hidden divergence from Go. Fixed arrays are **values** (copied on assignment) and support **componentwise arithmetic** for any element type that supports the operator:

```odin
a := [3]f32{1, 2, 3}
b := [3]f32{5, 6, 7}
c := a * b              // {5, 12, 21}
d := a + b              // {6, 8, 10}
e := 1 + (c - d) / 2    // scalar broadcasts: {0.5, 3, 6.5}
```

`swizzle(a, 2, 1, 0)` reorders elements at compile time. For numeric arrays of length ≤ 4, **field-name swizzles** work directly: `a.xyz`, `a.zxy`, `v.xy`, `c.r`, `c.rgba`. This is why almost no Odin code uses a custom `Vec3` struct — `distinct [3]f32` gives you everything.

Convert a fixed array to a slice with `arr[:]`. Pass `&arr` to share by reference.

### Slices `[]T`

A view: pointer + length, no ownership. Same syntax as Go for slicing (`xs[lo:hi]`, `xs[:]`, `xs[lo:]`). `len(s)` works; **`cap(s)` does not** — slices have no capacity. To allocate a slice you own:

```odin
s := make([]int, 10)        // zero-initialized, uses context.allocator
defer delete(s)
```

### Dynamic arrays `[dynamic]T`

Heap-backed, growable. Carry their allocator inside the value, so `append` and friends do not need an allocator argument after the first allocation.

```odin
xs: [dynamic]int           // zero value is empty, no allocation yet
defer delete(xs)
append(&xs, 1, 2, 3)       // NOTE: pointer-to-dyn-array, not value
xs2 := make([dynamic]int, 0, 16)   // len 0, cap 16
```

**`append` takes a pointer**, unlike Go's `slice = append(slice, x)`. Mistake here is one of the most common Go-isms.

Compound literals (`[dynamic]int{1, 4, 9}`) require the file directive `#+feature dynamic-literals` at the top. Without it, build with `make` + `append`.

### Fixed-capacity dynamic arrays `[dynamic; N]T` (recent)

Stack-allocated, no allocator. Internal layout is `{data: [N]T, len: int}` — same shape as `core:container/small_array.Small_Array(N, T)` but built in.

```odin
buf: [dynamic; 16]u32
n := append(&buf, 1, 2, 3)   // n == 3
cap(buf) == 16               // compile-time
```

**Overflow is silent.** `append` returns `n: int`, the number actually appended; when full, you get fewer items added (or zero). No panic, no OOM. This is a different contract from heap `[dynamic]T` — branch on `n` if it matters.

All standard builtins work: `append`, `resize`, `pop`, `pop_safe`, `clear`, `inject_at`, `assign_at`, `ordered_remove`, `unordered_remove`. Real-world usage in `core:nbio/impl_posix.odin:25`.

### Maps `map[K]V`

```odin
m := make(map[string]int)
defer delete(m)

m["Bob"] = 2
v := m["Bob"]                  // zero value if absent — no panic
v, ok := m["Bob"]              // 2-value form for presence check
exists := "Bob" in m           // Odin has the `in` operator for maps
delete_key(&m, "Bob")          // NOT delete(m, "Bob") — that's Go
```

`delete(m)` frees the whole map. `delete_key(&m, k)` removes one entry — the name and the `&` matter.

### `make` / `new` / `delete` / `free`

Allocator-aware builtins. Last argument is an allocator, defaulted to `context.allocator`.

```odin
p   := new(T)                       // ^T, zeroed
s   := make([]T, n)                 // []T,        len = n
xs  := make([dynamic]T, len, cap)   // [dynamic]T, optional cap
m   := make(map[K]V)
str := make([]u8, 64, my_arena_alloc)   // explicit allocator

free(p)
delete(s)        // works for []T, [dynamic]T, map, and string (if allocated)
delete(xs)
delete(m)
```

Don't `free(s)` a slice or `delete(p)` a pointer — the type-specific builtin matters. `delete(string_literal)` is also wrong because string literals are not allocated (see below).

### Strings

`string` is an **immutable** view: `{data: ^u8, len: int}`. UTF-8 encoded by convention. `len(s)` returns the byte count, not the codepoint count — for codepoints, range over the string (each item is a `rune`) or use `core:unicode/utf8.rune_count`.

```odin
s := "literal"            // NOT allocated — points at static data, no delete
clone := strings.clone(s) // allocated copy — owns memory
defer delete(clone)
```

Mutate by working in `[]u8` and converting back:

```odin
buf := []u8{'H', 'i'}
s   := string(buf)        // view over buf — no allocation
```

**`cstring` is NUL-terminated** for C interop. Distinct from `string`. `len(cstring)` is **O(N)** — it walks to the NUL. Casts:

```odin
W :: "Hellope"
X :: cstring(W)           // compile-time, OK for literals
y := string(X)            // O(N) — walks to NUL to compute length
```

Don't pass `string` to a C API expecting `const char*` — convert via `cstring(...)` (only safe for literals or NUL-terminated buffers).

### Indexing returns a copy

`xs[i]` on `[N]T`, `[]T`, or `[dynamic]T` produces a **copy** of the element, just like reading any value. To get a reference, use `&xs[i]`. To mutate via iteration, use `for &x in xs { ... }` (covered in control flow).

### Other shapes worth knowing exist

- `matrix[R, C]T` — first-class matrix type with operator support.
- `#soa [N]Vec3`, `#soa []Vec3`, `#soa [dynamic]Vec3` — structure-of-arrays layout, transparent field access.
- `bit_set[Enum]` — see special-features section.
## Type system

### `distinct` types — nominal, not structural

Type aliases are structural by default (`Vec3 :: [3]f32` is interchangeable with `[3]f32`). Use `distinct` to make a type **nominally** different — same layout, but the compiler refuses implicit conversions. This is how Odin gets strong typing for vectors, IDs, units, etc.

```odin
Vector3  :: distinct [3]f32       // not assignable from a plain [3]f32
Entity_ID :: distinct u32         // can't accidentally pass a Player_ID

p: Vector3 = {1, 2, 3}
q: [3]f32  = p           // error: type mismatch
q = [3]f32(p)            // explicit cast required
```

Operators carry over to `distinct`-of-numeric and `distinct`-of-array — `distinct [3]f32` keeps componentwise math.

### Enums — nominal, with implicit selectors

```odin
Foo :: enum { A, B, C }      // backing type defaulted; specify with `enum u8 {...}`
f: Foo = .A                  // implicit selector: type known from context
f = Foo.B                    // explicit form also fine
```

`.A` works in any position where the type is known: assignments, switch cases, map keys, `bit_set` literals. There is no `iota`-equivalent — values count from 0 unless overridden (`A = 1, B, C = 7`).

### `union` — closed, tagged

```odin
val: union { int, bool }     // can hold an int, a bool, or nil
val = 137
val = true
val = nil

// Type assertion — comma-ok form:
if i, ok := val.(int); ok { fmt.println(i) }

// Type switch:
switch v in val {
case int:  fmt.println("int",  v)
case bool: fmt.println("bool", v)
case:      fmt.println("nil")
}
```

`union` is **closed**: only the listed variants are allowed. The compiler stores a tag automatically; for unions of pointer-like types, `nil` doubles as the tag (no extra space). Make a union `distinct` to give it identity.

Polymorphic unions are how `Maybe(T)` is built (see below).

### `any` — open, type-erased

```odin
val: any = 137
val = "hello"
if i, ok := val.(int); ok { ... }
```

`any` is `{data: rawptr, id: typeid}` — a fat pointer plus runtime type info. **It does not own the data**, so the underlying value must outlive the `any`. Used by `fmt`, `reflect`, and generic helpers. For closed sets, prefer `union`.

### `Maybe(T)` / `?T` — optional values

`Maybe` is a builtin polymorphic union, equivalent to `union($T: typeid) { T }`. The shorthand is `?T`:

```odin
i: Maybe(u8) = nil
i = 123

x := i.?              // panics if nil
y, ok := i.?          // safe form: y is u8, ok is bool
```

`.?` is the unwrap operator — distinct from `.(T)`. For pointers, `Maybe(^T)` carries no extra tag because `nil` is the sentinel. Don't use `Maybe(T)` to signal errors — that's what error returns and `or_return` are for (see `errors.md`).

### Pointers

`^T` is a pointer to `T`. Dereference is `p^` (postfix caret). Field access auto-dereferences: `p.field` works whether `p` is `T` or `^T`. There is no `&&` or pointer-to-pointer special syntax — `^^T` works.

```odin
x := 42
p := &x          // ^int
p^ = 100         // dereference and assign
```

`rawptr` is Odin's `void*` — untyped, no arithmetic, used for FFI and the `any` type's data slot.

### `typeid` and parametric polymorphism

Generic parameters are introduced with `$`. Two flavors:

```odin
print_value :: proc(value: $T)         { /* T inferred from arg */ }
alloc_type  :: proc($T: typeid) -> ^T  { /* T passed explicitly:  alloc_type(int) */ }
```

`$T` alone says "infer T from this arg". `$T: typeid` says "T is an explicit type parameter". You can combine: `proc($T: typeid, x: T)`.

**Type specialization** constrains the parameter using `/`:

```odin
make_slice :: proc($T: typeid/[]$E, len: int) -> T  // T must be a slice; E is its element
copy_slice :: proc(dst, src: []$T) -> int           // both args constrained to same []T
allocate   :: proc(table: ^$T/Table, capacity: int) // T must be a Table specialization
put        :: proc(table: ^Table($Key, $Value), key: Key, value: Value)
```

The pattern `^$T/Table` means "pointer to some `T` that is a specialization of `Table`". Lets you write methods-in-spirit on polymorphic structs.

### Polymorphic structs

```odin
Table :: struct($Key, $Value: typeid) {
    count:     int,
    allocator: mem.Allocator,
    slots:     []Table_Slot(Key, Value),
}

t: Table(string, int)
```

Specialize at use site by passing concrete types as arguments.

### `where` clauses — compile-time constraints

```odin
cross_2d :: proc(a, b: $T/[2]$E) -> E
    where intrinsics.type_is_numeric(E) {
    return a.x*b.y - a.y*b.x
}

Foo :: struct($T: typeid, $N: int)
    where intrinsics.type_is_integer(T),
          N > 2
{
    x: [N]T,
    y: [N-2]T,
}
```

Multiple clauses comma-separated. Predicates live in `base:intrinsics` (`type_is_numeric`, `type_is_integer`, `type_is_pointer`, `type_has_field`, …). `where` failures produce compile errors at the call site, not deep inside the body.

### `typeid` as a runtime value

`typeid` is a runtime handle to a type. Used by `any`, `reflect`, and explicit-type generic procs. Compare with `==`. Get one with `typeid_of(T)` or by reading `val.id` on an `any`.
## `using`

`using` brings names from another scope into the current one. Four forms:

```odin
// 1. As a struct field — fields of the inner struct appear directly on the outer.
//    Closest thing Odin has to inheritance.
Entity :: struct {
    using position: Vector3,    // entity.x, entity.y, entity.z all work
    orientation:    quaternion128,
}

Frog :: struct {
    ribbit_volume: f32,
    using entity:  Entity,      // frog.x, frog.position.x, frog.entity.x — all work
    colour:        Colour,
}

foo :: proc(e: ^Entity) { fmt.println(e.x) }
frog: Frog
foo(&frog)             // subtype polymorphism — no vtable, layout known
foo(&frog.entity)      // explicit form
frog.x = 123           // promoted field

// 2. On a procedure parameter — fields of the param accessible unqualified inside.
foo :: proc(using entity: ^Entity) {
    fmt.println(position.x, orientation)
}

// 3. As a statement inside a scope — opens a struct/import into the local namespace.
//    Statement form is OFF by default; enable per-file with `#+feature using-stmt`.
foo :: proc(entity: ^Entity) {
    using entity
    fmt.println(position, orientation)
}

// 4. On an import to bring names in unqualified — discouraged in production code.
import "core:fmt"
using fmt
println("...")     // works, but pollutes scope
```

The struct-field form is by far the most useful — it gives composition + subtype-style upcasting without inheritance machinery. Use sparingly: too many `using` fields makes it hard to find where a name comes from.

## Enum-indexed arrays — `[Enum]T`

A fixed array whose length is determined by an enum, and whose **indices are enum values**, not integers. Bounds-checked at compile time, exhaustive by construction.

```odin
Direction :: enum { North, East, South, West }

offsets: [Direction][2]int = {
    .North = { 0, -1},
    .East  = { 1,  0},
    .South = { 0,  1},
    .West  = {-1,  0},
}

step := offsets[.North]      // indexed by enum value, not by int

for offset, dir in offsets { // range gives (value, enum_key)
    ...
}
```

Use this whenever you'd otherwise reach for "an array with one slot per enum case" — input maps, per-channel state, per-material textures. The compiler will reject indexing with a non-`Direction` value, and `len(offsets) == len(Direction)` is guaranteed.

Pairs naturally with `bit_set[Direction]` for "which subset" and `[Direction]T` for "value per case".

## `bit_set` — first-class flag sets

Built-in flag sets backed by an integer. Backed by enums or by integer/rune ranges. Set operations are real operators, not bitwise hacks.

```odin
Day  :: enum { Sun, Mon, Tue, Wed, Thu, Fri, Sat }
Days :: distinct bit_set[Day]

WEEKEND :: Days{.Sun, .Sat}

d: Days
d  = {.Sun, .Mon}
e := d + WEEKEND          // union
e += {.Mon}               // insert
e -= {.Sun}               // remove
inter := d & WEEKEND      // intersection
diff  := d - WEEKEND      // difference

if .Sat in e      { ... } // membership
if .Sat not_in e  { ... }
n := card(e)              // number of elements set

// Subset / superset comparisons:
a, b: Days
ok := a <= b              // a is a subset of b
ok  = a <  b              // strict subset
```

Range-backed sets and explicit backing types:

```odin
letters: bit_set['A'..='Z']           // backed by smallest int that fits
small:   bit_set[0..=8; u16]          // explicit u16 backing
```

`in`/`not_in` are operators (not function calls); they work on `bit_set` and `map`. Constant `bit_set` literals can be evaluated at compile time (`X :: .Sat in WEEKEND`).

## `#` directives — quick reference

Compiler directives. Two flavors: **on declarations / types** and **on expressions / statements**.

| Directive | Where | Use |
|---|---|---|
| `#partial` | `switch` | Opt out of exhaustiveness for enum/union switches. |
| `#align(N)` | `struct` / variable | Force alignment in bytes. |
| `#packed` | `struct` | Remove padding between fields. |
| `#raw_union` | `struct` | All fields share offset 0 — C-style union. |
| `#soa` | `[N]T`, `[]T`, `[dynamic]T` | Structure-of-arrays layout, transparent field access. |
| `#caller_location` | proc default arg | Captures `Source_Code_Location` of the caller — used by `assert`, allocators, logging. |
| `#any_int` | proc parameter | Accept any integer type, implicitly converted. |
| `#no_broadcast` | proc parameter | Disable scalar→array broadcast on this argument. |
| `#bounds_check` / `#no_bounds_check` | proc, block, statement | Force or disable runtime bounds checks for this scope. |
| `#force_inline` / `#force_no_inline` | call site | Inlining hints at the call. |
| `#load("path")` | expression | Embed a file's bytes at compile time, returns `[]u8`. |
| `#assert(expr)` | top-level / proc | Compile-time assertion. |
| `#config(NAME, default)` | expression | Read a `-define:NAME=value` build flag. |
| `#procedure` | expression | Name of the enclosing procedure (string), useful in logs. |
| `#location()` | expression | `Source_Code_Location` at the call site. |
| `#file`, `#line`, `#column` | expression | Pieces of `#location()`. |
| `#type` | expression | Disambiguate a procedure-type expression from a literal. |

File-level directives sit at the very top, prefixed with `#+`:

```odin
#+vet !using-stmt !using-param      // enable extra lints, disable some
#+feature dynamic-literals          // opt into compound `[dynamic]T{...}` literals
#+build windows, linux              // restrict file to listed OSes
#+private                           // file is package-private (rarely needed)
```

When in doubt, Grep `<odin-root>/core` for `#<name>` to see real usage before guessing.

## `@(...)` attributes — declaration metadata

Attributes attach metadata or behavior to a declaration. They sit on their own line above the decl (or several stacked). Distinct from `#` directives, which apply to types/expressions/statements. Grep `<odin-root>/core` for `@(<name>` to see real usage.

| Attribute | Applies to | Effect |
|---|---|---|
| `@(private)` | any top-level decl | Package-private. Equivalent to `@(private="package")`. |
| `@(private="file")` | any top-level decl | File-private — invisible to other files in the same package. |
| `@(require_results)` | proc | Caller must use the return value; discarding is a compile error. The `must_*` builtins use this. |
| `@(thread_local)` | package-level var | Per-thread storage. Each OS thread gets its own copy. |
| `@(rodata)` | package-level var | Place in read-only data segment. Mutation is UB / segfault. |
| `@(init)` | proc with no args/returns | Runs before `main`. Multiple `@(init)` procs run in unspecified order. |
| `@(fini)` | proc with no args/returns | Runs at program exit. |
| `@(test)` | proc taking `^testing.T` | Picked up by `odin test`. |
| `@(export)` | proc / var | Symbol is exported from the produced object/DLL. |
| `@(extra_linker_flags="...")` | proc / foreign block | Extra flags for the linker. |
| `@(link_name="...")` | foreign / exported decl | Override the symbol name. |
| `@(link_prefix="...")` | foreign block | Prepend a prefix when resolving symbol names. |
| `@(link_section="...")` | var / proc | Place in a named section (e.g. `.text.hot`). |
| `@(default_calling_convention="...")` | foreign block | Set the convention for all procs in the block. |
| `@(disabled=COND)` | proc | If `COND` (a constant bool, often `!ODIN_DEBUG`), calls compile to nothing. Used for asserts and instrumentation. |
| `@(enable_target_feature="...")` | proc | Enable a CPU feature for codegen of this proc. |
| `@(cold)` | proc | Hint: rarely called; deoptimize for size, prefer not to inline. |
| `@(optimization_mode="none"/"minimal"/"size"/"speed")` | proc | Per-proc optimization override. |
| `@(deprecated="msg")` | proc / type | Emits a warning when used. |
| `@(builtin)` | proc | Marks a builtin (almost never written outside `base:builtin`). |
| `@(objc_class="...")`, `@(objc_name="...")`, `@(objc_type=Class)` | proc | Objective-C interop binding metadata. |

### Deferred-call attributes — Odin's RAII story

Three forms attach a **cleanup procedure** to a call. The attached proc runs at end of scope, automatically. Combined with a `bool`-returning proc and `if`, this gives clean scoped patterns without a real `defer` at the call site.

```odin
// Run `unlock` at end of scope, with no arguments:
@(deferred_none=unlock)
lock :: proc(m: ^Mutex) { ... }

// Run `pool_pop` at end of scope, passing a copy of the *input* arguments:
@(deferred_in=pool_pop)
pool_push :: proc(p: ^Pool, item: Item) { ... }

// Run `end_scope` at end of scope, passing the *return value*:
@(deferred_out=end_scope)
begin_scope :: proc(name: string) -> Scope_Handle { ... }

// `deferred_in_out` passes both inputs and outputs to the cleanup proc.
```

Caller-side this just looks like a normal call — the cleanup is implicit:

```odin
do_thing :: proc() {
    lock(&mu)              // unlock(&mu) runs at scope exit
    handle := begin_scope("frame")  // end_scope(handle) runs at scope exit
    ...
}
```

Combined with `@(require_results)`, this is how `if window_scope(&w) { ... }` patterns work: the `bool` return gates the body, and the deferred proc closes the scope.

## `foreign` blocks — C interop teaser

Bind to native libraries by declaring procedure signatures inside a `foreign` block. The `---` terminator marks a procedure declaration with no body (Odin signature for an external symbol).

```odin
when ODIN_OS == .Windows {
    foreign import kernel32 "system:kernel32.lib"
}

foreign kernel32 {
    ExitProcess :: proc "stdcall" (exit_code: u32) ---
}

@(default_calling_convention = "std")
@(link_prefix = "Get")
foreign kernel32 {
    LastError :: proc() -> i32 ---     // actually GetLastError
}
```

Default calling convention for foreign procedures is `"c"`. Override per-block with `@(default_calling_convention="...")` or per-decl with `proc "stdcall"`. `@(link_name="...")` and `@(link_prefix="...")` map Odin names to symbol names.

For deep C interop (struct ABI, varargs, `cstring`, libc shim), see `core:c` and `core:c/libc`.
