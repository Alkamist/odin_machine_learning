## Rules

- Don't edit this file.
- Don't git commit without asking. Instead, if you reach a good spot to commit, give me a one-line message to commit with and I'll do it. If I ask you to, commit with a single-line message, no co-author, and no session pin.
- Prioritize code correctness, clarity, and maintainability for humans.
- Minimize cross-cutting concerns and dependencies when possible.
- Don't leave old cruft if it is never going to be useful later.
- Always be on the lookout for bad practices and correct them.
- Don't spend time solving problems that are already publicly solved optimally.

## Comments

- DON'T WRITE ANY COMMENTS.

## Sub-Agents

- When doing large edit arcs, spawn an instance of Opus as a sub-agent to do it, then check and critique its work in an adversarial way. The goal is to arrive at the best possible code you can agree on. Keep context short, don't reuse a subagent for numerous requests.

## Style

- Use descriptive variable names.
- Use tabs for indentation, but spaces for alignment within struct field groups and related assignment blocks.
- `Ada_Case` for types, `SHOUTING_CASE` for constants, `snake_case` for everything else.
- Use spaces between all operators. Don't do `a*b`, prefer `a * b`.
- IMPORTANT BECAUSE YOU FORGET AND IS AN EXCEPTION TO THE PREVIOUS RULE: One-liner named function and struct arguments don't have spaces: `my_proc(a, b, c=foo, d=bar)` and `My_Type{a=foo, b=bar}`.
- If a procedure argument has a default value, you must set it by name when calling.
- Utilize anonymous assignments when possible: `a = {a=1, b=2, c=3}` instead of `a = My_Struct{a=1, b=2, c=3}`.
- Prefer `a := My_Struct{a=1, b=2, c=3}` instead of `a: My_Struct = {a=1, b=2, c=3}`, including with default arguments.
- Don't use `do`, prefer brackets with the expression on a newline, unless it makes sense to do otherwise.
- Pass `loc := #caller_location` and use it accordingly where it makes sense.
- Pass caller locations into logging and asserts, YOU DON'T NEED TO PREFIX LOG TEXT WITH PACKAGE AND FUNCTION LOCATION CONTEXT MANUALLY.
- Long line lengths are fine when they improve clarity, don't do weird line-breaks in the middle of constructs.
- Prefer `a->proc_ptr()` instead of `a.proc_ptr(a)`.
- Prefer using constants where possible. Don't create a proc that returns something that could be a constant.
- Never use `@(private)`, `@(private="file")`, or `#+private`. Privacy is conveyed by naming, not compiler enforcement.
- Denote private intent with a leading underscore, but ONLY on functions and variables: `_helper_proc`, `_internal_state`.
- Constants and types never get a leading underscore, even when private: `MAX_COUNT`, `My_Type`.

## Odin

- Odin is a C alternative language with manual memory management.
- Odin is NOT Go.

### Packages

Use `odin root` to find the installation of Odin.

Check:
- `base` for required Odin packages.
- `core` for Odin's core packages.
- `vendor` for 3rd party packages.

They represent good idiomatic Odin code. Prefer to import and use them over redefining those functions if possible.

### Compiler source

`Odin/` is a checkout of the compiler itself, gitignored and pinned to the revision `odin version` reports. Nothing builds against it, and it is not the specification: running native is. Read it when native's behavior needs an explanation that running it cannot give, such as filing a bug in `docs/UPSTREAM.md`.

Searching it takes `rg --no-ignore`, since an ignored file is invisible to a default search and answers "no matches" rather than saying it skipped anything. Re-pin the checkout whenever the installed Odin moves, or it describes a compiler that is not the one under test.

### Tips

- Package names need to be unique.
- Prefer static allocation when it makes sense, fixed-capacity dynamic arrays exist: `[dynamic; N]T`.
- Be VERY careful with `context.temp_allocator`, and look into `runtime.DEFAULT_TEMP_ALLOCATOR_TEMP_GUARD()` if you are going to use it.
- Look into `mem.Tracking_Allocator` if you need to track down leaks.
- Use `odin check` to quickly check for errors. A library package without `main` needs `-no-entry-point`.
- When optimizing, make sure to build with `-o:speed`.
