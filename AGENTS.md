## Odin

- Odin is C alternative language with manual memory management. 
- Odin is NOT Go.
- Use `odin check` to quickly check for errors.

### Packages

Use `odin root` to find the installation of Odin.

Check:
- `base` for required Odin packages.
- `core` for Odin's core packages.
- `vendor` for 3rd party packages.

They represent good idiomatic Odin code. Prefer to import and use them over redefining those functions if possible.

### Tips

- Prefer static allocation when it makes sense, fixed-capacity dynamic arrays exist: `[dynamic; N]T`.
- Be VERY careful with `context.temp_allocator`, and look into `runtime.DEFAULT_TEMP_ALLOCATOR_TEMP_GUARD()` if you are going to use it.
- Look into `mem.Tracking_Allocator` if you need to track down leaks.
- There are no stack traces, ask the user to use a debugger if necessary.
- Constants are untyped, and generally don't need an explicit type.

## Style

- Reference the coding style of `ml.odin`.
- Be very picky about adding comments.
- Don't use characters in comments that aren't typical for normal keyboards.
- Use descriptive variable names.
- Use tabs for indentation, but spaces for alignment within struct field groups and related assignment blocks.
- `Ada_Case` for types, `SHOUTING_CASE` for constants, `snake_case` for everything else.
- Don't use `@(private)` or `@(private="file")`, prefix private procedures with an underscore, for example `_my_proc`.
- Don't use `do`, prefer brackets with the expression on a newline, unless it makes sense to do otherwise.
- Pass `loc := #caller_location` and use it accordingly where it makes sense.

## Optimization

- When optimizing, you MUST prove your theories by testing.
- Don't get lost in deep theories for no good reason, you need evidence.
- Reference `ggml`, `llama.cpp`, and `ollama`, which are proven to be very fast.