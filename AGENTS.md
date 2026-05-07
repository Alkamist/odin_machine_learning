## Style

- Reference the coding style of `ml.odin`.
- Be very picky about adding comments.
- Use descriptive variable names.
- Use tabs for indentation, but spaces for alignment within struct field groups and related assignment blocks.
- `Ada_Case` for types, `SHOUTING_CASE` for constants, `snake_case` for everything else.
- Don't use `@(private)`, prefix private procedures with an underscore, for example `_my_proc`.

## Optimization

- When optimizing, you MUST prove your theories by testing.
- Don't get lost in deep theories for no good reason, you need evidence.
- Reference `ggml`, `llama.cpp`, and `ollama`, which are proven to be very fast.

## Odin

- Odin is C alternative language with manual memory management. 
- Odin is NOT Go.

### Packages

Use `odin root` to find the installation of Odin.

Check:
- `base` for required Odin packages.
- `core` for Odin's core packages.
- `vendor` for 3rd party packages.

They represent good idiomatic Odin code. Prefer to import and use them over redefining those functions if possible.

### Tips

- Prefer static allocation when it makes sense.