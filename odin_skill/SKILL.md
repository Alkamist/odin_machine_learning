---
name: odin
description: Use whenever reading, writing, reviewing, or discussing Odin source code (`.odin` files, `ols.json`, Odin build commands). Odin is a manual-memory systems language that LOOKS like Go but is NOT Go — its syntax, standard library, and idioms differ in ways that trip up models trained on Go. This skill enforces verification against the user's local Odin installation to prevent hallucinated APIs and Go-isms.
---

# Odin

## Read this first — Odin is not Go

Models trained on large web corpora consistently hallucinate Odin code by pattern-matching to Go. **Before writing any Odin you are not 100% certain about, consult `not-go.md`.** Common failure modes:

- Inventing Go-style methods (`x.foo()`) — Odin has no methods, only procedures.
- Using `:=` / `make` / `for` / `defer` with Go semantics — all four differ in Odin.
- Writing `package main` imports as `"fmt"` instead of `"core:fmt"`.
- Inventing `core:` packages or symbols that don't exist.
- Ignoring allocators and the implicit `context` system.

## Verification protocol — non-negotiable

The user's Odin installation is the source of truth. Hallucinated APIs are the #1 failure mode this skill exists to prevent.

1. **At the start of any Odin task, run `odin root`** to locate the install (e.g. `C:\odin\`). Cache the path for the session. The standard library lives under `<root>/base`, `<root>/core`, `<root>/vendor`.
2. **Before writing any `core:` or `vendor:` symbol you are not certain exists**, Grep or Read the relevant package directory under `<root>` to confirm the procedure name, signature, and return type. See `stdlib-map.md` for which directory to look in.
3. **After writing non-trivial code, run `odin check .`** in the package directory. If the user wants it executed, `odin run . -debug`.
4. If a symbol you expected isn't there, **do not invent a substitute** — search the rest of `core` (`Grep -r` for the concept) or tell the user it's missing.

## Reference files

Load on demand:

- `not-go.md` — Syntax and idioms where Odin diverges from Go. Read before writing any Odin you haven't verified.
- `memory.md` — Allocators, the `context` system, `defer delete`, arenas, tracking allocator for leak checks.
- `errors.md` — Multiple return values, `or_return`, `or_else`, named returns. Odin has no exceptions.
- `stdlib-map.md` — Task → package index for `core:` and `vendor:`. Use to know where to Grep before reaching for a symbol.

## Code style (project-agnostic defaults)

- `Ada_Case` for types, `SHOUTING_CASE` for constants, `snake_case` for everything else.
- Tabs for indentation. Spaced alignment within struct field groups and related assignment blocks.
- Type-prefixed procedure names: `material_make`, `material_destroy` — not `make_material`.
- **Leading underscore on file-local procs and variables only**
  (`_helper_proc`, `_global_state`). **Types and constants never get
  the underscore prefix** — `Worker_Pool`, `MAX_RETRIES`, even when
  file-local. **Do NOT use `@(private)` or `@(private="file")`** — the
  underscore-prefix-on-procs-and-vars convention is what marks
  file-locality here, and the attribute adds visual noise without
  buying anything the convention doesn't already convey.
- Comments explain non-obvious *why*, never restate *what*. Default to none.

If the project has its own style guide, that overrides these.
