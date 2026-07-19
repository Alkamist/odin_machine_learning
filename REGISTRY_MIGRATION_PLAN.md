# Registry and Tensor Construction Migration Plan

Outcome of a design deliberation on the public API. The goal is one correct way to do
everything around parameters and tensor construction. Execute the phases in order; each
phase leaves the tree compiling and tests passing.

Follow CLAUDE.md at all times: no comments, tabs for indentation, named default args,
one-liner named-arg style `f(a, b, c=foo)`, do not commit without asking.

## Design summary (the "why")

- The registry maps **external names to model-owned tensors**. An entry exists so a
  loader or checkpoint can address the tensor by name. Anything with no external
  identity (config-derived constants) does not belong in it.
- Trainability is a property the tensor already carries (`has_gradient`); the only
  facts that are NOT derivable are "does training produce this value?" and "does a
  checkpoint save it?". Those are the two orthogonal flags.
- `nil` never carries meaning. Unspecified init is an error exactly where init
  matters (`.Train`), and normalized to `Init_None` where it cannot.

## Target public API

```odin
Parameter_Flag  :: enum { Train, Checkpoint }
Parameter_Flags :: bit_set[Parameter_Flag]
PARAMETER_DEFAULT_FLAGS :: Parameter_Flags{.Train, .Checkpoint}

Init_None   :: struct {}
Init_He     :: struct {}
Init_Xavier :: struct {}
Init_Normal :: struct { mean: f32, std: f32 }
Init_Value  :: struct { value: f32 }
Init :: union { Init_None, Init_He, Init_Xavier, Init_Normal, Init_Value }

Parameter :: struct {
	name:   string,
	tensor: Tensor,
	init:   Init,
	flags:  Parameter_Flags,
}

Registry :: struct {
	parameters: [dynamic]Parameter,
}

parameter_make     :: proc(r: ^Registry, prefix, name: string, type: Data_Type, shape: []int, init: Init = nil, flags := PARAMETER_DEFAULT_FLAGS, loc := #caller_location) -> Tensor
parameter_register :: proc(r: ^Registry, prefix, name: string, tensor: Tensor, init: Init = nil, flags := PARAMETER_DEFAULT_FLAGS, loc := #caller_location)

registry_destroy   :: proc(r: ^Registry, loc := #caller_location)
registry_randomize :: proc(r: ^Registry, loc := #caller_location)
registry_update    :: proc(opt: ^Optimizer, r: ^Registry, loc := #caller_location)
registry_copy      :: proc(dst, src: ^Registry, loc := #caller_location)
registry_gather    :: proc(dst, src: ^Registry, prefix := "")
registry_count     :: proc(r: ^Registry, flags := Parameter_Flags{.Train}) -> int
registry_read      :: proc(r: ^Registry, dst: []f32, loc := #caller_location)
registry_write     :: proc(r: ^Registry, src: []f32, loc := #caller_location)

checkpoint_save    :: proc(path: string, r: ^Registry, opt: ^Optimizer, metadata: map[string]string, loc := #caller_location) -> bool
checkpoint_load    :: proc(path: string, r: ^Registry, opt: ^Optimizer, loc := #caller_location) -> (metadata: map[string]string, ok: bool)
clip_gradient_norm :: proc(r: ^Registry, max_norm: f32, loc := #caller_location) -> f32
```

### Behavioral contract

| operation | filter |
|---|---|
| `registry_update`, `clip_gradient_norm`, `registry_read`, `registry_write` | `.Train in flags` |
| `checkpoint_save`, `checkpoint_load` | `.Checkpoint in flags` |
| `registry_randomize` | applies `init` unless it is `Init_None` |
| loaders (gguf/safetensors) | every entry, by name |
| `registry_count(r, flags=F)` | entries whose flags contain F; `{}` counts all |

- `parameter_make` allocates `{.Data, .Gradient}` iff `.Train in flags`, else `{.Data}`.
  Always `persistent=true`.
- `parameter_register` asserts: `.Train in flags` implies `has_gradient(tensor)`.
- Both entry procs: if `init == nil` and `.Train in flags`, assert with message
  "trainable parameter requires an init; pass Init_None if it is filled by a loader or
  by hand". If `init == nil` otherwise, store `Init_None{}`. `nil` is never stored.
- Both entry procs take the registry as first arg. Every `registry_*` proc takes
  `^Registry`. Nothing takes `[]Parameter` or `^[dynamic]Parameter`. Callers never
  slice.
- Name handling is unchanged from today's registry: names are cloned into the
  registry's allocator, prefixed with `prefix + "."` when prefix is non-empty.
- `registry_gather` clones names (applying prefix) into `dst`; the tensors are
  borrowed. Convention: gathered registries are call-scoped, built with
  `context.temp_allocator`, and never passed to `registry_destroy`. This is how
  composite models (jepa, gemma+lora) present a merged registry to
  checkpoint/clip/read/write.
- `registry_copy` keeps today's semantics: equal length, matching names, tensor copy.
- `checkpoint_save` keeps the F32/Bf16 assert, but it now only applies to
  `.Checkpoint` entries, so quantized frozen weights are skipped naturally rather
  than by caller discipline. Optimizer moments are saved for entries that have
  optimizer state, as today.

### Deletions (must not survive the migration)

From `ml.odin`:
- `Parameter` (old flat `{name, tensor}` form — replaced by the new struct)
- `parameter_append`, `parameters_len`, `parameters_read`, `parameters_write`

From `parameters.odin`:
- `Parameter_Info`, `register`, `registry_parameters`, `registry_parameter_count`,
  `registry_clear` (audit first: if `registry_clear` has callers, migrate them to
  `registry_destroy` or delete the call; do not keep it "just in case")

Tensor construction, from `ml.odin`:
- `make` — replaced by `alloc` with defaults (see Phase 2)
- `const_scalar` — folded into `scalar` (see Phase 2)

## Phase 1 — core package (`ml`)

Rewrite `parameters.odin` to the target API above (it becomes the registry file;
renaming the file to `registry.odin` is fine). In `ml.odin`, delete the flat-Parameter
block and re-point `clip_gradient_norm` at `^Registry`. Re-point `checkpoint.odin`
at `^Registry` with the `.Checkpoint` filter.

Notes:
- `registry_randomize` keeps today's He/Xavier rank-2 asserts and shape-derived fans.
- `registry_read`/`registry_write` iterate `.Train` entries in registry order; length
  asserts against `registry_count(r, flags={.Train})` element sum as today.

## Phase 2 — tensor construction (`ml.odin`)

- `alloc` gains defaults: `alloc :: proc(type, shape, persistent := false, buffers := DEFAULT_ACTIVATION_BUFFERS, loc := #caller_location) -> Tensor`.
- Delete `make`. Call sites (tests, mostly) become
  `alloc(type, shape, persistent=true, buffers=DEFAULT_PARAMETER_BUFFERS)`.
- Fold `const_scalar` into `scalar`: `scalar :: proc(type, value, persistent := false, loc := #caller_location) -> Tensor`.
  `persistent=true` gives today's `const_scalar` behavior (persistent, `{.Data}` only);
  the default gives today's step-scoped `scalar`.
- `zeros`, `zeros_like`, `scratch`, `tensor` are unchanged. Add no new constructors.

## Phase 3 — networks

For every network: the params field becomes `params: ml.Registry`, the `parameters`
proc becomes a `registry_gather` one-liner, `update` becomes
`ml.registry_update(opt, &model.params)`, `destroy` calls
`ml.registry_destroy(&model.params)` plus explicit destroys for constants.

- **mlp**: mechanical. `parameter_make(&mlp.params, ...)` calls gain nothing new;
  keep explicit `Init_He`/`Init_Value` (init is now required for trainables).
- **lora**: `A`/`B` via `parameter_make` as today. `scale` LEAVES the registry:
  it becomes `ml.scalar(dtype, alpha / f32(rank), persistent=true)` held as a plain
  field and destroyed in `lora.destroy`. Delete its `register` line.
- **llama**: mechanical; every `parameter_make` already states init. For the
  inference/training duality nothing changes: a model built without gradient buffers
  registers with `flags={}` — add a `trainable` (or flags) parameter to its `make`
  path mirroring how it decides buffer allocation today.
- **gemma**: keep the `_register_parameters` one-block name table using
  `parameter_register`. Frozen quantized weights: `flags={}`, `init=nil` (normalizes
  to `Init_None`). Trainable entries state init explicitly. All constants leave the
  registry and become plain fields destroyed in `gemma.destroy`:
  `embed_scale`, `ple_token_scale`, `ple_ctx_scale`, `ple_combine_scale`,
  `softcap`, `softcap_inv`, `v_norm_ones_sliding`, `v_norm_ones_full`.
  Replace `trainable=ml.has_gradient(tensor)` inference with explicit flags per
  entry (`{.Train, .Checkpoint}` when built for training, `{}` when frozen/inference).
- **gemma/llama loaders**: iterate `model.params.parameters`; name lookup logic
  unchanged.
- **jepa**: migrate registries mechanically first (its mlps carry their own
  registries; `jepa.parameters` gathers with prefixes). Its `save`/`load` wrappers
  build a gathered registry (temp allocator) and call `checkpoint_save/load`.

## Phase 4 — examples and tests

- `examples/mnist`, `examples/cartpole`, `examples/llm_chat`: follow the network API
  changes; no structural changes expected.
- Tests: replace `ml.make(...)` per Phase 2; `gemma_lora_check` switches
  `ml.parameters_read(params[:], ...)` to `ml.registry_read(&gathered, ...)` on a
  gathered registry. Audit every test that builds a `[dynamic]ml.Parameter` or
  `[dynamic]ml.Parameter_Info` and convert to `ml.Registry`.

## Phase 5 (optional, separate change) — jepa target encoder as a buffer

The target encoder is EMA-updated state (`lerp_assign`), never gradient-trained. Give
`mlp.make` a way to build without gradient buffers, and register the target encoder's
tensors with `flags={.Checkpoint}`. This deletes its wasted gradient buffers and turns
its checkpointing from accidental (it happens to have gradient buffers today) into
declared. Do this only after Phases 1-4 are verified; it changes jepa checkpoint
compatibility considerations.

## Out of scope (do NOT do here)

Deliberately deferred items from the same API sweep; do not fold them in:
- `linear`/`linear_q4_k`/`linear_q6_k` dtype dispatch unification
- fused-op signature/default unification (`rmsnorm_rope` etc.)
- `context_scope` vs `context_begin` blessing and assert-message fix
- underscoring the CPU backend's internal op procs
- `get_data_bytes`/`set_data_bytes` removal

## Verification

After each phase:
1. `odin check . -no-entry-point` for the root package; `odin check` each backend,
   network, example, and test package that was touched.
2. Build and run the test programs under `tests/` (CPU at minimum; CUDA tests if the
   machine allows).
3. After Phase 3: run `examples/mnist` briefly and confirm loss decreases; run
   `tests/gemma_lora_check` and `tests/checkpoint_check`.
4. Grep-verify the deletions list: `Parameter_Info`, `parameter_append`,
   `parameters_len`, `parameters_read`, `parameters_write`, `registry_parameters`,
   `registry_parameter_count`, `ml.make(`, `const_scalar` must have zero hits outside
   this document.
5. Checkpoint compatibility: files written after this change no longer contain
   constant tensors (gemma). Old checkpoints containing them will fail the "missing
   tensor" path in reverse — `checkpoint_load` only looks up registry entries, so
   extra tensors in old files are ignored and loading old checkpoints still works.
   Verify this explicitly with `tests/checkpoint_check`.
