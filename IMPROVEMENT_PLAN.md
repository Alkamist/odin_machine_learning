# General-Purpose ML Library Improvement Plan

Goal: evolve this library from an LLM-inference engine with a training core into a general-purpose machine learning library, without losing what is already good about it.

This plan is self-contained: it records the findings of a full design/implementation review (2026-07-18, branch `ai_improvements`, HEAD `857ce6c`), the reasoning behind each change, and concrete acceptance criteria. All `file:line` references are against that commit and will drift as edits land — treat them as pointers, not gospel. **Re-verify every claim against the current source before changing code.** Read `CLAUDE.md` first and follow it exactly (notably: no comments, tabs for indentation, `Ada_Case`/`SHOUTING_CASE`/`snake_case`, named default args, don't commit without asking).

## Architecture summary (for orientation)

- `ml.odin` — the core: `Tensor` (rank ≤ 6, dtypes `{Bf16, F32, Q4_K, Q6_K}`), a thread-local `Context` holding a linear tape of up to `MAX_OPERATIONS` recorded `Operation`s, eager forward execution, reverse-order `backward`. Gradients are always F32 (`buffer_dtype`). Ops are a closed union `Operation_Variant` mirrored by `Operation_Kind`, `OPERATION_SET_ALL`, and the `operation_kind` switch — four hand-maintained parallel lists.
- `Backend` is a struct of proc pointers plus `forward_ops`/`backward_ops` capability bitsets. Front-end ops decompose into simpler ops when a fused kernel is missing (`gelu_mul`, `rmsnorm_rope`, `add_rmsnorm`, `linear_q4_k_gate_up_geglu`, `rmsnorm_rope_write_cache` all check `training`/`forward_ops` and fall back). This decomposition mechanism is one of the library's best ideas — preserve it.
- `backends/cpu/cpu.odin` — every op implemented twice by hand (F32 and Bf16 clones), a worker-pool `parallelize`, arena-based transient memory.
- `backends/cuda/` — NVRTC runtime compilation of `#load`-embedded `.cu` sources, pipeline cache, activation pool, CUDA graph auto-capture for decode, cuBLAS for linear, flash-style inference attention, llama.cpp-style Q4_K/Q6_K MMVQ kernels.
- `networks/` (gemma, llama, mlp, jepa, lora), `loaders/` (gguf, safetensors, weights), `tokenizers/` (gemma, gpt2), `checkpoint.odin`, `quants.odin`, `examples/` (mnist, cartpole, llm_chat, fetch), `tests/` (grad_check, adam_check, clip_check, cases registry, parity vs CUDA).

Preserve these existing strengths: capability-bitset decomposition, F32-always gradients, the activation pool, CUDA graph capture, NVRTC-at-runtime with no build-time CUDA dependency, the shared test-case registry, `#caller_location` threading through asserts/logs.

Build/check commands:
- `odin check . -no-entry-point` (root library), same per package.
- `odin test tests` and `odin test tests\parity` (parity needs a CUDA GPU; see Phase 5.4).
- Optimized builds: `-o:speed`.

---

## Phase 1 — Memory-safety and correctness bugs — DONE (2026-07-18)

All items below are fixed on `ai_improvements`. Decisions and contracts established while fixing them:

- **Gradient-sink contract (1.2):** a tensor without a gradient buffer is a gradient sink — backend backward procs skip accumulation into it. This is what makes frozen weights (QLoRA base, including rmsnorm norm weights) and `scratch` inputs safe, and gives stop-gradient semantics for free. Enforced in every backward proc on both backends (`ml.has_gradient` on CPU, `.ptr == 0` on CUDA); CUDA rmsnorm kernels take nullable `dx`/`dw`; CUDA attention backward asserts against partial-null dq/dk/dv. Regression test: `tests/gradient_sink_check.odin`.
- **1.3 decision: option (b), not (a).** Quantized-in-file → quantized-in-model replacement is required (GGUF Q4_K/Q6_K inference), so `write_tensor` keeps its replace semantics and all loaders now take models by pointer (`gemma.load_safetensors(^Gemma)`, `llama.load_safetensors(^Llama)`) and re-establish tied-embedding aliases after load. Note: safetensors sources can never be quantized (`_safetensors_dtype` maps only F32/F16/BF16), so the old by-value bug was latent, not live.
- **Zeros contract (1.5): option (b).** Transient `.Data` buffers are uninitialized storage — every op must fully overwrite its output and must not read it before writing. `.Gradient` buffers and persistent buffers ARE zeroed on both backends (backward accumulates into gradients). CPU `batched_matmul_forward_f32` was the one violator (axpy accumulation) and now zeroes its output rows first. `-define:ML_CPU_POISON=true` fills transient `.Data` with NaN to enforce this; the CPU test suite passes under it. `lerp_assign`/`accumulate_mean` operate on caller-provided persistent tensors and are exempt by design.
- **1.6:** minimum fix taken — `_emit_position_upload` now records the per-forward position and panics on a conflicting second value instead of silently reusing the first. Per-op position slots remain future work if heterogeneous positions in one forward are ever needed.
- **1.8:** commit `3af9acd` only bound the context on the creating thread. Now `_gctx` and `_compile_pipeline` lazily bind the primary context per thread, keyed on a device generation counter so re-init after full teardown rebinds correctly.
- **1.9 tokenizer finding:** the plan's `▁` dummy-prefix claim is WRONG for Gemma — the shipped `tokenizer.json` normalizer is only `Replace(" " → "▁")` (Gemma uses `add_dummy_prefix=false`), which matches the Odin implementation exactly. No fix needed; golden-token tests remain Phase 5.3 work.
- **1.9 `_upload_indices`:** replaced temp-allocator staging (use-after-free window under `backward`'s temp guard; pageable async HtoD illegal during capture) with a reusable pinned staging pool on the CUDA `Context`, released after the stream sync in `clear`. Auto-capture re-captures every decode step and patches via `GraphExecUpdate`, so host pointers stay fresh.
- **1.9 grid limits:** `_dispatch` centrally asserts `grid_y`/`grid_z <= MAX_GRID_DIM_YZ`.
- **1.9 quant-linear dtype:** both backends assume Bf16 activations for `linear_q4_k`/`linear_q6_k` (CPU reads `[^]Bf16`, CUDA runs `.R_16BF` GEMM/MMVQ), so the front end now asserts Bf16-only input instead of the previous "Bf16 or F32".
- 1.1 front-end F32 asserts added to `clamp`/`min`/`max`/`mean_squared_error`/`smooth_l1`/`cross_entropy`; `ml.copy` asserts dtype+shape; `scalar`/`fill_normal`/`fill_value` panic on quant dtypes; gguf loader rejects dim/count overflow, hostile string lengths, and duplicate tensor names; safetensors validates byte ranges against shape×dtype; llama `make` honors its allocator; tokenizer `encode` returns `(ids, ok) #optional_ok`; gemma prefill tail runs as one chunk; 1.4 saves/restores `ml.optimizer_iteration` (round-trip test in `tests/checkpoint_check.odin`); 1.7 pipeline cache is mutex-guarded and keyed by source+options.
- Verified: `odin check` clean on all touched packages, `odin test tests` green (6 tests incl. poison mode), `odin test tests\parity` green on an RTX 3090 Ti.

Original findings kept below for reference.

### 1.1 CPU F32-only ops overflow buffers on Bf16 input (VERIFIED)

`data()` at `backends/cpu/cpu.odin:396` unconditionally returns `([^]f32)(ptr)[:t.count]`. A Bf16 tensor's data buffer is `count * 2` bytes, so any op that calls `data()` on a Bf16 tensor reads/writes `count * 4` bytes — 2x out of bounds.

Affected (no dtype guard in the backend, and no F32 assert in the `ml.odin` front-end proc either): `clamp_forward` (cpu.odin:991), `min_forward`/`max_forward` (cpu.odin:1017–1067), `transpose_forward`, `mean_squared_error_forward` (cpu.odin:2479), `smooth_l1_forward`, `cross_entropy_forward` (cpu.odin:2566). Audit the whole file for other unguarded `data()`/`gradient()` calls — do not trust this list to be exhaustive.

Related: CPU `linear_q4_k`/`linear_q6_k` forward reads input as `[^]ml.Bf16` (cpu.odin:1441) but `ml.linear_q4_k` (ml.odin:1476) accepts F32 input — F32 input silently reinterprets bytes.

Fix approach: add dtype asserts in the `ml.odin` front-end procs for ops that are genuinely F32-only (matching what `transpose` already does at ml.odin:1305), OR implement the Bf16 paths. Prefer asserts first (small, safe), implement Bf16 later as needed. For quant linears, either assert Bf16 input or insert a cast in the front end.

Acceptance: no code path can reach a `data()` call on a non-F32 tensor; a test constructs each formerly-affected op with a Bf16 tensor and observes a clean assert (not corruption).

### 1.2 CUDA `_add_backward` writes through null gradient pointer (VERIFIED)

`backends/cuda/ops.odin:570–612`: `_add_backward` dispatches `add_back_b` (and `add_back_a`) unconditionally with `gradient(b).ptr`, which is 0 when `b` has no gradient buffer (e.g. tensors from `scratch`, or `zeros` allocated under `training=false`). `_mul_backward` (ops.odin:1641–1642) and `_min_backward` (ops.odin:788–789) already guard with `if dap != 0` / `if dbp != 0` — mirror that pattern in `_add_backward`, and audit every other `_*_backward` in ops.odin for the same missing guard. Check the CPU backend for the equivalent issue (CPU gradient() on a missing buffer transmutes a zeroed handle — verify what actually happens and guard if needed).

Acceptance: backward through `add(x, scratch_tensor)` on CUDA does not launch a kernel with a null pointer; all binary-op backwards have consistent guards.

### 1.3 `weights.write_tensor` destroy-and-replace vs by-value loaders (VERIFIED design flaw, latent use-after-free)

`loaders/weights/weights.odin:141–144` (`_write_quant`): when the file tensor is quantized, the proc destroys the target tensor and writes a NEW tensor through the pointer. But `networks/gemma/loader.odin:24–32` and `networks/llama/loader.odin:18–21,38–46` pass pointers to LOCAL COPIES of model fields (the model is passed by value), so the model's real field would keep pointing at freed memory and the replacement would be dropped. It only works today because those particular tensors are F32/BF16 in the files currently loaded. Additional hazard: `llama/loader.odin` ties embeddings (`lm_head_weight` aliases `token_embeddings`); a replace would silently break the tie.

Fix approach: make the contract explicit and single-moded. Options (pick one, document in the proc signature shape):
- (a) `write_tensor` never reallocates: it requires the destination to already have the right dtype, and dequantizes into it when dtypes differ (or errors).
- (b) `write_tensor` always takes `^Tensor` and may replace — then all loaders must take models by pointer (`load_gguf` already takes `^Gemma` at loader_gguf.odin:14; `load_safetensors` takes `Gemma` by value at loader.odin:14 — that inconsistency is the symptom) and must not write through copies, and tied-weight aliases must be re-established after load.
Option (a) is simpler and safer; prefer it unless there is a reason quantized-in-file → quantized-in-model replacement is required.

Acceptance: loaders take models by pointer or never mutate tensor identity; a test loads a file whose tensor is quantized while the model field is not (or vice versa) and the model ends up correct with no leak (verify with `mem.Tracking_Allocator`).

### 1.4 Checkpoint resume breaks Adam bias correction (VERIFIED)

`checkpoint.odin` saves/restores Adam `m`/`v` moments but never persists `Optimizer.iteration` (and `optimizer_step` recomputes `bias_correction1/2` from `iteration`, ml.odin:575–576). After a resume, bias correction restarts at t=1 with warm moments — wrong updates for the early post-resume steps.

Fix: store `iteration` in the checkpoint metadata (e.g. `ml.optimizer_iteration`), restore it in `checkpoint_load` when `opt != nil`. `checkpoint_metadata_u64` already exists as the parsing helper. Also audit: `networks/jepa/jepa.odin:288` smuggles its own step counter through metadata but never restores `opt.iteration` — fix the jepa load path to benefit too.

Acceptance: save → load → `optimizer_step` continues with the pre-save iteration count; a round-trip test asserts it.

### 1.5 `zeros` does not zero on CUDA

`backends/cuda/buffer.odin:42–44`: `buffer_alloc` memsets only when `kind != .Data || persist`; transient `.Data` buffers recycled through the activation pool are returned uninitialized. Every current CUDA op fully overwrites its output, so this is invisible — but CPU `batched_matmul_forward_f32` (cpu.odin:2846–2849) already ACCUMULATES into its output via axpy and is only correct because Odin's allocator zeroes; the same op pattern on CUDA would produce garbage. The `ML_CPU_POISON` config (cpu.odin:331) shows the footgun is known.

Fix: choose one contract and enforce it everywhere:
- (a) `zeros` means zeroed: memset transient .Data allocations on CUDA (cost: one memset per activation; measure — the pool reuses exact sizes so this is a per-op memset, possibly significant for decode; can be skipped during graph capture only if the graph path is proven overwrite-only).
- (b) rename the semantic: outputs are uninitialized and every backend op MUST fully overwrite its output; then fix CPU `batched_matmul` to not rely on zeroing (initialize the output inside the op), and state the contract where `zeros` is declared.
Option (b) is likely the performance-correct choice; it requires auditing every CPU op that reads its own output before writing (search for `+=` on `data(output)`).

Acceptance: the contract is written down at `zeros`, both backends satisfy it, and a poison-mode test (fill pool buffers with NaN, run each op, check output has no residue-dependence) passes on CPU; parity covers CUDA.

### 1.6 CUDA `position_dev` is first-writer-wins per forward

`backends/cuda/ops.odin:79–86` (`_emit_position_upload`): a single per-forward device scalar, uploaded once, guarded by `position_written_this_forward`. `_rope_forward` uses `v.position_offset`, `_attention_cache_forward` uses `v.cache_position`, `_rmsnorm_rope_*` use their own offsets. If two ops in one forward pass carry different positions — which the front end freely allows — the second silently gets the first's value.

Fix (minimum): assert that every subsequent upload in the same forward matches the recorded value, so heterogeneity panics instead of silently corrupting. Fix (proper): give each consuming op its own position slot (a small ring of device scalars indexed per op, or pass position as a kernel arg where graph capture allows; the whole point of the indirection is CUDA-graph replayability, so keep positions in device memory but stop sharing one slot).

Acceptance: two ropes with different `position_offset` in one forward either work correctly or assert loudly; the LLM decode path still graph-captures.

### 1.7 `_gpu.pipeline_cache` race + weak cache key

`backends/cuda/pipeline.odin:80,126`: `_compile_pipeline` reads/writes the shared `_gpu.pipeline_cache` without `_gpu_mutex` (everything else touching `_gpu` locks). Also the cache key is only `source_name`, so `extra_options` are ignored on a cache hit — currently dodged by baking `D_HEAD` into the filename (ops.odin:1363–1367), a trap for the next option.

Fix: take `_gpu_mutex` around cache lookup/insert (compile outside the lock, double-check on insert), and key the cache by `source_name` + the full options string.

Acceptance: two threads compiling concurrently is safe; a pipeline compiled with different `extra_options` gets a distinct cache entry.

### 1.8 CUDA context not bound on non-creating threads

`backends/cuda/cuda.odin:214` calls `CtxSetCurrent` only in `context_create`; nothing in the dispatch path binds the context for other threads. Commit `3af9acd` ("Bound the CUDA primary context per thread...") claims per-thread binding — **investigate what that commit actually does before changing anything**; if binding exists via another mechanism, document where; if not, bind the primary context lazily per thread (thread-local flag) at the top of backend entry points.

Acceptance: forward/backward from a thread other than the creator works (or is explicitly asserted against, if single-thread dispatch is the intended contract — in which case assert it).

### 1.9 Smaller verified/reported issues (batch these)

- `ml.copy` (ml.odin:345–351) asserts equal length but not equal dtype/shape; a Bf16→F32 copy has mismatched byte counts. Assert type and shape.
- `ml.scalar` (ml.odin:275–287) and `fill_normal`/`fill_value` (ml.odin:435–474) use `#partial switch` and silently do nothing for Q4_K/Q6_K. Add a default panic arm.
- gguf loader: `element_count *= d` (loaders/gguf/gguf.odin:196–199) can overflow with hostile u64 dims; `_skip_array_payload` computes `need := count * 8` (gguf.odin:474) — overflow makes `need` negative and the bounds check passes, walking the cursor backwards; same for `_skip_scalar_payload:442`. Use checked math / reject counts above a sane cap. Duplicate tensor names overwrite the map entry and leak the previous `shape` slice (gguf.odin:210) — reject duplicates.
- safetensors loader: validate `end - start == shape_element_count * dtype_size` at parse time (loaders/safetensors/safetensors.odin:135 only bounds-checks the range).
- `_upload_indices` (backends/cuda/ops.odin:1375–1385): `MemcpyHtoDAsync` from `context.temp_allocator` pageable memory — the slice can be recycled before the copy completes, and a pageable async HtoD during graph capture is not legal. Use a persistent pinned staging buffer, or synchronous copy when not capturing + assert-not-capturing for the ops that need it (Select/Cross_Entropy during decode capture).
- `networks/llama/llama.odin:58,63`: `make` takes `allocator` but `builtin.make([]Layer, ...)` uses `context.allocator` (gemma sets `context.allocator = allocator` at gemma.odin:195). Fix llama to honor it.
- CUDA inference flash-attention path (ops.odin:1248) uses `grid.y = token_count` with no check against the 65535 grid-Y limit; add the assert (the constant `MAX_GRID_DIM_YZ` already exists at ops.odin:52–54). Audit other dispatches for the same.
- Tokenizers: gemma tokenizer never adds SentencePiece's leading `▁` dummy prefix (tokenizers/gemma/gemma.odin:201) — first-token ids likely differ from reference implementations; verify against a reference tokenization of a few strings and fix. Both tokenizers silently truncate on unencodable input (gemma.odin:223–240, gpt2.odin:172–174) — return an ok/error indication.
- `examples/llm_chat/backend_gemma.odin:117–134`: prefill tail degrades to one-token-at-a-time (`take = 1` for any remainder after 64-token chunks — a 63-token tail costs 63 forwards). If this is a CUDA-graph constraint, name it in a constant; otherwise process the remainder as one chunk.

---

## Phase 2 — Parameter registry — DONE (2026-07-18)

Implemented as designed, with these concrete decisions:

- Registry lives in `parameters.odin` (root package): `Parameter_Info{name, tensor, init, trainable}` with an `Init` union (`Init_He`, `Init_Xavier`, `Init_Normal{mean, std}`, `Init_Value{value}`, nil = never randomized). Generic procs: `register`, `registry_destroy`, `registry_clear` (names only — for loaders that replace tensors), `registry_randomize`, `registry_update`, `registry_copy`, `registry_parameters` (trainable entries only), `registry_parameter_count`. `register` asserts trainable ⇒ has gradient buffer and self-initializes the list allocator.
- `parameter_make(&list, prefix, name, type, shape, init=…, trainable=…) -> Tensor` allocates AND registers in one call, deriving the buffer set from `trainable` ({Data, Gradient} vs {Data}) — used by mlp/lora/llama so a parameter cannot be created without being registered. gemma deliberately keeps create-then-register: its registration must live solely in `_register_parameters` so the post-`load_gguf` rebuild re-runs the exact same walk; `parameter_make` there would fork registration truth into two drift-prone walks. `ml.make` remains for standalone (non-parameter) persistent tensors, e.g. in tests.
- `trainable` is explicit at registration, NOT derived from `has_gradient`: jepa's EMA target encoder has gradient buffers but must never be optimizer-updated (weight decay would corrupt it).
- Each module owns its registry; composition prepends prefixes at `parameters` view time (mlp inside jepa, lora adapters inside gemma). Tied tensors register once; aliases never register.
- mlp/lora/jepa converted with byte-identical checkpoint names (asserted by `tests/registry_check.odin`, which also leak-checks the full lifecycle). jepa needed zero changes.
- llama and gemma: five-way walks deleted, both now implement `parameters` (checkpointable for the first time). Names are the HF safetensors names from their loaders. gemma's three LoRA ladders (`update_lora`/`randomize_lora`/`lora_parameter_count`, no external callers) are deleted — the trainable-only registry view makes QLoRA selection automatic in the unified `update`/`randomize`/`parameters`.
- gemma `load_gguf` rebuilds the registry (`registry_clear` + `_register_parameters`) after loading, because the quant path destroy-and-replaces tensors, which would otherwise leave stale handles in the registry.
- Forward passes merged: llama `forward`/`forward_cached` → one `_forward(model, tokens, cache=nil)`; gemma `forward_with_hidden`/`forward_cached` → one `_forward` preserving all divergences (notably: fused `_gate_up_geglu` only on the cached path with no gate/up LoRA — the uncached path always used the manual ladder; final-cast behavior differs per path and is kept verbatim).
- Shared helpers folded into `ml`: `per_head_rmsnorm` (was llama `_per_head_rmsnorm` ≡ gemma `_qkv_norm`, which took an unused model param), `Kv_Cache`/`Kv_Layer_Cache` + `kv_cache_destroy`/`kv_cache_reset` (`llama.Cache` and `gemma.Cache` are aliases; makes stay model-specific for sliding-window/shared-layer capacity logic), and `const_scalar` (was gemma `_make_const_scalar` ≡ lora's inline scale baking).
- Acceptance test `tests/gemma_lora_check.odin`: tiny gemma with QLoRA runs a real CPU forward/backward/Adam step, asserts the parameters view is exactly the 8 LoRA tensors, round-trips a checkpoint including Adam moments and iteration into a fresh model, and shows zero leaks under `mem.Tracking_Allocator`.
- Verified: `odin check` clean everywhere, `odin test tests` green (7 tests, also under `ML_CPU_POISON`), `odin test tests\parity` green on GPU.

Original plan kept below for reference.

## Phase 2 (original) — Parameter registry (highest-leverage refactor)

### Problem

There is no module/parameter abstraction. `networks/llama/llama.odin` hand-writes five parallel walks of the same field tree: `make` (:58), `destroy` (:94), `copy` (:119), `randomize` (:144), `update` (:326). `networks/gemma/gemma.odin` does the same at ~3x scale (:194, :332, :400, :464) plus three more LoRA ladders (`update_lora`/`randomize_lora`/`lora_parameter_count`, :505–542). Any new field must be added to every walk; forgetting one is a silent leak or a frozen weight. Worse: **neither llama nor gemma implements `parameters`** (only mlp/lora/jepa do), so `ml.checkpoint_save`, `ml.clip_gradient_norm`, and `ml.parameters_read/write` are unusable with the two flagship models — you cannot checkpoint a llama fine-tune or save gemma LoRA adapters today.

### Design

Build a registry in `ml` that a model fills once at construction:

- Extend `Parameter` (ml.odin:394) or add a `Parameter_Info` carrying: full dotted name, tensor, an init spec (enum: `He`, `Xavier`, `Normal{mean, std}`, `Zeros`, `Ones`, `None`), and a `trainable: bool` (LoRA freezes base weights — this replaces the hand-rolled "only update LoRA tensors" ladders).
- A model's `make` registers every tensor as it creates it (a helper like `register(&model.params, prefix, name, tensor, init=..., trainable=...)`). Then the following become generic one-liners over the list: `destroy`, `randomize` (dispatch on init spec), `update` (skip non-trainable), `parameters` (filter/clone), `parameter_count`, checkpoint save/load, gradient clipping.
- Naming should match the checkpoint/safetensors conventions already used by mlp/lora/jepa (`lora.odin:89` uses PEFT-style names — keep that, it makes adapters interoperable).

### Tasks

1. Add the registry types + helpers to `ml.odin` (or a new `parameters.odin` in the root package).
2. Convert `mlp`, `lora`, `jepa` first (small, already have `parameters` — behavior must not change; jepa's checkpoint format must stay loadable or be explicitly version-bumped).
3. Convert `llama` and `gemma`: delete the five-way walks, implement `parameters` for both, keep the tied-embedding aliasing explicit (tied tensors register once; the alias is a view, not a second parameter).
4. Wire gemma LoRA saving: `checkpoint_save` over the LoRA-only parameter subset.
5. Fold the duplicated helpers exposed by this refactor into `ml` or a shared spot: `_per_head_rmsnorm` (llama.odin:216) ≡ `_qkv_norm` (gemma.odin:595); the KV `Cache` struct (llama.odin:174–213 ≡ gemma.odin:604–651); `_make_const_scalar` (gemma.odin:317 ≡ lora.odin:34–44).
6. Merge the duplicated forward passes: llama `forward` vs `forward_cached` (:284 vs :225) and gemma `forward_with_hidden` vs `forward_cached` (:787–890 vs :660–784) each differ only in attention call + position offset — unify into one forward parameterized by an optional cache. Watch the small existing divergences (gemma cached path has the fused-vs-LoRA gate/up branch at :738–750; the uncached path repeats the LoRA ladder) — the unified version must preserve both behaviors.

Acceptance: llama and gemma expose `parameters`; `checkpoint_save`/`checkpoint_load` round-trips a gemma-with-LoRA model including optimizer state; `mem.Tracking_Allocator` shows zero leaks on make → load → train-step → destroy; the five-way walks are gone.

---

## Phase 3 — Core API cleanups

### 3.1 Optimizer

- Move hyperparameters into the struct, set once (an `optimizer_init` or config struct); `optimizer_step` should not take `learning_rate`/`beta1`/... per call. Keep the ability to change LR per step (schedules) via a field write or explicit setter.
- Separate gradient accumulation from configuration: the `period` counter (ml.odin:552–579, default 128 while every caller passes 1) conflates the two. Either make accumulation explicit at the call site or default `period=1`.
- Keep AdamW as the implementation but shape the API so an SGD/momentum variant can be added without redefining `Optimizer` (a `kind` enum + variant params is enough; do NOT build a premature abstraction hierarchy). Backend `update` already takes the whole `Optimizer` — extend, don't redesign.
- Note: CPU `update` zeroes gradients as a side effect (cpu.odin:484–519). Verify CUDA `adam_*.cu` matches, then document this in the `Backend.update` contract (it is currently implicit).

### 3.2 Loss reduction semantics

`backward` seeds the loss gradient with all-ones (ml.odin:848–852), so an unreduced loss silently trains with sum-reduction — `examples/mnist/main.odin:123` does exactly that (effective LR scales with batch size) while cartpole takes `mean` explicitly. Decide the contract: either (a) `backward` requires a scalar loss (assert `loss.count == 1`, breaking-change, forces explicit reduction — recommended for a general-purpose library), or (b) keep sum semantics but say so at `backward`. If (a), fix mnist accordingly.

### 3.3 Tensor construction and I/O ergonomics

- `ml.tensor` only takes a flat `[]f32` and yields 1-D (ml.odin:266); every call site writes `reshape(tensor(x), {b, d})`. Add a shape parameter: `tensor(data, shape)` (keep the 1-D overload via proc group).
- Add a way to take the last row (or a leading-dim slice) of a 2-D result on-device: `examples/llm_chat/main.odin:172–182` copies the entire `[tokens, vocab]` logits to host to keep one row (~64 MB per prefill chunk at Gemma's 262k vocab). A `slice_leading(input, start, end)` op (mirror of `slice_trailing`) covers it. CPU + CUDA forward at minimum; backward optional initially (assert like other forward-only ops but see 3.5).
- `argmax` over the trailing dim (returns host `[]int` or an int tensor once those exist — see Phase 6): mnist (`main.odin:112–116`) and jepa hand-roll it on host today.

### 3.4 Derive the op-list boilerplate

`Operation_Variant`, `Operation_Kind`, `OPERATION_SET_ALL`, and `operation_kind` (ml.odin:643–803) are four hand-maintained parallel lists; the `#assert` at :751 checks only counts. Improvements: derive `OPERATION_SET_ALL` as `~Operation_Set{}` or a loop over the enum; replace the `operation_kind` switch with union-tag reflection if Odin allows deriving the mapping (investigate `intrinsics`/`reflect` — variant order in the union matching enum order can be exploited and statically asserted per-variant with `intrinsics.type_variant_index_of`). If full derivation isn't possible, at least statically assert per-op that variant index == enum value so mapping mistakes are impossible.

### 3.5 Explicit dtype/backward support tables

Op support is patchy with no table: CUDA `Exp/Clamp/Min/Softmax/Entropy/Cross_Entropy` are F32-only (ops.odin:696–1587 asserts), `Rmsnorm_Rope_Write_Cache`/`Attention_Cache` Bf16-only; CPU `Sqrt` F32-only; CPU panics on nine backward arms (cpu.odin:619–647) while the front end's `training` checks are the only thing keeping training away from them. Make the invariants checkable: front-end procs should assert dtype support up front (uniformly, like `transpose` does), and the "web of implicit invariants" (front-end decomposition rules ↔ backend capability sets) should be exercised by a test that walks every op × dtype × backend and confirms either clean support or a clean assert (see Phase 5).

---

## Phase 4 — Re-layer the op set (general-purpose core, LLM ops as extension)

The op union is currently a transformer instruction set: `Rmsnorm_Rope_Write_Cache`, `Linear_Q4_K_Gate_Up_Geglu`, `Attention_Cache` (sliding-window ring semantics) are user-facing core variants, while there is no axis-generic reduce, no general broadcast, no dropout, no conv. The decomposition mechanism already in place is the right tool: keep fused ops but treat them as an extension tier that always has a composed-of-core-ops fallback (most already do; `Attention_Cache` and `Rmsnorm_Rope_Write_Cache` are currently fused-or-nothing — give them decompositions or explicitly document them as inference-only extensions).

Add to the core, in this order (each with CPU F32+Bf16 forward/backward, CUDA at least F32, a `cases.odin` entry, and parity coverage):

1. **Axis-generic reductions**: `sum`, `mean`, `max_reduce` over a chosen axis (current `mean` only drops the trailing dim — jepa needed `mean(transpose(x))` to reduce over batch, jepa.odin:421–423, and `transpose` is rank-2 F32-only). Implementation note: reduce over the trailing axis + a cheap permute/view is an acceptable first implementation.
2. **General broadcasting** for the elementwise binary ops. Current rule is "b equals a's trailing shape or is scalar" (ml.odin:1059–1068); `min`/`max` don't even get that (same-shape only) — at minimum unify min/max with the others, at best implement NumPy-style broadcasting. The centralized broadcast-tiling helper (commit aa2650c) is where this generalizes.
3. **Dropout** (training-only op, identity in inference; needs RNG plumbing — see Phase 6 RNG).
4. **Bias in `linear`** (optional bias tensor in the `Linear` variant; mlp currently does a separate `add`, llama/gemma omit bias entirely).
5. **Embedding as a first-class name**: `select` already is the lookup; consider a `embedding(table, ids)` alias so users find it, or document it.
6. **`slice_leading` / `argmax`** from 3.3.
7. **conv1d/conv2d + max/avg pooling** — only after the above; this is the gate for non-transformer workloads (CNNs). CPU first with a straightforward im2col or direct loop implementation, CUDA via cuBLAS-backed im2col. Do not hand-write a cuDNN competitor.

Also fill the CUDA coverage gap for plain training: `forward_ops`/`backward_ops` (backends/cuda/cuda.odin:189–200) omit Sub, Div, Sqrt, Max, Mean, Transpose, Slice, Concat, Layernorm, Log_Softmax, MSE, Smooth_L1, Relu, Sigmoid, Batched_Matmul, Permute, Causal_Mask, Lerp_Assign, Accumulate_Mean — a relu+MSE MLP cannot train on GPU today. Port these kernels (most are trivial elementwise; consider a shared elementwise-kernel template with an NVRTC `-DOP=` define instead of 19 new files — see 4.1).

### 4.1 Kill the dtype-clone duplication while you're in there

Roughly half of cpu.odin (~3600 lines) is F32/Bf16 hand-copies of the same op (e.g. `add_forward` :731–760, `rope_forward_f32/_bf16` :1616–1709, attention ×4). Parametrize with `$T` generics plus the existing `_unary_forward_dispatch` pattern (cpu.odin:2627). On CUDA, most kernels exist as f32/bf16 file pairs — collapse with an NVRTC `-DDTYPE=` option (the mechanism exists: `D_HEAD` at ops.odin:1362) once 1.7's cache-key fix lands. Shared device helpers (bf16 round-to-nearest-even with NaN handling is duplicated per kernel) belong in `kernels/common/` next to `broadcast.cuh`.

Do this as a mechanical refactor with parity tests green before and after; it halves the maintenance surface before the new ops from this phase multiply it.

---

## Phase 5 — Test suite hardening

Current state: idiomatic `@(test)` suites; good shared case registry (tests/cases/cases.odin:56–91) driving both grad-check (central difference, careful `h` selection) and CPU↔CUDA parity; Adam and clip checks against analytic references. But coverage stops exactly where risk starts.

1. **Cover the 11 untested ops** — `Linear_Q4_K`, `Linear_Q4_K_Gate_Up_Geglu`, `Linear_Q6_K`, `Rmsnorm_Rope`, `Rmsnorm_Rope_Write_Cache`, `Add_Rmsnorm`, `Gelu_Mul`, `Attention_Cache`, `Cast`, `Lerp_Assign`, `Accumulate_Mean` — exactly the fused/quantized kernels the LLM path runs. For fused ops, the oracle is the unfused composition (the front end can be forced onto the composed path by masking `forward_ops`). For quant linears, the oracle is dequantize (quants.odin) + plain linear. CUDA declares backward for `Linear_Q4_K`/`Linear_Q6_K` (cuda.odin:197) with no CPU reference — oracle via dequantized-weight linear backward.
2. **Bf16 coverage**: unit tests for `bf16_from_f32` (round-to-nearest-even, NaN); duplicate a representative subset of op cases at Bf16 with appropriate tolerances; parity for the Bf16 kernels.
3. **Golden/round-trip tests**: checkpoint save→load round-trip including Adam moments and `iteration` (pairs with 1.4); Q4_K/Q6_K dequantization against reference block values; tokenizer golden tokens for a handful of strings vs reference implementations (this will catch the `▁` prefix issue in 1.9); a tiny fixed-weight MLP forward compared elementwise to precomputed values.
4. **Fix the vacuous-green parity suite**: `_cuda_available` failure just logs and passes (tests/parity/parity.odin:150–153, 262–266). `core:testing` has no skip status — gate with a `-define:ML_REQUIRE_CUDA=true` so CI/GPU boxes fail loudly when CUDA is absent, and log an unmissable SKIPPED banner otherwise. Also add an assertion that every case whose op is in CUDA `forward_ops` actually got parity-checked (the silent per-op skip at parity.odin:162–165 means adding a case doesn't guarantee coverage).
5. **Shape sweep**: each case currently tests one shape (e.g. add only (2,3)+(3)); add scalar broadcast, rank-3, and sizes crossing CUDA block/tile boundaries (>= a few thousand elements; attention with T > one block).
6. **Loader robustness tests**: truncated/malformed gguf and safetensors headers must error, not crash (pairs with 1.9 overflow fixes).
7. **Hygiene**: dedupe `_adam_grad`/clip reference code copy-pasted between adam_check.odin:19 and parity.odin:291; make `cases.get()` thread-safe (lazy global build at cases.odin:39–54 is racy if a second test calls it — `sync.Once` or eager init); test Adam with `period > 1`; add a determinism test (same seed → identical outputs, needs Phase 6 RNG) and a `mem.Tracking_Allocator` leak test around a full train step; write a `tests/README.md` or root doc section recording how to run both packages.

---

## Phase 6 — Breadth: what "general purpose" still needs

Ordered by value; each is a self-contained work item.

1. **Sampling/generation utilities** (new package, e.g. `sampling/`): temperature, top-k, top-p over a logits row, plus a generate loop with stop tokens and prefill chunking. Extract from `examples/llm_chat/main.odin:184–275` and `backend_gemma.odin:117–134`; the example then shrinks to wiring.
2. **RNG control**: a seeding surface. `fill_normal` uses the global `core:math/rand` (ml.odin:435) so reproducibility is inexpressible. Thread a `rand.Generator` through Context or take it as a parameter with a default; dropout (Phase 4) needs a device-side story on CUDA (Philox counter-based or precomputed masks — precomputed host masks are an acceptable first cut).
3. **Dataset/batching helpers** (new package, e.g. `data/`): index shuffling, minibatch iteration, train/val split. Extract the patterns from `examples/mnist/main.odin:48–77,173–251`. Keep it small — slices and iterators, not a framework.
4. **Error handling at the library boundary**: loaders and checkpoint I/O already return `bool`/log — extend to typed errors where a caller could react. Backend-internal CUDA/NVRTC failures may keep panicking (research-harness pragmatics), but document that contract. Do NOT convert shape asserts to errors — asserts with `#caller_location` are the right call for programmer errors.
5. **Attention limits**: training attention materializes the full `[H, T, T]` f32 softmax on both backends with hard caps (`token_count <= 2048` at ops.odin:1233/1252/1810, `head_size <= 512` at :1216/1279; cuda.odin:436, cpu.odin:526). A flash-attention backward removes the O(H·T²) memory and the cap. Large task; do only when long-context training matters.
6. **Integer tensor dtype**: cross-entropy targets are host `[]int` smuggled through the op variant (ml.odin:1916–1919) and re-uploaded every step (ops.odin:1592); `select` indices likewise. An `I32` dtype makes targets/indices resident and unlocks argmax-as-tensor. Medium-size change touching `Data_Type`, buffers, and both backends — schedule deliberately.
7. **Config/toolchain robustness**: NVRTC include paths hardcode CUDA 12.5/12.6 on Windows (pipeline.odin:28–29) — probe `CUDA_PATH` env var first; make `--use_fast_math` (pipeline.odin:99) configurable per context (training may want it off).
8. **Multi-device**: currently device 0, one global `_gpu` (cuda.odin:98,123), no cross-backend `ml.copy` (ml.odin:347). Explicitly OUT OF SCOPE for this plan; do not half-build it. Cross-backend copy (CPU↔CUDA staging) is the one piece worth adding early since it's needed for device data pipelines.

---

## Phase 7 — Threading contract

Stated user requirement: the library should be usable from multiple threads without the user having to worry much. The architecture is already close: `_current_ctx` is `@(thread_local)` (ml.odin:121), and the CUDA backend resolves all per-context state by casting the current context (`_gctx`, backends/cuda/cuda.odin:102), so each context owns its tape, op arena, activation pool, CUDA stream, and cuBLAS handle. Global GPU state is `_gpu_mutex`-guarded for create/destroy and all buffer ops. What is missing is a stated contract, enforcement, and a few internal-sync gaps.

### 7.1 Define the contract (documentation + asserts, not new machinery)

- Tier 1 (guaranteed): a `Context` and every transient tensor created under it belong to one thread at a time. Each thread creates its own context. All library-internal globals (NVRTC pipeline cache, CUDA buffer pool and primary context, CPU worker pool) are internally synchronized — N threads each with their own context require no user locking.
- Tier 2 (guaranteed with a rule): persistent tensors (parameters, `make`) may be shared read-only across contexts/threads for inference. Writes (weight upload, checkpoint load) must complete before other threads read; define the sync point (the writing context's stream is synchronized by `buffer_get`/`context` teardown — specify exactly which call publishes, likely "any `buffer_get` on the writing context or an explicit `ml.synchronize()`" — add `synchronize` if no clean publish point exists).
- Tier 3 (explicit non-goal for now): concurrent training on shared parameters. Backward accumulates gradients (`+=`) and the optimizer state map (`Optimizer.state`, ml.odin:503) is unsynchronized — data-parallel training is per-thread replicas + explicit reduce, delivered later as a utility (see 7.5), never implicit.

Record the contract in the README/root docs, not comments.

### 7.2 Close the internal synchronization gaps

- Items 1.7 (pipeline cache lock + key) and 1.8 (per-thread `CtxSetCurrent` binding) are prerequisites for Tier 1 — treat them as part of this contract.
- CPU worker pool (backends/cpu/cpu.odin:175–210): one global pool; concurrent `parallelize` from two contexts serializes behind `_pool_mutex`. Keep that (it prevents oversubscription) but document it as the policy. Add an assert against calling `parallelize` from inside a worker thread (today that deadlocks: the worker would wait on `_done_wg` it participates in). A thread-local `_in_worker` flag is enough.
- Verify whether Odin's default `context.random_generator` is thread-safe / per-thread; the RNG work in Phase 6.2 must not introduce a cross-thread race via `fill_normal`.
- `cases.get()` lazy-build race (Phase 5.7) — same theme, test-side.

### 7.3 Cheap enforcement so violations are loud

- Stamp the context with its owning thread id in `context_begin`; assert it matches in `_record_forward`/`clear`/`backward` (debug-gated via a `-define` if the check shows up in profiles — it won't; it's one compare).
- Transient tensors: stamping every tensor is intrusive; instead give `Context` a monotonically increasing `clear_generation` and (debug-define only) stamp transient allocations with it, asserting on op input that the generation matches. This catches both cross-thread transient use and stale-tensor-after-`clear` on a single thread — the second is a footgun even single-threaded.
- `context_begin` on a context that is already active on another thread should assert (a simple `owner_thread_id != 0` check).

### 7.4 Prove Tier 2: shared-weights parallel inference test

Add a test (GPU-gated like parity): load or randomize one small model's persistent weights, spawn N threads each with their own context + KV cache, run identical decode steps concurrently, assert all threads produce the identical token sequence and that it matches a single-threaded run. On CPU too (pool contention makes this a correctness-under-serialization test). This is the test that makes the "don't worry about threads" promise real.

### 7.5 Data-parallel training utility (later, after Phase 2)

Per-thread model replicas + a reduce step. With the Phase 2 parameter registry this is small: `parameters_read`/`parameters_write` already give flat views; add a `parameters_reduce_mean(replicas: [][]Parameter)` (or gradient-level equivalent) and let one thread own the optimizer. Do not attempt lock-free shared-parameter training.

---

## Execution notes for the implementing agent

- Work the phases in order; within a phase, items are independent unless noted. Phase 1 items are individually committable ("give a one-line message and let the user commit" per CLAUDE.md).
- For every fix: re-read the cited code first (lines have drifted), confirm the failure path actually exists, write/extend the test, then fix. Several findings above were verified by direct read (marked VERIFIED); the rest came from review agents and must be re-confirmed.
- Run `odin check` per touched package and both test packages after each item. CUDA-touching changes need a GPU run of `odin test tests\parity`; if no GPU is available, say so explicitly rather than reporting green.
- Per CLAUDE.md: spawn an Opus sub-agent for large edit arcs (Phase 2, Phase 4.1) and adversarially review its output.
- Do not add comments to code. Contracts that must be written down (e.g. the `zeros` semantics decision in 1.5, backend `update` gradient-zeroing) go in doc-comment-free form: either this file, a README, or assert messages.
- When a decision point is marked with options (1.3, 1.5, 3.2), pick the recommended option unless implementation reveals a blocker; record the choice by updating this file.
- Update this file as items complete (strike through or move to a Done section) so it stays the source of truth.
