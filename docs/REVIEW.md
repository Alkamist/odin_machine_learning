# Library review — 2026-07-19

Full-codebase review (~18k lines Odin + CUDA kernels) covering the core engine, CPU backend,
CUDA backend, loaders/tokenizers/sampling, networks, and examples. Every high-priority finding
was verified against the source before inclusion.

**Verdict:** well-engineered library. The math is overwhelmingly correct, the file parsers are
more disciplined than typical hand-rolled loaders, and the test architecture (shared op-case
registry driving both CPU gradient checks and CPU-CUDA parity, plus the NaN-poison define) is a
standout. Problems cluster in lifecycle/ownership contracts, silent-failure paths, and the GPT-2
tokenizer diverging from its reference.

---

## High-priority issues (all confirmed)

1. **`safetensors.save` can destroy the only copy of a checkpoint.**
   `loaders/safetensors/safetensors.odin:330-334` removes the destination, then on rename failure
   also removes the tmp file — both old and new checkpoints gone. A locked target file on Windows
   (AV scan, another process) is enough to trigger it. The `os.remove` error is also ignored.

2. **Calling `update` before the first `optimizer_step` silently NaNs weights.**
   `bias_correction1/2` are zero until `optimizer_step` runs (`optimizer.odin:117-118`); CPU
   `_update` divides by them unconditionally (`cpu.odin:503-504`). The natural
   `backward(); registry_update(...)` loop without the `if optimizer_step(...)` gate produces
   inf/NaN with no diagnostic.

3. **The CPU SIMD fast path has never compiled.** Reproduced:
   `odin build backends/cpu -microarch:x86-64-v3` fails with five "Undeclared name: simd" errors —
   `cpu.odin:238-277` uses `simd.fma`/`simd.reduce_add_bisect` without `import "core:simd"`. The
   default target skips the `when` branch, so `odin check` passes and every build so far has run
   the scalar fallback. The whole `_simd_*` layer (including the bf16 shuffle) is dead, untested
   code. After fixing, spot-check numerics — it changes who computes every dot product.

4. **GPT-2 tokenizer: special tokens never emitted.** In `tokenizers/gpt2/gpt2.odin`,
   `added_tokens` is loaded (line 120) and freed (line 133) but never consulted in `encode`.
   `<|endoftext|>` gets pretokenized into `"<|" "endoftext" "|>"` and BPE'd into ordinary tokens.
   CORRECTION (verified against HF tokenizers 0.22.2 with the real SmolLM2 tokenizer.json): the
   originally-reported digit-splitting divergence was a false positive. SmolLM2's pre_tokenizer is
   `Digits(individual_digits=true)` + ByteLevel, so the per-digit splitting was a faithful
   implementation, not a bug. The only real digit-adjacent issue was the
   `unicode.is_digit`/`unicode.is_number` inconsistency (HF Digits uses Rust `char::is_numeric`,
   i.e. category N).

5. **Llama loader: untied model with no `lm_head.weight` loads "successfully" with random head
   weights.** `networks/llama/loader.odin:44-50`: when `!cfg.tied_embeddings` and the file lacks
   the tensor, neither branch fires and the proc returns `true`. Related: `use_qk_norm=true`
   allocates and applies q/k norm weights but the loader never loads them — latent trap.

6. **Embedding lookup on a quantized table silently returns zeros.**
   `backends/cpu/ops_shape.odin:43-53`: `_select_forward` computes `row_bytes` from
   `data_type_size`, which returns 0 for `.Q4_K/.Q6_K`, so every copy is a no-op. `select` /
   `embedding` accept any dtype at the op layer, and a quantized embedding table is the normal
   GGUF layout.

7. **`attention_with_cache` never validates `key.type`.** `ops.odin:178-180` checks Q==V and
   k_cache==v_cache but not `key` — which is raw-byte-copied into the cache and matmul'd against
   Q. An F32 key with a Bf16 cache writes 4*kv_size bytes per row into rows sized 2*kv_size: a
   buffer overrun in `_attention_cache_forward`. The non-cache `attention` checks all three.

8. **GGUF: unbounded `n_dims` allocation from a hostile file.** `gguf.odin:194` allocates
   `make([]int, n_dims)` from an untrusted u32 before validation — `n_dims = 0xFFFFFFFF` requests
   ~34 GB from a tiny file. Everything else in the parser is well bounds-checked.

9. **CUDA: teardown while auto graph-capture is armed panics.** `clear()` re-arms stream capture
   each decode step; `context_destroy` (`cuda.odin:296`) then calls `StreamSynchronize` on a
   capturing stream → error → `cuda.check` panics. Same for `buffer_free` of a persistent buffer
   mid-capture; `enable_decode_graph(false)` leaves the capture armed forever.

10. **CUDA kernel edge cases.**
    - `cast_f32_to_bf16.cu:10-13` writes 2 bytes past exact-sized destinations for odd element
      counts (activation pool uses exact-size slots; saved only by cuMemAlloc granularity).
    - Fallback `attention_cache_bf16.cu:86-98` does `uint4` loads requiring `head_size % 8 == 0`;
      host asserts only evenness. head_size 100 + odd kv-head index → MISALIGNED_ADDRESS fault.
    - F32 variants of `rmsnorm_rope.cu`/`add_rmsnorm.cu` decode the weight as packed bf16 —
      currently unreachable, dead-but-wrong. Delete or fix; assert `weight.type == .Bf16` host-side.

    Correction recorded during verification: the reported "silent K/V desync on window==0 cache
    overflow" is guarded by the assert at `ops.odin:186`; it only becomes silent corruption under
    `-disable-assert`. Real but lower severity.

## Design-level issues

- **Registry ownership is a whole-object boolean that lies.** `parameter_register` sets
  `owns_tensors = true` for tensors it did not allocate (`registry.odin:79`), so
  `registry_destroy` frees caller-owned tensors (currently a silent no-op only by CPU-backend
  luck). Registry-wide flag also means owned and borrowed parameters cannot mix, and one
  `parameter_make` poisons a registry against `registry_gather`. Should be per-parameter.

- **K-cache write handshake is the hackiest pattern in the codebase.** Correctness of
  `rmsnorm_rope_write_cache` + `attention_with_cache` depends on an invisible protocol: CUDA
  writes K in the fused op and records it in `k_cache_written_this_forward` so attention skips
  its write; CPU silently degrades the fused op to `rmsnorm_rope` and attention does the write.
  Nothing documents that a third backend must implement exactly one of the two.

- **Hidden core/backend contracts.** `clip_gradient_norm` allocates `ctx.grad_norm_accumulator`
  via the backend but core `_context_destroy` never frees it — every backend must remember to
  (CPU does, by reaching into the transmuted slice). `copy` compares backends by proc-table
  pointer, so tensors from two contexts of the same backend pass the check.

- **Assert/error boundary leaks.** Policy (errors for untrusted input, asserts for programmer
  error) is right and mostly followed. Exceptions: `checkpoint_save` asserts on quantized tensors
  with default flags (which include `.Checkpoint`) while `checkpoint_load` returns an error enum;
  `weights.set_floats` panics where siblings return false; KV-cache exhaustion is an assert so a
  long chat hard-crashes; `backward` silently no-ops on nil context / zero ops; several fused-op
  dispatch sites deref `_current_ctx` raw instead of `current_context(loc)`.

- **Gemma/llama drift.** Different config vocabularies (`hidden_size`/`embedding_size`,
  `vocab_size`/`vocabulary_size`, `for_training`/`trainable`), verbatim-duplicated cache wrappers
  and forward-prologue asserts, LoRA locked inside gemma despite `networks/lora` existing,
  `gemma.Config` heap-allocates `layer_types` (separate `config_destroy`, aliased-slice
  double-free trap) while `llama.Config` is a clean constant. `llm_chat` hand-builds a ~150-line
  vtable to abstract over them.

- **Cruft:** `jepa.plan`/`Planner_Config`/`train_decoder_step`/`update_decoder` (orphaned by
  TD-MPC migration), `gemma.forward_with_hidden`, `llama.copy`, CPU `_cast_bytes_accumulate`,
  several never-read CUDA fields/constants, hand-rolled int/float parsers in `llm_chat`,
  `sim.reset`/`destroy` duplication. `tests/README.md` references a root `README.md` that does
  not exist.

## Public API assessment

Good blend overall. MNIST is the proof: complete trainable model + batching + eval in ~130
readable lines. Thread-local context + flat tape gives PyTorch-like ergonomics without hidden
allocation. Builtin shadowing (`ml.len`, `ml.copy`, ...) reads well at call sites at the cost of
`builtin.` noise internally — fair trade. The `Registry` pattern is the standout (mlp is a
73-line complete model with checkpoint names for free).

Gaps:
- **`ml.clear()` discipline is the sharpest edge** — cartpole calls it 9 times across helpers;
  jepa needs the two-phase `clear(); ... clear(training=true)` dance per train proc; forgetting
  one is a silent bug. Wants a scoped construct (see plan Phase 3).
- **No last-position-logits option** — prefill computes the full-vocab lm_head GEMM per token and
  the chat example discards all but the last row.
- **KV-cache exhaustion** not surfaced as a recoverable condition.
- Minor: negative axes other than -1 silently mean "last axis" in `mean`/`sum`/`max_reduce`
  (`ops.odin:467`); `mean_squared_error`/`smooth_l1` check element count not shape; `slice`
  allows start==end (trips an unrelated assert) and flattens rank-N while `slice_leading/trailing`
  preserve rank; `transpose` F32-only while `permute` is not; arena-exhaustion message names
  nonexistent `OP_ARENA_DEFAULT_SIZE`; checkpoint partial adam_m/adam_v silently dropped;
  `registry_read/write` assert F32 while Bf16 params are trainable.
- Withdrawn: sampling RNG seeding — `core:math/rand` already takes `gen := context.random_generator`,
  so callers can inject a seeded generator via Odin's context. Just add a determinism test.
  (Sampling nits kept: `decode_tokens` counts a stop token as decoded; nil-logits `Eval_Proc`
  contract undocumented.)

## Additional minors (by area)

- **GGUF:** duplicate KV keys silently overwrite (tensor names are rejected); `get_array_str` is
  O(N^2) over string arrays; alignment not validated as power-of-two; tensor ranges not checked
  for overlap; one unlogged `.Malformed` path.
- **safetensors:** unknown dtypes skip size-vs-shape validation; `json.is_valid` + `json.parse`
  double-scans; written files don't pad header to 8-byte alignment.
- **weights:** rope permute assumes even head_size (odd → last row of each head silently zero /
  stale); `read_floats` returns temp-allocator memory with ownership unstated; full-f32
  materialization per tensor in temp allocator.
- **Tokenizers (both):** decode silently skips out-of-range ids while encode can emit them;
  duplicate vocab ids pass validation; O(n^3) worst-case BPE on long space-free segments; gemma
  normalizer spec in tokenizer.json ignored rather than validated.
- **CPU backend:** per-row heap alloc in Q4_K/Q6_K parallel jobs; cache-attention scratch grows
  with absolute position not window; `Attention` allocates O(H*T^2) softmax scratch even for
  inference; `_parallelize` work-gate only wired for f32 linear; FTZ only on pool workers +
  creating thread (contexts can migrate); `LAYERNORM_EPSILON` hardcoded in backend while rmsnorm
  threads `eps`; `context_destroy` doesn't assert persistent map empty.
- **CUDA backend:** `device_init`/`device_destroy` refcount asymmetry; `_include_arg` lazy-init
  data race (benign); timing procs skip `_gpu_mutex`; 32-bit index math in
  `cross_entropy_f32.cu`/`select.cu`/`select_back.cu` overflows past 2^31 elements; q8_1 `s`
  field matches neither ggml nor its own comment (unread today); early-return-before-shuffle UB
  patterns currently unreachable but uncontracted; ~50x duplicated 1-D launch boilerplate.
- **Core:** `context_end` can release ownership of a still-stacked context on re-entrant begin
  sequences; optimizer state keyed by raw buffer pointer (address reuse inherits stale Adam
  moments); `copy` doesn't check buffer-presence match (stale gradients survive); gradient
  accumulation sums rather than averages (interacts with clip threshold / epsilon — conscious
  decision wanted); checkpoint_save materializes all weights + moments in temp memory at once.
- **Networks/examples:** cartpole death-transition learning signal depends on decision-tick
  timing (mailbox drops terminal snapshots); `agent.stop` on never-started agent is nil-join;
  `snapshot.valid` never cleared (worker re-processes at idle-spin); mnist `_idx_read` doesn't
  validate rows*cols; `GEMMA_EOS_ID` hardcoded and missing `<|turn|>` becomes -1 in stop tokens
  silently; `gemma.odin:240` dtype_bytes ternary silently treats non-F32 as 2 bytes.

## Verified strong (adversarially checked, no issue)

Softmax/cross-entropy max-subtraction everywhere; layernorm/rmsnorm backward match standard
derivations incl. projection terms; CUDA q4_K/q6_K mmvq + dequant kernels match ggml bit-for-bit;
flash-attention online-softmax cross-warp combines correct; `bf16_from_f32` correct RNE with NaN
quieting; CPU thread-pool protocol sound; CUDA bindings ABI-clean (`_v2` names, size_t, i64
strides all right); GGUF/safetensors bounds+overflow discipline; loader error-path cleanup
complete; quants.odin dequant verified element-by-element against llama.cpp layout; sampling
top-k/top-p/temperature math correct; ring-cache wrap logic consistent; graph-exec
update-or-reinstantiate path correct.

---

# Fix plan

Each phase is a clean commit point. Verification for every phase: `odin check` all packages, full
`tests` suite (plus `-define:ML_CPU_POISON=true`), goldens; CUDA-touching phases also run the
parity suite on the GPU machine.

## Phase 1 — Data loss & silent corruption

1. `safetensors.save`: leave tmp on rename failure (log its path); check `os.remove` error.
   Test: locked destination.
2. Assert `opt.iteration > 0` in `update`, message pointing at `optimizer_step`.
3. Add `import "core:simd"`; run tests with `-microarch:x86-64-v3`; spot-check AVX vs scalar
   numerics. Add that flag to the test README as a supported config.
4. GPT-2: added-token segmentation before pretokenize (gemma-style longest-match); keep
   per-digit splitting (it is SmolLM2's spec'd pre_tokenizer) but use `unicode.is_number` to
   match HF Digits. Golden fixtures for special-token + numeric prompts generated with HF
   tokenizers.
5. Llama loader: fail loudly on untied + missing lm_head; delete `use_qk_norm` until a config
   needs it.
6. Assert non-packed dtype in `_select_tensor`.
7. `attention_with_cache`: assert `key.type == query.type` and `key.type == k_cache.type`.
8. GGUF: reject `n_dims > MAX_TENSOR_RANK` before allocating.
9. CUDA: finish/abort armed capture in `context_destroy`, persistent `buffer_free`,
   `enable_decode_graph(false)`.
10. CUDA kernels: 2-byte tail store in `cast_f32_to_bf16.cu`; `%8` gate (or scalar fallback) in
    `attention_cache_bf16.cu`; delete F32 variants of rmsnorm_rope/add_rmsnorm + host assert
    Bf16 weight.

## Phase 2 — Ownership & lifecycle

11. Per-parameter ownership: `.Owned` in `Parameter_Flags`, set only by `parameter_make`;
    `registry_destroy` frees only owned; delete `owns_tensors` and the `registry_gather`
    restriction.
12. Free `grad_norm_accumulator` in core `_context_destroy`; document backend contract that
    `buffer_free` works on an inactive-but-live context.
13. K-cache protocol (decision, revised during implementation): explicit per-op flags instead of
    a mandatory fused op — the original "uniform op" idea did not cover V-writes or KV-shared
    layers. `rmsnorm_rope_write_cache` returns `wrote_cache: bool`; `attention_with_cache` takes
    `k_already_cached` / `v_already_cached`, stored in the op variant. Gemma threads the bool
    through and passes both flags true for KV-shared layers. Deletes the hidden
    `k/v_cache_written_this_forward` sets from the CUDA context entirely — cache-write
    responsibility is now encoded in the recorded graph, visible to any backend.
14. Assert/error boundary: `checkpoint_save` returns error enum; `weights.set_floats` returns
    false; `backward` asserts on nil context / zero ops; fused dispatch uses `current_context(loc)`.
15. KV-cache exhaustion recoverable: `forward_cached -> (logits, ok)` or `cache_remaining` idiom
    (shape decided in Phase 4; plumbing here). Chat example trims or refuses.

## Phase 3 — Scoped pass construct

```odin
@(deferred_out=_pass_end)
pass :: proc(training := false, loc := #caller_location) -> ^Context {
	clear(training=training, loc=loc)
	ctx := current_context(loc=loc)
	ctx.pass_open = true
	return ctx
}
```

- `_pass_end` clears `pass_open`; `_record_forward` asserts `pass_open` → building ops outside a
  pass (today's silent stale-graph bug) becomes a loud assert at the call site.
- `backward` additionally asserts a training pass.
- `ml.clear()` kept during migration (also sets `pass_open`); migrate examples/jepa/cartpole,
  then decide whether bare `clear` survives.
- Open question (lean no): should `_pass_end` assert a training pass called `backward`? Catches
  "forgot backward" but forbids legitimately abandoning a pass.

## Phase 4 — API pass (batched decisions; design before code)

- Optimizer shape: `registry_step(opt, registry) -> stepped: bool` doing gate + corrections +
  update (+ clip ordering) internally; existing pieces stay public for custom loops.
- Gemma/llama convergence: one config vocabulary; shared cache wrappers + forward-prologue
  asserts in `ml`; LoRA plumbing extracted into `networks/lora` for both; derive
  `gemma.Config.layer_types` from layer index so Config becomes a value type (`config_destroy`
  dies, double-free trap with it).
- Common LM interface: formalize the `llm_chat` vtable as a `Language_Model` struct of proc
  pointers next to the networks.
- Last-position logits: `Logits_Mode.All / .Last` param on `forward_cached` (or hidden + explicit
  lm_head — decide at implementation). Delete `forward_with_hidden`.
- Sampling: determinism test via seeded `context.random_generator`; fix decode-token count nit;
  document nil-logits `Eval_Proc` contract.

## Phase 5 — Hardening & cruft sweep

- Loaders/tokenizers: GGUF duplicate-key rejection, `get_array_str` O(N^2) fix, safetensors
  unknown-dtype validation, weights odd-head_size rope assert, `read_floats` allocator parameter,
  decode range checks, added-token id validation.
- Core minors: negative-axis assert, MSE/smooth-L1 shape assert, `slice` start<end + rank
  consistency, `transpose` dtype parity, arena message constant name, checkpoint partial-moment
  warning, `registry_read/write` Bf16.
- Dead code deletions: jepa planner remnants, `llama.copy`, `_cast_bytes_accumulate`, CUDA
  never-read fields/constants, `llm_chat` parsers → `core:strconv`, `sim.reset` duplication.
- Root `README.md`: threading contract (already cited by tests/README), backend contract
  (Phase 2), pass discipline (Phase 3).

Rough sizing: Phases 1-2 ~a day each, Phase 3 small, Phase 4 largest (gemma/llama convergence),
Phase 5 mechanical.
