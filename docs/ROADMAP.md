# Roadmap

Gap analysis from the 2026-07-19 full review, whose fix arc is complete. Each item is
self-contained enough to pick up cold: current state, design sketch, files, and how to
verify.

The mission is two tracks — local quantized LLM **inference** and small-model
**training** / continual-learning experiments — so there is no single return axis to
sort by. Within each track, the ordering rule is:

1. **Silent-correctness bugs first** (wrong output that looks fine).
2. **Loud failures next** (can't load / can't run, but you know it).
3. **Capacity ceilings last** (works, but not at the size you want).

## Known defects (fix regardless of feature order)

Latent correctness smells, each a near-one-liner, none gated behind a feature landing:

- `quantize_q8_1_bf16.cu` stores `s = d * sum(x)`, which matches neither ggml
  (`sum(x)`) nor its own comment. Currently unread by any consumer — fix or document
  before anything reads `ds.y`.
- `attention_long` parity case is time-seeded and rarely grazes tolerance. Pin the seed
  (see `tests/parity`).

# Inference track

## I1. tokenizer.json unsupported-spec rejection (silent-correctness)

**Problem.** Tokenizers are hand-ported per model family (`tokenizers/gemma`,
`tokenizers/gpt2`), and the tokenizer.json normalizer/pretokenizer specs are ignored
rather than interpreted. The SmolLM2 digit-splitting incident during the review is the
failure mode: the file declared `Digits(individual_digits=true)` and the hardcoded port
was almost "fixed" away from correct behavior. This is the dangerous class — a hand-port
that silently disagrees with the spec produces plausible-but-wrong ids.

**Design.** Before building any interpreter, close the silent-divergence hole cheaply: at
load time, parse the tokenizer.json `normalizer` / `pre_tokenizer` / `model` / `decoder`
sections enough to enumerate the combinators present, and **hard-error** on any the
active hand-port does not implement. This is a much smaller deliverable than a full
interpreter and eliminates the entire failure class on its own — an unsupported spec
becomes a loud load error instead of a wrong tokenization.

**Verify.** Feed a tokenizer.json with a combinator the port lacks; load must fail loud.
Existing goldens (`tests/golden/golden.odin` gemma, `tests/golden/gpt2_check.odin`
SmolLM2/HF-pinned) must still pass.

## I2. GGUF dtype coverage (loud-failure)

**Problem.** Only Q4_K / Q6_K / BF16 / F32 tensors load (`loaders/weights`,
`backends/cuda/kernels/linear/`, `quants.odin`). Much of what people actually download
is Q8_0, Q5_K-mix, or F16 — "run any GGUF" fails more often than it succeeds.

**Order of value.** Q8_0 (simplest block format, very common), F16 (trivial — decode
path only), Q5_K (rounds out the K-quant mixes; the scale/min unpack scaffolding from
Q4_K transfers). Each needs: block layout in `quants.odin` (verify element-by-element
against llama.cpp `dequantize_row_*`, as was done for Q4_K/Q6_K), a dequant path in
`loaders/weights`, CUDA mmvq or dequant-to-bf16 kernel (`Q8_1_BLOCK_*` quantize stream
already exists for the dp4a path), CPU linear path, and a golden block fixture in
`tests/golden` plus a parity case. The parity gate ("every registry case's op is in
CUDA forward_ops") keeps coverage honest automatically.

## I3. Sampling quality knobs (interactive-use gap)

**Problem.** `sampling/sampling.odin` has temperature/top-k/top-p only. Small models
loop badly without repetition control; these are table stakes for interactive use.

**Add** (in rough order): repetition penalty (needs recent-token window — thread
`out_tokens` or a ring of recent ids into `sample`), presence/frequency penalties,
min-p. Keep the existing shape: pure functions over a logits row, options in `Sampler`.
The `generate` loop already owns the token history. Seedable determinism is already
solved via `context.random_generator` (pinned by
`tests/sampling_check.odin:test_sample_deterministic_with_seeded_generator`).

## I4. tokenizer.json interpreter (deferred — build on the 3rd family)

**Problem.** Once I1 lands, every new model family still means a new hand-written
tokenizer. That cost is real but only bites when a third family arrives; building a
general interpreter speculatively competes with the mission (same logic as the deferred
`Language_Model` vtable — revisit when a third consumer exists).

**Design.** One `tokenizers/hf` package that parses and interprets the small set of
combinators covering the popular families:
- pre_tokenizers: `Sequence`, `ByteLevel` (with/without regex), `Digits`, `Split`
  (the GPT-2 regex as a special case, as today), `Metaspace`, `Whitespace`.
- normalizers: `Replace`, `NFC`/`NFKC` (core:unicode has decomposition tables; if NFKC
  is too heavy at first, validate-and-reject instead of silently ignoring).
- model: BPE with merges (both current implementations already have correct merge
  loops to reuse); byte_fallback.
- Longest-match added-token segmentation is already implemented twice; write it once.

**Migration.** gemma and gpt2 packages become thin wrappers or get deleted once the
interpreter passes their goldens. Keep the golden-fixture pattern: generate reference
ids with HF `tokenizers` (available locally: `python -c "import tokenizers"`), pin them
in `tests/golden/`.

**Perf notes while in there** (from review, currently minor): O(n^3) worst-case BPE on
long space-free segments (use a heap like SentencePiece), per-segment `strings.index`
scan over all added tokens (precompute or trie).

## I5. Op generality (do lazily, when a model needs it)

Known limitations, acceptable today, each a wall for some future architecture:
- Broadcasting is trailing-tile only (`_assert_broadcastable` in ops.odin).
- `permute` is 3-axis; `transpose` is F32-only (CUDA kernels exist only for f32).
- `select` is the only gather; no scatter.
- Conv is im2col + 2d pools only.
Recommendation: extend on demand with the case-registry discipline (add the op case →
gradient check + parity come free), rather than speculatively.

# Training track

## T1. Gradient checkpointing (capacity ceiling — the binding one)

**Problem.** The tape (`tape.odin`, `MAX_OPERATIONS` 16384) retains every activation;
training attention scratch is O(heads * T^2) (`backends/cpu/cpu.odin:_alloc_scratch`,
`attention_train.cu` with its 2048-token `d_p_row` cap). Training anything
transformer-sized in depth or context is memory-bound with no recourse. This is the
single binding constraint on training ambitions; everything else about training is
sound (verified gradients, parity-tested kernels, AdamW).

**Design sketch.** Segment-level recomputation on the existing flat tape:
- Add `ml.checkpoint_scope()` (or a per-op flag) marking a tape segment whose interior
  activations may be dropped after forward; record only the segment's boundary inputs.
- `backward` walks segments in reverse; for a dropped segment, re-run its forward
  (the ops are already recorded with their variants — re-dispatch `backend.forward`)
  into freshly allocated transient buffers, then run the segment's backward, then free.
- Requires: transient-buffer reuse across recomputation (CPU arena already free-alls
  per pass; needs a per-segment watermark instead), and determinism of forward replay
  (dropout is the one stochastic op — store its mask or seed per instance).
- Natural segment boundary for the networks: one transformer layer.

**Interactions.** The CUDA activation pool (`backends/activation_pool`) already does
size-matched replay reuse for inference; recomputation wants the same trick on the
training path. The 2048-token training-attention cap and the O(T^2) softmax scratch
(`cpu.odin` finding: allocated even at inference) are worth fixing in the same arc.

**Verify.** Gradient-check suite must stay bit-identical with checkpointing on/off for
a fixed seed; add a case registry dimension for "checkpointed" so parity covers it.

## T2. Optimizer/trainer breadth (cheap, on demand)

- SGD+momentum (RL baselines want it); trivial next to the existing AdamW plumbing in
  `optimizer.odin` + backend `update` procs.
- Multi-GPU / data parallel: out of scope until a concrete need; noted for honesty.

# Non-goals (current default: do not build)

## Batched / multi-sequence inference

**Current state.** Single sequence per context; parallel inference = one context per
thread over shared weights (documented threading contract, tested by
`parallel_inference_check`). No batched KV cache, no continuous batching.

**Decision.** The per-thread-context model is treated as sufficient for a personal-use
library, so this is a non-goal by default. Promote it to the roadmap only if serving
becomes an actual goal — and note that it is the largest single item here, reshaping
`attention_with_cache` (per-sequence cache positions), the cache layout (paged or
per-slot), and `generate`. Do not start it casually.

## Deferred by explicit decision (do not redo without new evidence)

- `Language_Model` vtable package — after convergence + `logits_mode=.Last`, the
  generic surface per backend is ~4 lines; revisit only when a third network or second
  consumer exists.
- safetensors unknown-dtype laxness — byte ranges are bounds-checked; rejecting
  unknown dtypes would refuse files with ignorable auxiliary tensors.
- `decode_tokens` counting the stop draw — intended semantics, asserted by
  `test_generate`.

# Standing verification bar for all of the above

`odin check` every touched package; `odin test tests` (plus `-define:ML_CPU_POISON=true`
and `-microarch:x86-64-v3`); `odin test tests/golden`; `odin test tests/parity` on the
GPU machine. New ops go through `tests/cases/` so gradient check and parity are
automatic. New tokenizer/quant formats get HF/llama.cpp-pinned golden fixtures.
