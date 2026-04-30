# TODO / future work

Notes accumulated during the PyTorch-parity / GPU-optimization session.
Ordered roughly by leverage, not by difficulty.

## North star: what this library should run

Goals: SOTA transformer LLMs (Gemma 4 class), game-playing agents,
audio processing (ASR, embeddings), and audio generation. The Gemma 4
target is the most demanding and is largely a superset of what the
others need on the architecture side, so the transformer stack is the
trunk and everything else branches off it.

Cross-cutting capabilities the current library is missing for these
goals (in addition to the dtype/perf items below):

- **GQA + RoPE as an op + sliding-window attention.** Required by
  every modern transformer, not just Gemma 4. RoPE must be a real op
  with config (theta, p-RoPE variant), not baked into `attention`.
- **KV-cache inference path.** `attention_decode` (single-token Q
  against cached K/V) plus a way to grow K/V buffers in place. Without
  this you can train but cannot "run" a model.
- **Conv1d / Conv2d ops.** Audio frontends, vision encoders, CNN
  trunks for AlphaZero-style agents. Examples currently sidestep this
  by flattening MNIST.
- **FFT / mel-spectrogram preprocessing.** CPU-only, lives outside the
  autograd graph. Gates all audio work.
- **Autoregressive generation loop.** Categorical/Gumbel sampling +
  KV-cache step. Needed for text, audio tokens, Decision Transformer
  action rollouts.
- **VQ / residual VQ with straight-through gradient.** Neural audio
  codecs (EnCodec/SoundStream) and discrete-latent world models.

Suggested phasing:
1. Dtype foundation (phase 1 below). DONE.
2. Modern attention surface for training: RMSNorm, RoPE-as-an-op, GQA,
   tied embeddings — all forward + backward. SwiGLU MLP comes free
   from existing primitives. DONE — `networks/llama/llama.odin` exists
   and `pytorch_parity llama_train` validates the full stack.
3. SmolLM2-135M forward-parity inference. DONE — safetensors loader
   plus `examples/smollm_inference/main.odin` matches HF logits within
   3e-4 on the 30-layer 135M checkpoint.
4. **Odin GPT-2 BPE tokenizer + sampling loop.** "Watch it generate"
   path. See "Next up" under the SmolLM2 section.
5. Stack-validation pretrain: ~30–50M params, ~1B tokens, overnight
   run. Sanity check that the training loop, optimizer, and data
   pipeline are correct end-to-end.
6. SmolLM2-135M fine-tune on a small corpus. Reuses the training loop
   from step 5; once forward parity is in (DONE), this is mostly data
   plumbing.
7. Sliding-window attention + KV-cache + `attention_decode`. Promotes
   the library from "can run a small generation demo" to "can run
   long-context inference at speed."
8. Gemma 4 E4B end-to-end: pull bf16 safetensors from HF, run
   inference, match logits. Forcing function for any remaining gaps
   (multimodal-tower-skipping in loader, SentencePiece tokenizer,
   etc.).
9. From-scratch pretrain at 160M+ scale. Match a published recipe
   (Pythia-160M or SmolLM2-135M). This is the "trainable
   intelligence" claim landing.
10. Conv1d/2d + FFT. Unlocks audio and vision frontends.
11. Branch into one of {audio generation, world-model RL, MoE
    training} — roughly equal effort off the same trunk.

## Current plan: SmolLM2-135M fine-tune path

Decided this session. The Gemma 4 E4B inference POC was reframed as a
later milestone after realizing it validates inference-only infra
(KV-cache, sampling, multimodal loader) without exercising the
training story, which is the actual project goal.

SmolLM2-135M is architecturally a miniature Gemma (RMSNorm, RoPE, GQA
9q/3kv, SwiGLU MLP, tied embeddings, GPT-2 BPE tokenizer). Every op
needed to load and fine-tune it is also needed for Gemma later. Fits
comfortably on a 3090 Ti (~5–7 GB train memory with FP32 master + bf16
compute + Adam).

Steps in order, each runnable:

1. **Op surface** (forward + backward, bf16 + f32, CPU + GPU):
   - RMSNorm — DONE. New `Rmsnorm` variant in `ml.odin` (`weight`,
     `rstd` scratch — no mean), proc `ml.rmsnorm(input, weight)`.
     CPU `rmsnorm_forward`/`rmsnorm_backward` split into `_f32` and
     `_bf16` paths mirroring `layernorm`. GPU has four shaders
     (`rmsnorm_stats`, `rmsnorm`, `rmsnorm_back_input`,
     `rmsnorm_back_weight`) plus bf16 variants; bf16 forward/backward
     require even `size`. Coverage: dtype_roundtrip bf16 case
     (CPU + GPU), pytorch_parity `rmsnorm` and `rmsnorm_big` (M=64,
     N=128) on both backends.
   - RoPE-as-an-op with `theta` config — DONE. Already promoted to
     `ml.rope(input, head_count, base = 10000) -> output`. Uses
     `cos_cache` / `sin_cache` scratch tensors. The transformer block
     calls it directly (no inlined RoPE).
   - GQA in `attention` — DONE. Reshaped the API: `ml.attention(query,
     key, value, n_q_heads, n_kv_heads = n_q_heads, causal = true)`.
     The op variant carries `key`/`value` plus the existing scratch
     tensors; `Operation.input` holds query (mirrors the `Add{b}` /
     `Linear{weight}` pattern). Shapes:
     `query: [T, n_q_heads * D]`, `key/value: [T, n_kv_heads * D]`.
     Inside kernels, `kv_h = h * n_kv_heads / n_q_heads` — K/V are
     never expanded into a buffer. CPU forward parallelizes over
     q-heads; backward parallelizes over kv-heads (with the inner
     loop covering the q-head group) so dK/dV writes are race-free.
     GPU back_kv shader is one workgroup per `(kv_head, key)`,
     iterating the q-head group internally; back_q / back_d /
     forward stay one workgroup per `(q_head, query)`. Coverage:
     dtype_roundtrip Bf16 GQA case (4 q-heads × 2 kv-heads),
     pytorch_parity `attention_gqa` and `attention_gqa_big`
     (SmolLM2-shaped: 9 q-heads × 3 kv-heads, T=64) on CPU + GPU.
     The disabled `attention_bf16_coopmat.comp` was minimally
     updated to take Q/K/V separately and asserts `n_kv == n_q` at
     dispatch — full GQA support deferred until the coopmat
     attention path becomes profitable.
   - Tied embeddings — DONE. No new op was needed: re-using one
     `Tensor` across `ml.select` (embed lookup) and `ml.linear`
     (lm_head) just works because both backward paths accumulate
     into `gradient(weight)` with `+=`. `networks/transformer/`
     gained a `tied_embeddings` flag (default true): when true,
     `transformer.output_weight` aliases `transformer.token_embeddings`
     and is skipped in `make`/`destroy`/`copy`/`randomize`/`update`
     so the shared tensor isn't double-allocated or double-stepped.
     Coverage: pytorch_parity `tied_embeddings` test
     (vocab=32, embed=8, 12 tokens) checks loss + grad_w against a
     PyTorch reference where one `nn.Parameter` is reused for embed
     and lm_head.
   - PyTorch parity tests for each, plus a `_big` variant for
     attention to exercise the GQA path at realistic shapes — DONE
     (`rmsnorm` + `rmsnorm_big`, `attention_gqa` + `attention_gqa_big`,
     `tied_embeddings`, plus the new `llama_train` end-to-end test).

   **Network module — DONE.** `networks/llama/llama.odin` composes the
   above ops into a Llama/SmolLM2-shaped model: `select` → per-layer
   (RMSNorm → q/k/v_proj → RoPE on Q/K → GQA attention → o_proj →
   residual → RMSNorm → SwiGLU MLP → residual) → final RMSNorm → tied
   lm_head. Weight field names match HuggingFace safetensors so the
   loader work (step 3) is mechanical. Provides `SMOLLM2_135M_CONFIG`
   (30L / 576E / 9q×3kv / D=64 / FFN=1536 / vocab=49152 /
   rope_base=100000). Coverage: `pytorch_parity llama_train` (2 layers,
   embed=32, 4q×2kv, 20 Adam steps, F32, loss curve matches PyTorch
   reference within 1e-3 abs / 5e-3 rel) plus a `tests/smollm_smoke/`
   that allocates the full 30-layer config and runs one forward
   (71 ms on CPU at 8 tokens) to confirm dimensions check out
   end-to-end.

2. **Stack-validation pretrain** (overnight, ~30–50M params):
   - Streaming dataset loader (FineWeb-Edu sample or similar).
   - GPT-2 BPE tokenizer (matches SmolLM2 — saves work later).
   - Training loop with gradient accumulation, checkpointing,
     periodic eval. Live in `examples/pretrain_small/`.
   - Mixed-precision recipe: FP32 master weights, bf16 compute,
     FP32 Adam state. Subsumes the deferred mixed-precision
     example below.
   - Acceptance: loss drops from ~10 nats to ~4 nats on FineWeb-Edu
     within a few hours. Curve shape matches published references
     for similarly-sized models.

3. **Safetensors loader** — DONE.
   - Generic `loaders/safetensors/safetensors.odin`: read file,
     parse the v0.3 JSON header, expose `get_info(name)` /
     `get_bytes(name)` lookups. JSON is parsed with
     `core:encoding/json` (parse_integers=true).
   - SmolLM2/Llama mapping in `networks/llama/loader.odin`:
     `llama.load_safetensors(model, path)` walks the HF tensor
     names (`model.embed_tokens.weight`, `model.layers.{i}.*`,
     `model.norm.weight`, optional `lm_head.weight`) and uploads
     them. Handles both F32 and BF16 storage by converting to F32
     master weights. **Q/K projection rows are permuted at load
     time** to convert HF's `rotate_half` RoPE layout
     (`[first_half, second_half]` per head) into our `ml.rope`
     layout (interleaved pairs). V/o_proj need no permutation.
   - Currently reads the whole file into RAM. mmap optimization
     deferred until 8B-class checkpoints make it matter.

4. **Forward parity check** — DONE.
   - `examples/smollm_inference/main.odin` allocates
     `llama.SMOLLM2_135M_CONFIG`, loads HF weights, runs forward
     on a fixed prompt, compares to logits dumped by
     `tools/smollm_dump.py` (which downloads the HF checkpoint
     and runs `transformers` as the reference).
   - Result on prompt *"The capital of France is"*: max abs logit
     diff **3e-4**, top-5 predictions match HF identically at
     every position. " Paris" sits at position 2 in the next-token
     top-5, same rank as HF.
   - Forward at 5 tokens runs in ~50 ms on CPU; weight load takes
     ~200 ms. Single-threaded would be slower; this used 8 CPU
     threads.

5. **Fine-tune**:
   - Pick a small instruction or domain corpus.
   - Reuse the training loop from step 2.
   - A few thousand steps. Acceptance: loss drops cleanly,
     greedy-decode samples reflect the fine-tune corpus.

### Next up: Odin tokenizer + sampling loop ("watch it generate")

Forward parity is in place; the missing piece between that and "type a
prompt and see SmolLM2 talk" is text I/O. User decision: the tokenizer
will be implemented entirely in Odin (no Python tokenizer wrapper).

Components, in implementation order:

1. **GPT-2 byte-level BPE tokenizer** in `tokenizers/gpt2/` (or
   similar). The HF `tokenizer.json` for SmolLM2-135M is already saved
   at `smollm_data/tokenizer.json`. Structure:
   - `model.type = "BPE"`, vocab size 49152, 48900 merges.
   - Pre-tokenizer is a `Sequence` of `Digits{individual_digits=true}`
     then `ByteLevel{add_prefix_space=false, use_regex=true}`.
   - ByteLevel decoder (inverse byte-to-unicode mapping).
   - 1 entry in `added_tokens` (the `<|endoftext|>`-equivalent;
     verify exact name when implementing).
   Implementation pieces:
   - Parse `tokenizer.json` with `core:encoding/json`.
   - Build the standard 256-byte → Unicode-printable map and its
     inverse.
   - Implement the GPT-2 pre-tokenizer regex
     `' ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`. Check
     `core:text/regex` for Unicode property class support
     (`\p{L}`, `\p{N}`); if missing, hand-roll a small character-
     class state machine — that's what gpt2.c, llama.cpp, and
     tokenizers-rs all do.
   - BPE merge step: per pre-token word, repeatedly merge the
     highest-priority adjacent pair (using a `map[Pair]int` rank
     lookup built from the merges list). Standard algorithm —
     ~50 lines.
   - Decoder: ID → BPE-token strings → concatenate → byte-level
     decode → UTF-8.
   - Acceptance: encode/decode parity against HF's tokenizer for
     ~50 fixed prompts (a small parity script + a binary that
     prints token IDs would do it).

2. **Sampling loop.** Lives in `examples/smollm_chat/main.odin` (or
   inside the existing `smollm_inference` example). Naive: re-runs
   the full forward each generated token. KV-cache stays deferred —
   short generations (< 100 tokens) at the SmolLM2-135M scale should
   be tolerable on CPU (~50 ms × N²/2 / batch). Modes:
   - Greedy: argmax of last-row logits.
   - Top-k with temperature: top-k indices, softmax with `T`,
     sample via cumulative + `rand_float32`.
   Optional: top-p (nucleus). Defer if greedy + top-k cover the
   demo case.

3. **Chat example.** `examples/smollm_chat/main.odin`: CLI takes a
   prompt, tokenizes, runs N sampling steps, decodes incrementally,
   prints tokens as they arrive. Matches HF's greedy continuation
   for the first ~20 tokens of *"The capital of France is"* as the
   acceptance check.

Once this is in, we also have the BPE tokenizer needed for step 2
(stack-validation pretrain) and step 4 (SmolLM2 fine-tune corpus
encoding).

Deferred until after the above lands:
- Sliding-window attention, KV-cache, `attention_decode`. Needed
  for fast generation at long context; not needed for short demo
  generations or for fine-tune validation.
- SentencePiece tokenizer. Gemma uses it; SmolLM2 uses GPT-2 BPE,
  which is simpler.
- 8B-scale checkpoint loading + multimodal-tower-skipping. Gemma
  4-specific.
- Attention coopmat performance tuning (currently disabled). Not
  blocking 135M training; revisit when something benches slow.
- `select` bf16 path. SmolLM2 fine-tune in mixed precision keeps
  `token_embeddings` as F32 master and casts the lookup output to
  Bf16; a bf16 select would shave one cast per step but is not
  blocking.

## Multi-dtype foundation (precondition for serious LLM work)

The project goal is to infer and train Gemma-class LLMs. That requires BF16
(or at minimum FP16) tensors end-to-end: BF16 weights and activations,
FP32 master weights for the optimizer, and tensor-core matmul on top.
Without dtype support the project caps at "small models in FP32," and the
coopmat shader work below would need to be redone once buffers gain types.

### Status (as of last session)

**Phase 1 (foundations) — DONE.** Done across `ml.odin`,
`backends/cpu/cpu.odin`, `backends/gpu/{backend,buffer,gpu,ops}.odin`,
plus `tests/dtype_roundtrip/`.

- `Data_Type` extended to `{F32, F16, Bf16}`. Note: `Bf16` is
  `distinct u16`, with `bf16_from_f32` (round-to-nearest-even,
  NaN-preserving) and `bf16_to_f32` helpers in `ml.odin`. Style choice:
  the type is `Bf16` (not `BF16`); enum cases stay capitalized as
  `.F32`/`.F16`/`.Bf16`.
- Backend buffer interface is now byte-oriented:
  `buffer_alloc(byte_count: int, persist, loc)`,
  `buffer_get/set(buffer, []byte, loc)`. The CPU helpers
  (`data`/`gradient`/`adam_m`/`adam_v`) now slice via `[^]f32` cast +
  `t.count` since the slice header carries byte count.
- `alloc`/`zeros`/`make` carry `type: Data_Type = .F32`. `zeros_like`
  and `_zeros_*` helpers preserve source type. `alloc` rounds byte
  count up to a 4-byte multiple — required because the GPU bf16/f16
  shaders pack two halves per `uint`, so an odd element count must be
  writable through the trailing uint.
- Public typed APIs (`set_data`/`get_data`/`get_gradient`,
  `upload_tensor`/`download_tensor` in the GPU backend) keep `[]f32`
  signatures with an `assert(t.type == .F32)`. Raw byte access:
  `set_data_bytes`/`get_data_bytes`. `fill_normal`/`fill_value` switch
  on `t.type` and emit F32/F16/Bf16.
- `backward()` currently asserts the loss tensor is F32.

**Phase 2 (`cast_to` op) — DONE.** `cast` is reserved in Odin so the
proc is `ml.cast_to(input, target_type)`. The op variant is `Cast{}`
(empty marker — src and dst types are read off `op.input.type` /
`op.output.type`).

- CPU: full support for all `{F32, F16, Bf16}` pairs, forward and
  backward. Implementations: `cast_forward` / `cast_backward` in
  `cpu.odin`, with shared `_cast_bytes` and `_cast_bytes_accumulate`
  helpers. Backward accumulates (`+=`) into `input.gradient`.
- GPU: F32 ↔ Bf16 only. Four shaders under
  `backends/gpu/shaders/cast/*.comp` (with built `.spv`):
  `cast_f32_to_bf16`, `cast_bf16_to_f32`, plus `_back` variants used
  by `cast_backward`. Same-type cast forward falls through to `_copy`
  (a vk buffer copy); same-type backward currently panics (intentional
  — flag if it ever shows up). The shaders pack two bf16/uint and
  dispatch one thread per pair; no special Vulkan extensions needed.
- F16 GPU support was **deliberately deferred** — Gemma 4 is bf16, so
  this can wait until something asks for it.
- Tests: `tests/dtype_roundtrip/main.odin` covers F32/Bf16/F16 byte
  round-trip and `cast_to` forward + backward on both backends. All
  pass. `tests/gpu_unified_check` (the existing F32 op suite) still
  passes — F32 path is untouched.

**Phase 2.5 (`add` dtype-generic on Bf16) — DONE.** `ml.add` now asserts
`a.type == b.type`. CPU `add_forward`/`add_backward` switch on dtype with
f32-accumulator paths for Bf16 and F16 (f32 instead of f16 for the bf16
path because casting bf16 → f16 loses range, and CPUs widen f16 to f32
for arithmetic anyway). GPU has three new packed-bf16 shaders
(`add_bf16`, `add_back_a_bf16`, `add_back_b_bf16`) that thread per output
pair so each thread writes a whole packed uint and avoids partial-uint
races. Test added in `tests/dtype_roundtrip` (broadcast forward + dx/db
backward) covering both backends.

**Phase 3a (naive Bf16 `linear` forward) — DONE.** Took the contained
"BF16 matmul plumbing" half of phase 3 first, before the coopmat work.
`ml.linear` now asserts `input.type == weight.type`. CPU `linear_forward`
splits into `_f32` and `_bf16` paths (bf16 dot product accumulates in
f32, stores result via `bf16_from_f32`); `linear_backward` panics on any
non-F32 type. GPU has a new `linear_bf16.comp` mirroring `linear.comp`'s
tile structure but with packed-uint loads and bf16 output writes — each
thread owns a TM=4 × TN=4 sub-tile, writes 2 whole packed uints per row,
which requires `input_size` and `output_size` to be even (asserted at
dispatch). Tests added covering both a small (M=4, K=8, N=6) and a
multi-tile (M=64, K=64, N=128) shape with values chosen so the bf16
result is bit-exact against the f32 reference. `linear_backward` panics
on bf16 in the GPU backend too — that ships with phase 3b.

**Phase 3b (Bf16 `linear` forward via cooperative matrix) — DONE.** RTX
30+ exposes `VK_KHR_cooperative_matrix` + `VK_KHR_shader_bfloat16` with
`shaderBFloat16CooperativeMatrix = true` and a 16/16/16 bf16/bf16/fp32
config; we now query that at device init, set `_gpu.coopmat_bf16`, and
enable `cooperativeMatrix`, `shaderBFloat16Type`,
`shaderBFloat16CooperativeMatrix`, and `storageBuffer16BitAccess` in the
device feature chain.

The vendored `vendor:vulkan` binding doesn't yet expose the bf16
extension, so `gpu.odin` carries hand-written constants
(`KHR_SHADER_BFLOAT16_EXTENSION_NAME`, `COMPONENT_TYPE_BFLOAT16_KHR =
1000141000`, `Physical_Device_Shader_Bfloat16_Features_KHR`) until the
binding catches up.

`linear_bf16_coopmat.comp` is intentionally minimal: one subgroup per
workgroup, one 16×16 output tile per subgroup, K loop calls
`coopMatMulAdd` directly — no shared-memory staging or per-warp tiling
yet. Buffers are typed `bfloat16_t`; W is loaded ColumnMajor to take its
[N, K] storage as B's transposed [K, N]. Backend dispatches coopmat when
`_gpu.coopmat_bf16` is true and all three dims are multiples of 16; else
falls back to the naive `linear_bf16.comp`. The dtype_roundtrip
multi-tile test (M=64, K=64, N=128) exercises the coopmat path; the
small test (M=4, K=8, N=6) exercises the fallback.

**Phase 3c (Bf16 `linear_backward`) — DONE.** CPU `linear_backward`
splits into f32 and bf16 paths (bf16 dot product accumulates in f32 like
the forward). GPU has new `linear_back_input_bf16.comp` and
`linear_back_weight_bf16.comp` — naive thread-per-(c, k_pair) and
thread-per-(o, k_pair) respectively, each thread writing a whole packed
uint to dodge partial-uint races. K (input_size) must be even (asserted
at dispatch). Coopmat backward is deferred — the existing F32 backward
shaders are also naive, so this matches the project's "tiled GEMM is
overkill until backward becomes a profile blip" rule.

**Phase 3d (Bf16 `batched_matmul` fwd + bwd) — DONE.** Same shape as
linear: `ml.batched_matmul` asserts `a.type == b.type`. CPU has bf16
`*_forward_bf16` and `*_backward_bf16`. GPU has three new bf16 shaders
(`batched_matmul_bf16`, `batched_matmul_back_input_bf16`,
`batched_matmul_back_weight_bf16`), all naive thread-per-output-pair
with N (and K for backward) required even at dispatch. Coopmat for
batched_matmul is deferred to the perf pass alongside coopmat for
linear backward and attention.

## Remaining bf16 work for an end-to-end transformer step

Ordered by what unlocks the most when done:

1. **Bf16 `attention` (fused fwd + 3 bwd shaders) — DONE.** `ml.attention`
   now allows F32 or Bf16 input; output type matches input, scratch
   (`softmax_outputs`, `lse`, `d_p_scratch`, `d_acc`) stays F32. CPU has
   `attention_forward_bf16` / `attention_backward_bf16` mirroring the F32
   path with f32 inner compute and bf16 storage. GPU has four new shaders
   (`attention_bf16`, `attention_back_d_bf16`, `attention_back_kv_bf16`,
   `attention_back_q_bf16`) that load bf16 via packed-uint shift-extract,
   run the same online-softmax FA2 algorithm in f32, and pair-pack writes
   back to bf16. Backward kernels stage per-thread dQ/dK/dV in shared
   memory and have the first D/2 threads do the bf16 RMW so adjacent d's
   share a packed uint without races. Even `head_size` is required and
   asserted at dispatch. Tests in `tests/dtype_roundtrip` cover bf16
   attention forward + backward against an F32 reference on both backends
   (5e-2 tolerance, since softmax/exp prevent bit-exact compare).

2. **Bf16 element-wise/normalization ops.** Mechanical rollout following
   the `add` pattern (thread-per-output-pair, f32 inner compute, packed
   bf16 storage):
   - `mul`, `sub`, `div` — DONE (CPU + GPU, parity tests).
   - `gelu`, `silu`, `relu`, `sigmoid`, `tanh`, `exp` — DONE (CPU + GPU
     via `_unary_forward_gpu` / `_unary_backward_gpu` helper, parity tests).
   - `layernorm` — DONE. Stats and forward shaders use packed-bf16 I/O
     with f32 mean/rstd scratch; backward input/weight use the same
     pair-aligned RMW pattern. Even `size` required.
   - `softmax`, `log_softmax`, `entropy` — DONE. F32 reductions
     internally; even `size` required for pair-aligned writes. `entropy`
     uses one workgroup per output pair to handle the `output[count]`
     packed write without races.
   - `rope`, `causal_mask`, `permute`, `concat`, `slice`, `mean` —
     DONE (CPU + GPU). Pure-copy ops (`slice`, `concat3`, `permute`,
     `causal_mask`) use one-thread-per-pair packed-uint reads/writes
     so the two halves of one packed bf16 slot are owned by exactly
     one thread (no race on RMW). `rope` rotates whole bf16 pairs:
     `head_size` is even so `(i_lo, i_hi)` always lands on an aligned
     packed slot, one thread does both rotations and a single packed
     write. `mean` forward uses one workgroup per row plus an atomic
     CompSwap to RMW the half of the output pair belonging to that
     row; backward is one thread per input pair.
     `ml.slice` and `ml.permute` now allocate output as `input.type`
     instead of hardcoded F32.

3. **First bf16 transformer parity test — DONE.** New
   `transformer_train_bf16` test in `tests/pytorch_parity` mirrors the
   library's transformer block in PyTorch with bf16 weights/activations
   (FP32 master → bf16 cast inside `forward`, FP32 logits at the end
   so `backward()` gets an F32 loss). Loss curves match within ~3e-4
   absolute over 10 Adam steps on both CPU and GPU at the
   `mlp_train`-style tolerance.

   Gap surfaced and fixed: `slice_trailing` had no bf16 path on either
   backend — CPU `cpu.odin` used the F32-only `data()` / `gradient()`
   helpers, and the GPU shaders typed the buffer as `float[]`. Both
   read and wrote 4 bytes per "bf16 element," producing silent memory
   corruption that the dtype_roundtrip suite missed (it covers `slice`
   but not `slice_trailing`). Added bf16 paths in `cpu.odin`,
   `slice_trailing_bf16.comp`, and `slice_trailing_back_bf16.comp`
   (one-thread-per-pair packed-uint reads with RMW on the backward
   scatter, same pattern as the existing bf16 pure-copy ops).

4. **Coopmat perf pass.** Shared layout across every shader: BM=64,
   BN=64, BK=16, 4 subgroups arranged 2×2 per WG, each subgroup owning
   a 2×2 grid of 16×16 output coopmat tiles, inputs staged through
   shared memory. Subgroup size = 32 hardcoded. Build script now
   passes `--target-env=vulkan1.3` (needed for `GroupNonUniform`).
   - **Linear forward, linear backward (input + weight),
     batched_matmul forward + backward — DONE.** Eligibility for each
     shader requires the relevant output dims to be multiples of 64
     and the contraction dim a multiple of 16; smaller/odd shapes
     fall back to the existing naive packed-uint shaders. Coopmat
     backward shaders load the existing output tile as a bf16 acc
     coopmat, convert to fp32, accumulate via `coopMatMulAdd`, and
     convert back to bf16 on store, so they preserve the `+=`
     semantics that the naive backward used.

     RTX 3090 Ti microbench (BF16 peak ~150 TFLOPS):
     - linear forward:
       - 512×768×768:    2.34 → 3.58 TFLOPS  (1.5x)
       - 512×2048×2048:  5.88 → 15.62 TFLOPS (2.7x)
       - 2048×2048×2048: 7.99 → 27.89 TFLOPS (3.5x)
     - linear backward (sum of dx + dw GEMMs):
       - 512×768×768:    1.27 → 4.91 TFLOPS  (3.9x)
       - 512×2048×2048:  1.53 → 14.67 TFLOPS (9.6x)
       - 2048×2048×2048: 1.81 → 17.33 TFLOPS (9.6x)
     The backward win is largest because the old naive bf16 backward
     was bottlenecked on packed-uint RMW serialization, not compute.

     End-to-end transformer bench (`tests/gpu_transformer_bench`,
     L=12 H=8 E=512 T=256): F32 full step 76.1 ms → Bf16 full step
     30.9 ms (**2.46x**). Forward alone: 1.49x. The full-step
     speedup is dominated by backward.

     Tried-and-rejected variants for linear forward: BM=BN=128
     (faster on huge but slower on medium/big due to
     under-occupancy); BK=32 (no measurable win, slightly worse on
     big shapes).

     Coverage: `dtype_roundtrip` multi-tile linear test (M=64, N=128,
     K=64) exercises the linear coopmat path; new
     `batched_matmul (coopmat-tile)` test (BATCH=2 M=K=N=64)
     exercises both the bmm forward and backward coopmat paths.

   - **Attention forward — implemented but disabled.**
     `attention_bf16_coopmat.comp` is a full FA2-with-Q-tiling
     rewrite: BR=16 queries per workgroup, BC=64 keys per K-tile, 1
     subgroup of 32 threads, MAX_D=64 capping shared-memory at
     ~32 KB. Q@K^T and P@V both run on tensor cores via
     `coopMatMulAdd`; the online softmax stages scores through fp32
     shared memory, runs SIMT per-row max/sum via `subgroupMax`/
     `subgroupAdd` (32 threads × 2 cols each = BC=64), and writes
     bf16 P back to shared for the P@V matmul. Output is folded
     into a fp32 running accumulator with per-row alpha rescaling,
     then normalized and pair-packed to bf16 at the end. Correctness
     verified by a new `dtype_roundtrip` case at T=32, H=2, D=16
     (the smallest shape that hits the coopmat path).

     **Disabled in dispatch.** On the bench shapes (D=64, T=64–256)
     this runs ~18% *slower* than the existing SIMT bf16 shader.
     Root causes: the per-row softmax loop is sequential across
     16 rows (one `subgroupMax` + `subgroupAdd` per row), each K-
     tile costs 4 barriers (Q/K/V stage + score store + softmax +
     output fold), and shared-memory budget (~32 KB at MAX_D=64,
     ~58 KB at MAX_D=128) limits occupancy. The SIMT shader hides
     the same compute under simpler control flow with much less
     shared-memory pressure, so the tensor-core throughput
     advantage doesn't materialize at these sizes. Left as a
     foundation for the backward pass and for future tuning
     (multi-subgroup BR=32, fewer barriers via more-aggressive
     overlap, specialization-constant D, larger seq-len bench
     shapes where the coopmat advantage will dominate).

   - **Attention backward — not started.** The three backward
     shaders (`attention_back_d`, `attention_back_kv`,
     `attention_back_q`) are mostly per-token reductions plus
     `softmax_outputs[H, T, T]` recompute, not pure GEMMs, so a
     coopmat port is a bigger restructure than the linear/bmm
     ports. Best done together with the forward redesign once the
     forward profile is unblocked.

5. **Mixed-precision recipe in `examples/`.** End-to-end demo of the
   standard "FP32 master weights, BF16 forward/backward, FP32 optimizer
   state" pattern on a small transformer. Deliverable for "this library
   is a real bf16 stack."

### Open / next phases

The bf16 trunk is functionally complete for training. Next work is
driven by the SmolLM2-135M plan above, not by more bf16 perf. Items
left from this section that were deferred rather than retired:

- **Make attention coopmat profitable.** Forward shader exists
  (`attention_bf16_coopmat.comp`) but currently slower than SIMT.
  Most promising next moves: parallelize the per-row softmax across
  threads (use a (row, col) thread layout instead of sequential
  rows), reduce barrier count by fusing Q-load with the first K
  iteration, multi-subgroup BR=32 to amortize K/V staging across
  more queries, specialization constants for D so shared-memory
  isn't sized for the worst case. Once the forward wins, port the
  same Q-tile layout to backward. Revisit once 135M training
  benches show attention as a bottleneck — at seq=1024 the coopmat
  path may already win without further tuning.

- **Mixed-precision recipe in `examples/`.** Subsumed by step 2 of
  the SmolLM2 plan: the stack-validation pretrain run is the
  end-to-end mixed-precision recipe.

**Design decisions to lock in at phase 1:**

- **Op dtype mixing.** When `linear(x: F32, w: BF16)` is called: error,
  not auto-cast. Forces explicit `cast` ops, keeps optimization tractable.
- **Output dtype.** Compute ops (`linear`, `add`, etc.) preserve input
  dtype. Internal accumulators in shaders are FP32 regardless (this is
  what tensor cores do anyway).
- **Optimizer state dtype.** Adam M/V stay FP32 even when the parameter
  is BF16. Hardcode in the optimizer; don't try to make `Buffer_Kind`
  per-buffer-typed.
- **Mixed precision (master weights).** Don't build this into the core
  Tensor — handle it at the example level by keeping two tensors
  (FP32 master, BF16 compute view) and casting each step. Simpler
  primitives, same end result.

## GPU performance

- **Flash Attention v2 — Q-tiled forward and tiled backward.** Forward now
  uses online softmax with K streamed in tiles of `BC=64`, removing the
  `MAX_T=4096` cap and the `scores[MAX_T]` shared array, but still runs one
  workgroup per `(head, query)` (BR=1). Real FA2 also tiles Q (BR>1) so K and
  V are reused across the whole BR×BC score block. Backward is still 3
  kernels (D-precompute, dKV, dQ) with no Q/K tiling. Tiling both should
  pull `attention_causal` from ~3x off cuDNN toward ~1.5x. Reference: ggml's
  `flash_attn.comp` and the Tri Dao FA2 paper.

- **Tensor-core matmul via VK_KHR_cooperative_matrix.** The current `linear`
  shader is FP32 SIMT. RTX 30+ tensor cores (BF16, FP16) deliver 4-8x more
  FLOPs and `cooperative_matrix` is the Vulkan path to them. Closes the
  remaining `linear_fwd` gap (~5x off cuBLAS). **Blocked on the multi-dtype
  foundation above** — coopmat needs BF16 buffers to be the natural input
  type; doing it on FP32 buffers means writing the conversion shim twice.
  Reference: ggml's `mul_mm_cm2.comp`.

- **Larger-shape GPU bench coverage.** Current speed bench uses small shapes
  where Vulkan's per-dispatch overhead dominates. Add benches at training-
  realistic shapes (e.g. transformer step at seq=512, embed=768) to see
  steady-state kernel performance, not launch overhead.

- **Cache descriptor sets in `_dispatch`.** Every `_dispatch` call does a fresh
  `vkAllocateDescriptorSets`, which is the per-shader floor (~50-100 us based
  on the small-op timings). Caching by `(pipeline, buffers)` would shave that
  for repeated identical dispatches. Broad benefit, no single op moves
  dramatically.

- **GPU-side fill kernels.** `ml.fill_value` and `ml.fill_normal` currently
  build the buffer on the CPU and do a synchronous upload, which forces a
  `vkQueueWaitIdle` mid-frame. A `fill_constant.comp` and a Philox-style
  `fill_normal.comp` would keep initialization on-device.

## GPU robustness

- **Hardcoded `head_size <= 256` in attention backward shaders.** The shared
  memory caches (`k_shared`, `v_shared`, `q_shared`, `do_shared`) are fixed
  `float[256]`. Asserted at dispatch time but should either grow dynamically
  via specialization constants or auto-fall-back to a tiled variant.

- **`head_size > WG=64` in attention backward kernels.** Per-thread d_K/d_V
  accumulators only handle one `d` element each. Heads larger than 64 would
  drop the rest. Either fix the loop or use specialization constants.

- **Vulkan specialization constants for shader tile sizes.** This session's
  `LINEAR_LOCAL_X` / `TILE_M` mismatch bug came from having a tile size in two
  places (the shader and `ops.odin`). Specialization constants make the shader
  the single source of truth. ggml does this throughout. Bigger refactor but
  removes a whole bug class.

## CPU performance

- **Real GEMM (or link BLAS) for `linear` and `batched_matmul`.** CPU
  `linear_fwd` is 6-8x off MKL/OpenBLAS at the bench shape. Fastest path is
  linking OpenBLAS for `cblas_sgemm`; afternoon of work, becomes as fast as
  PyTorch on matmul, costs a vendor dependency. Hand-rolled tiled GEMM is
  weeks-to-months for a competitive one.

- **Switch CPU attention from `softmax_outputs[H, T, T]` to LSE-only.** Current
  CPU `attention` saves the full softmax matrix for backward (~2 MB per call
  at the bench shape). Flash Attention v2 saves only `lse[H, T]` (~8 KB) and
  recomputes attention weights in backward. Same algorithm we use on GPU
  already; would unify the two scratch layouts and matter at long sequence
  lengths.

## Test coverage

- **End-to-end transformer parity tests.** DONE — `pytorch_parity` now
  has `transformer_train_bf16` (GPT-2-style block, FP32-master + bf16
  compute) and `llama_train` (RMSNorm + RoPE + GQA + SwiGLU + tied
  lm_head, F32). Both compare loss curves to PyTorch references, and
  the SmolLM2 forward-parity check at `examples/smollm_inference/`
  validates the full 30-layer 135M model against HF.

- **Larger-shape variants for every parity test.** Add a `_big` variant for any
  op with internal tile sizes. The new `linear_big` (M=64) is the template;
  worth adding for `attention`, `batched_matmul`, `softmax`, `layernorm` so
  any future tile-size tuning trips the suite immediately.

- **GPU stress test for races on `linear_backward` style accumulation.** The
  CPU `linear_backward` two-pass fix solved the race; the GPU version should
  be audited the same way and have a stress test that runs the same input
  many times and asserts bit-identity.

## Style / housekeeping

- The `attention_fused` vtable flag was removed once both backends had a fused
  Attention. If a third backend is added that wants the old composition path,
  resurrect `_attention_compose` (it's in git history) and the flag.

- `ml.fill_normal` / `ml.fill_value` use Odin's RNG (`core:math/rand`) seeded
  per process. PyTorch parity tests sidestep this by uploading PyTorch-
  generated weights. If you ever want bit-identical RNG between CPU and GPU
  paths, you'll need a deterministic in-library RNG.

- Bench `_checksum` downloads 1 float as a sync token. Cute, but a
  `ml.sync()` proc that just submits + waits without a download would express
  intent more clearly.
