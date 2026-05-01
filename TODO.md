# TODO / future work

Ordered roughly by leverage, not by difficulty.

## North star: what this library should run

Goals: SOTA transformer LLMs (Gemma 4 class), game-playing agents,
audio processing (ASR, embeddings), and audio generation. The Gemma 4
target is the most demanding and is largely a superset of what the
others need on the architecture side, so the transformer stack is the
trunk and everything else branches off it.

Cross-cutting capabilities still missing:

- **Conv1d / Conv2d ops.** Audio frontends, vision encoders, CNN
  trunks for AlphaZero-style agents. Examples currently sidestep this
  by flattening MNIST.
- **FFT / mel-spectrogram preprocessing.** CPU-only, lives outside the
  autograd graph. Gates all audio work.
- **VQ / residual VQ with straight-through gradient.** Neural audio
  codecs (EnCodec/SoundStream) and discrete-latent world models.
- **SentencePiece tokenizer.** Gemma uses it; SmolLM2 uses GPT-2 BPE
  (already shipped).

Current status of the trunk:

- Dtype foundation (F32 / Bf16 end-to-end, coopmat tensor-core matmul)
  — done.
- Modern attention surface for training (RMSNorm, RoPE-as-an-op, GQA,
  tied embeddings, SwiGLU MLP) — done.
- SmolLM2-135M forward parity inference (max abs logit diff 3e-4 vs HF)
  — done.
- Odin GPT-2 BPE tokenizer + sampling loop (`examples/smollm_chat`)
  — done.
- KV-cache + `attention_with_cache` (CPU + GPU) — done.
- Sliding-window attention (forward + backward + cache, CPU + GPU)
  — done.
- Gemma 4 tokenizer (`tokenizer.json` byte-fallback BPE, 52-prompt HF
  parity in `tests/gemma_tokenizer/`) — done.
- `ml.rope` `rope_fraction` parameter (CPU + GPU forward + backward)
  — done.
- `ml.rmsnorm` `unit_offset` flag and per-op `eps` parameter — done.
- Gemma 4 E4B forward parity inference (max abs logit diff 2.99e-4
  vs HF, mean 3.78e-5) — done.
  - `networks/gemma/{gemma.odin,loader.odin}` ship the full text
    stack: scaled embeddings, PLE pathway, KV sharing on layers
    24..41, alternating sliding/full attention with per-type
    `head_dim` (256 / 512), `partial_rotary_factor=0.25` on full
    layers, per-layer `layer_scalar`, `final_logit_softcapping=30`.
  - `examples/gemma_inference/` runs forward and verifies against
    `gemma_data/expected_logits.bin`.
- Gemma 4 E4B chat / decode at GPU bf16 (~6 tok/s on a 3090 Ti, 16 GB
  VRAM resident) — done. `examples/gemma_chat/` does prefill +
  per-token decode through KV cache, greedy + top-k/temperature
  sampling, EOS stop. Several pieces of supporting infrastructure
  landed in this milestone:
  - **GPU persistent-buffer sub-allocator** (`backends/gpu/buffer.odin`,
    `Pool_Block` in `gpu.odin`). NVIDIA's Windows Vulkan driver
    reserves a multi-MB minimum page per `vkAllocateMemory`, so the
    naive one-allocation-per-tensor path OOM'd at ~6 GB used despite
    20 GB free. Sub-allocating from 256 MB pool blocks (or larger
    when a single buffer exceeds that) drops the alloc count from
    ~500 to ~40 and unblocked bf16 inference.
  - **`gemma.make` `for_training` flag**. `ml.make` defaults to
    allocating `{Data, Gradient, Adam_M, Adam_V}` — 4× memory.
    `gemma.make` now defaults to `{Data}` only and only allocates the
    full 4-buffer set when `for_training=true`. This was the actual
    cause of the bf16 OOM (36 GB targeted instead of 9 GB).
  - **`embed_tokens_per_layer` host-side**. The per-layer-input
    embedding table is `[vocab × num_layers × ple_dim]` = 5.6 GB at
    bf16, exceeding Vulkan's typical `maxStorageBufferRange` (2 GiB
    on most drivers). Stored as raw bytes on the host; per-forward
    we look up the `[T, num_layers * ple_dim]` slice and upload it.
  - **Bf16 GPU `select` shader** (`backends/gpu/shaders/select/select_bf16.comp`)
    — embedding lookup at bf16 by treating the table as packed
    `uint`s and copying row pairs.
  - **`ml.scalar` accepts a `dtype` parameter**, defaulting to F32.
    Required because `ml.mul` requires matching dtypes and Gemma
    multiplies bf16 tensors by scalar scales (`sqrt(hidden_size)`,
    `1/sqrt(2)`, the softcap divisor, etc.).
  - **`ml.select` now preserves input dtype** instead of always
    returning F32 (was hard-coded).
  - **`MAX_D` raised to 512** in `attention_cache{,_bf16}.comp` and
    the runtime cap in `backend.odin` — Gemma 4 full-attention layers
    use `head_dim=512`.
  - **`ml.rmsnorm` `eps` parameter** flowed through the GPU shaders
    (`rmsnorm`, `rmsnorm_bf16`, `rmsnorm_stats`, `rmsnorm_stats_bf16`)
    so Gemma's `eps=1e-6` doesn't fall back to a hard-coded `1e-5`.
- Gemma 4 E4B-IT (instruction-tuned) chat working end-to-end at
  GPU bf16. Greedy decode on `"What is the capital of France?"`
  returns `"The capital of France is Paris.<turn|>"` (8 tokens,
  matches HF F32 byte-for-byte). bf16 GPU forward parity vs HF F32
  reference: max abs logit diff **1.49**, mean **0.18** — pure
  numerical drift over 42 layers, no correctness gap. Supporting
  pieces:
  - **Tokenizer special-token preprocessing** (`tokenizers/gemma/`).
    Encoder now scans `added_tokens` for the longest match at each
    cursor position and emits the special ID directly; gaps go
    through normal byte-fallback BPE. Required for chat templates
    where `<bos>`, `<|turn>`, `<turn|>` need to be single token IDs.
  - **`examples/gemma_chat` chat template wrapping** behind `--chat`.
    Wraps the user message as `<bos><|turn>user\n…<turn|>\n<|turn>model\n`
    (Gemma 4's actual format, *not* the older `<start_of_turn>` /
    `<end_of_turn>` from Gemma 2/3). Stops on `<eos>` or `<turn|>`.
  - **Sampling extensions** in `examples/gemma_chat`: `--top-p`
    (nucleus, applied after `--top-k`), `--ignore-eos`, and
    `--no-cache` for diagnostic comparison vs `forward_cached`.
  - **`tools/gemma_dump.py` always overwrites** `model.safetensors`.
    Previously a size-only check left the prior variant's weights
    in place after switching `MODEL_ID` (base and IT have identical
    safetensors sizes), which was the actual cause of a multi-hour
    chase of phantom "bf16 forward bug" symptoms.
- Bug-fix sweep that came out of the chat work — done.
  - **GPU activation sub-allocator** (`backends/gpu/buffer.odin`,
    `_create_pooled_activation_buffer`). Replaced the exact-size
    `pool` with a bump-arena that resets on `clear()`. Activations
    no longer require one `vkAllocateMemory` per buffer, so
    `forward(prompt+N)` for varying N (which `--no-cache` chat or
    batch-shape-changing training does) stays bounded in VRAM.
    Fixes the `gemma_chat --no-cache` OOM.
  - **CPU bf16 `select`** (`backends/cpu/cpu.odin`). Forward is now
    a dtype-agnostic byte-row copy; backward has a Bf16 branch with
    f32 accumulation. CPU bf16 chat would now work end-to-end (GPU
    is still preferred for speed).
- Sliding-window-aware KV-cache eviction (ring buffer) — done.
  `attention_with_cache` modulo-indexes K/V rows by
  `t_capacity = k_cache.shape[0]` and the write splits at the wrap
  (CPU memcpy and GPU `vkCmdCopyBuffer`). New `t_capacity` push
  constant in `attention_cache{,_bf16}.comp`. `gemma.cache_make`
  allocates `[sliding_window, kv_size]` for sliding layers and
  `[t_max, kv_size]` for full layers, dropping each sliding layer's
  cache from ~256MB to ~1MB at bf16 in 128k context. Shared layers
  in Gemma now reuse the source layer's K/V tensor via an in-pass
  carry array instead of slicing the (possibly wrapped) cache
  buffer; `_cache_row_slice` removed. `tests/dtype_roundtrip` gains
  a ring test where `k_cache.shape[0] == window` so writes wrap
  repeatedly. Effective context is now bounded by full-layer
  storage, not sliding storage.

Remaining phases, in order:

1. **Gemma 4 E4B fine-tune** on a small corpus. Reuses the training
   loop from the stack-validation pretrain (run it second, point it
   at E4B weights). Note: `gemma.make` already supports
   `for_training=true` to allocate the full
   `{Data, Gradient, Adam_M, Adam_V}` buffer set; with FP32 master
   weights at 8B params the optimizer state alone is ~96 GB and
   will need partial CPU/disk offload.

2. **Stack-validation pretrain** at ~30–50M params. The mixed-precision
   training recipe; useful before fine-tuning E4B for real.
3. **From-scratch pretrain at 160M+ scale** matching a published
   recipe (Pythia-160M or SmolLM2-135M). The "trainable intelligence"
   claim landing.
4. **Gemma 4 multimodal** (vision then audio). Conv1d/2d + FFT come in
   here. Audio is E2B/E4B only; vision exists across all dense sizes.
5. Branch into one of {audio generation, world-model RL, MoE training}
   — roughly equal effort off the same trunk. (Gemma 4 26B-A4B gives
   us a real MoE target in-family.)

## Gemma 4 E4B text-only inference (next concrete milestone)

Gemma 4 was released April 2, 2026. Family overview (from the model
card and HF blog):

| Model    | Layers | Effective | Total | Context | Sliding | PLE | Audio |
|----------|--------|-----------|-------|---------|---------|-----|-------|
| E2B      | 35     | 2.3B      | 5.1B  | 128K    | 512     | yes | yes   |
| E4B      | 42     | 4.5B      | 8B    | 128K    | 512     | yes | yes   |
| 26B-A4B  | 30     | 3.8B act  | 25.2B | 256K    | 1024    | no  | no    |
| 31B      | 60     | 30.7B     | 30.7B | 256K    | 1024    | no  | no    |

26B-A4B is MoE: 8 active / 128 total + 1 shared expert. All sizes use
a 262K SentencePiece vocab. **AltUp and MatFormer are gone** in Gemma
4 (they were in Gemma 3n) — this is what makes E4B tractable.

E4B specifics confirmed from `gemma_data/config.json` (HF
`google/gemma-4-e4b`):

- 42 layers, hidden 2560, intermediate 10240.
- 8 query heads, 2 KV heads (4:1 GQA). `head_dim = 256` on sliding
  layers but **`global_head_dim = 512`** on full-attention layers — the
  q/k/v_proj output widths differ by layer type. Confirm by inspecting
  the safetensors weight shapes for one full vs one sliding layer's
  `q_proj.weight`.
- Layer pattern is exactly 5 sliding : 1 full, repeating 7×; full
  layers at indices 5, 11, 17, 23, 29, 35, 41.
- Sliding RoPE: `rope_theta = 10_000`, full rotation.
  Full RoPE: `rope_theta = 1_000_000`, `partial_rotary_factor = 0.25`,
  `rope_type = "proportional"` (this is the p-RoPE the model card
  alludes to — only the first quarter of head dims rotate).
- `num_kv_shared_layers = 18`.
- PLE: `vocab_size_per_layer_input = 262_144`,
  `hidden_size_per_layer_input = 256`.
- `final_logit_softcapping = 30.0` IS applied by
  `Gemma4ForCausalLM.forward`: `logits = cap * tanh(logits / cap)`
  after `lm_head`. Composed from existing primitives in our forward
  (`mul`, `tanh`, `mul`); no new op needed.
- `hidden_activation = "gelu_pytorch_tanh"` — the tanh-approximation
  GeLU, which is what our `ml.gelu` already implements
  (`backends/gpu/shaders/gelu/gelu.comp` confirms). No change needed.
- Tokenizer ships only as `tokenizer.json` (HF tokenizers fast
  format) — no `tokenizer.model` SentencePiece protobuf is published.
  Step 2 below needs revising accordingly.

Architectural deltas vs our existing Llama/SmolLM2 stack:

- **RMSNorm.** Plain `x * weight / rstd` — Gemma 4's `Gemma4RMSNorm`
  does *not* apply the `+1` weight offset that earlier Gemma
  generations did (verified by reading
  `transformers/models/gemma4/modeling_gemma4.py`). The
  `unit_offset` flag on `ml.rmsnorm` is therefore unused for E4B,
  but stays in as a harmless opt-in for hypothetical future ports
  of Gemma 2 / 3 weights. Eps is `1e-6` (vs our `1e-5` default) —
  small enough that bf16 numerics likely hide the difference, but
  add an `eps` parameter to `ml.rmsnorm` to be safe.
- **Pre + post norms around attention and MLP** (four norms per block,
  not two). Each block is `x + post_attn_norm(attn(pre_attn_norm(x)))`
  then `x + post_mlp_norm(mlp(pre_mlp_norm(x)))`.
- **GeGLU MLP**, not SwiGLU. `gate ⊙ gelu(up)` vs SmolLM2's
  `gate ⊙ silu(up)`. We have `gelu` already; this is just a wiring
  change in `networks/gemma`.
- **QK-norm + V-norm.** RMSNorm applied to Q and K (each `[head_dim]`,
  with learned scale) **after** the q/k projections but **before** RoPE.
  V also goes through an RMSNorm but with `with_scale=False` — i.e. just
  `x / rstd`, no learned weight. Easiest path on our side: pass an
  all-ones constant tensor as the v_norm weight. Q-norm is broadcast
  across heads — view Q as `[T*n_q_heads, head_dim]` (via `ml.reshape`)
  before the rmsnorm call, then reshape back.
- **Attention `scaling = 1.0`.** Because q_norm/k_norm pre-normalise
  Q and K, Gemma drops the usual `1/sqrt(head_dim)` factor on the
  attention scores. Our `ml.attention` hardcodes `1/sqrt(head_size)`.
  Cleanest workaround with no API change: at load time, multiply the
  `q_norm.weight` row by `sqrt(head_dim)`. RoPE preserves magnitude, so
  the pre-baked scale flows through and exactly cancels our attention's
  built-in divisor. Zero runtime cost.
- **Alternating local / global attention layers.** We already have
  sliding-window attention; this needs a per-layer
  `attention_type ∈ {local, global}` flag and a layer-pattern config
  (typically 5 local : 1 global, but read it off the HF config).
- **Dual RoPE configurations** per layer type: standard RoPE (small
  base, e.g. 10000) on local layers, **p-RoPE** ("proportional RoPE",
  partial RoPE that rotates only a fraction of head dims) on global
  layers for long-context. Add a `rope_fraction: f32 = 1.0` parameter
  to `ml.rope`; when < 1.0, only the first `fraction * head_dim` dims
  rotate, the rest pass through.
- **Per-Layer Embeddings (PLE).** Parallel low-dim conditioning
  pathway, smaller models only (E2B/E4B). For each token, a small
  per-layer vector is produced from a token-identity embedding plus a
  context-aware projection of the main embedding; each layer uses its
  vector to modulate hidden states via a lightweight residual block
  after attention and feed-forward. Contained — one new module that
  reads a `[vocab, n_layers, ple_dim]` lookup table plus a projection,
  applied at the end of each block.
- **Shared KV cache.** Last `num_kv_shared_layers` layers reuse K/V
  tensors from the last non-shared layer of the same attention type.
  Extension to our existing `Cache`: a shared layer's cache slot
  points at an earlier layer's slot instead of allocating its own.
- **262K SentencePiece tokenizer.** Replaces the GPT-2 BPE we have.
  Big vocab → embedding table is ~150 MB at f32 / ~75 MB at bf16, fine
  on 24 GB.

Implementation order, each step runnable:

1. **Read HF config + dump reference logits.** Pull the E4B
   `config.json` and write `tools/gemma_dump.py` mirroring
   `smollm_dump.py`: download the HF checkpoint, run `transformers`
   forward on a fixed prompt, save logits and the per-layer
   `attention_type` / `num_kv_shared_layers` / `rope_fraction` /
   PLE config to JSON. This pins the exact numbers (hidden_size,
   head_dim, q/kv head counts, intermediate_size, RoPE bases, etc.)
   the model card omits.
2. **Gemma tokenizer** in `tokenizers/gemma/`. Gemma 4 ships only
   `tokenizer.json` (HF tokenizers fast format), not the SentencePiece
   `.model` protobuf. Two viable paths: parse `tokenizer.json`
   directly (it embeds the byte-fallback BPE vocab + merges in a
   well-documented JSON schema), or extend `gemma_dump.py` to extract
   vocab/merges from the live tokenizer into a compact binary that the
   Odin side reads. The latter is faster to land; the former keeps the
   loader symmetric with `tokenizers/gpt2/`. Implementation is still
   byte-fallback BPE with whitespace-prefix pre-tokenization (no
   regex). Acceptance: encode/decode parity against HF on ~50 fixed
   prompts including multilingual + code.
3. **`ml.rope` `rope_fraction` parameter** (CPU + GPU forward, plus
   backward for the eventual fine-tune). Backward parity test added
   to `pytorch_parity` at `rope_fraction = 0.5`.
4. **`ml.rmsnorm` `+1` weight offset flag.** One more bool on the op
   variant; both backends; parity test.
5. **`networks/gemma` module.** Composes the above into a Gemma block
   (pre-attn-norm → q/k/v_proj → QK-norm + V-norm-no-scale → RoPE
   (per-layer config: `head_dim=512, base=1e6, fraction=0.25` on full
   layers; `head_dim=256, base=10000, fraction=1.0` on sliding) →
   attention (local on sliding layers, with `sliding_window=512`) →
   o_proj → post-attn-norm → residual → pre-feedforward-norm → GeGLU
   MLP (tanh-approx GeLU, already in `ml.gelu`) → post-feedforward-norm
   → residual → PLE block). The q/k/v_proj output widths differ by
   layer type (`global_head_dim=512` on full vs 256 on sliding) — store
   per-layer in the config. Attention `scaling=1.0` is achieved by
   pre-baking `sqrt(head_dim)` into `q_norm.weight` at load time, so
   our existing `ml.attention` (which divides by `sqrt(head_size)`)
   produces the unscaled scores Gemma wants. Apply
   `final_logit_softcapping = 30.0` (`logits = cap * tanh(logits / cap)`)
   after `lm_head`. Each layer also has a per-layer trained `layer_scalar`
   buffer (`[1]`, values ~0.05–0.78 in the released checkpoint, NOT a
   no-op) that multiplies the post-PLE residual at the end of the
   block. Field names match HF safetensors. Provides
   `GEMMA4_E4B_CONFIG`.
6. **Gemma loader** in `networks/gemma/loader.odin`. Walks HF tensor
   names, skips vision-tower (`vision_tower.*`) and audio-tower
   (`audio_tower.*`) prefixes for text-only inference, loads the rest.
   Same Q/K row-permutation issue as Llama (HF `rotate_half` ↔ our
   interleaved RoPE pairs) — verify whether Gemma's HF impl uses
   `rotate_half` or interleaved before assuming the SmolLM2 permute
   carries over.
7. **PLE pathway.** New op or composed primitives: per-layer embedding
   lookup + projection + residual injection. Parity test against a
   PyTorch reference of the same pathway.
8. **Shared KV cache layers.** Extend `networks/llama.Cache`-style
   cache (or fork it for Gemma) so a layer's cache can alias an
   earlier layer's cache. KV-write is skipped on shared layers; reads
   index the shared slot.
9. **`examples/gemma_inference/main.odin`.** Allocates
   `GEMMA4_E4B_CONFIG`, loads safetensors, runs forward on a fixed
   prompt, compares to the dumped HF logits. Acceptance: max abs
   logit diff ≤ 5e-4 (slightly looser than SmolLM2's 3e-4 because of
   the bigger model and bf16 weights — adjust if it lands tighter).
10. **`examples/gemma_chat/main.odin`.** Same pattern as
    `smollm_chat`: prefill + per-token decode through KV cache, with
    sliding-window-aware cache eviction once context exceeds 512
    tokens. Greedy + top-k/temperature sampling.

Once steps 1–10 land, fine-tune is the next phase below — most of the
remaining work there is data plumbing and the mixed-precision training
loop, not new ops.

Open questions, mostly resolved by the config dump:

- `rope_fraction`: 0.25 on full-attention layers, 1.0 on sliding.
  ✔ resolved.
- Layer pattern: 5 sliding : 1 full, repeating 7×. ✔ resolved.
- `num_kv_shared_layers`: 18. ✔ resolved.
- PLE dimension: 256 (`hidden_size_per_layer_input`); per-layer-input
  vocab is 262144 (same as token vocab). ✔ resolved.

Still open after step 1, to nail down by reading HF's Gemma 4 source:

- `rotate_half` vs interleaved RoPE — the dump script's
  `sniff_rope_convention` will report this; if it returns
  `"rotate_half"`, the Q/K row permute from `networks/llama/loader.odin`
  applies as-is.
- Exact PLE residual injection point (post-MLP residual vs
  end-of-block addition vs gated mix). Read
  `transformers/models/gemma4/modeling_gemma4.py`.
- Whether QK-norm in Gemma 4 uses the same `+1` offset RMSNorm as the
  block norms or a plain RMSNorm.
- Whether the `proportional` `rope_type` on full layers does anything
  beyond `partial_rotary_factor` (e.g. base-frequency scaling for long
  context). Check `rope_utils.py` in transformers.

## Gemma 4 E4B fine-tune

Once forward-parity inference lands, point the stack-validation
pretrain loop (below) at E4B safetensors and a small instruction or
domain corpus. Mixed precision (FP32 master + bf16 compute + FP32
Adam) is mandatory at this scale — at 8B params, FP32 Adam state
alone is ~96 GB, so optimizer state will need to live partly on CPU
or on disk. Plan that out before starting the run; this is the first
milestone where 24 GB VRAM is genuinely tight.

## Stack-validation pretrain

Overnight run, ~30–50M params:

- Streaming dataset loader (FineWeb-Edu sample or similar).
- Training loop with gradient accumulation, checkpointing, periodic
  eval. Lives in `examples/pretrain_small/`.
- Mixed-precision recipe: FP32 master weights, bf16 compute, FP32 Adam
  state. Doubles as the end-to-end mixed-precision example.
- Acceptance: loss drops from ~10 nats to ~4 nats on FineWeb-Edu within
  a few hours. Curve shape matches published references for similarly
  sized models.

Reuses the GPT-2 BPE tokenizer from `tokenizers/gpt2/` (already SmolLM2
compatible) and the safetensors loader from `loaders/safetensors/`.

## SmolLM2-135M fine-tune

Once the pretrain loop above is validated, point it at a small
instruction or domain corpus and a few thousand steps. Acceptance: loss
drops cleanly, greedy-decode samples reflect the fine-tune corpus.

## GPU performance

- **Make attention coopmat profitable.** Forward shader exists
  (`attention_bf16_coopmat.comp`) but currently ~18% slower than the
  SIMT bf16 shader on bench shapes (D=64, T=64–256). Most promising
  next moves: parallelize the per-row softmax across threads (use a
  (row, col) thread layout instead of sequential rows), reduce barrier
  count by fusing Q-load with the first K iteration, multi-subgroup
  BR=32 to amortize K/V staging, specialization constants for D so
  shared-memory isn't sized for the worst case. Once forward wins, port
  the same Q-tile layout to backward (currently three separate kernels
  with no Q/K tiling). Revisit once 135M training benches show
  attention as a bottleneck — at seq=1024 the coopmat path may already
  win without further tuning.

- **Flash Attention v2 — Q-tiled forward and tiled backward.** Forward
  uses online softmax with K streamed in BC=64 tiles, but still BR=1
  (one workgroup per `(head, query)`). Real FA2 also tiles Q (BR>1) so
  K and V are reused across the BR×BC score block. Backward is still
  3 kernels (D-precompute, dKV, dQ) with no Q/K tiling. Tiling both
  should pull `attention_causal` from ~3x off cuDNN toward ~1.5x. Note:
  partly subsumed by the coopmat attention work above. Reference:
  ggml's `flash_attn.comp` and the Tri Dao FA2 paper.

- **Larger-shape GPU bench coverage.** Current speed bench uses small
  shapes where Vulkan's per-dispatch overhead dominates. Add benches at
  training-realistic shapes (transformer step at seq=512, embed=768) to
  see steady-state kernel performance, not launch overhead.

- **Cache descriptor sets in `_dispatch`.** Every `_dispatch` call does
  a fresh `vkAllocateDescriptorSets`, which is the per-shader floor
  (~50-100 us based on the small-op timings). Caching by
  `(pipeline, buffers)` would shave that for repeated identical
  dispatches.

- **GPU-side fill kernels.** `ml.fill_value` and `ml.fill_normal`
  currently build the buffer on the CPU and do a synchronous upload,
  which forces a `vkQueueWaitIdle` mid-frame. A `fill_constant.comp`
  and a Philox-style `fill_normal.comp` would keep initialization
  on-device.

## GPU robustness

- **Hardcoded `head_size <= 256` in attention backward shaders.** The
  shared memory caches (`k_shared`, `v_shared`, `q_shared`, `do_shared`)
  are fixed `float[256]`. Asserted at dispatch time but should either
  grow dynamically via specialization constants or auto-fall-back to a
  tiled variant.

- **`head_size > WG=64` in attention backward kernels.** Per-thread
  d_K/d_V accumulators only handle one `d` element each. Heads larger
  than 64 would drop the rest. Either fix the loop or use specialization
  constants.

- **Vulkan specialization constants for shader tile sizes.** Tile sizes
  currently live in two places (the shader and `ops.odin`), which has
  produced at least one mismatch bug. Specialization constants make the
  shader the single source of truth. ggml does this throughout.

- **Regenerate `.spv` after `rope` push-constant change.** Run
  `backends/gpu/shaders/build.bat` after pulling — the `position_offset`
  parameter added for KV-cache decode grew the push-constant layout by
  a u32.

## CPU performance

- **Real GEMM (or link BLAS) for `linear` and `batched_matmul`.** CPU
  `linear_fwd` is 6-8x off MKL/OpenBLAS at the bench shape. Fastest
  path is linking OpenBLAS for `cblas_sgemm`; afternoon of work, becomes
  as fast as PyTorch on matmul, costs a vendor dependency. Hand-rolled
  tiled GEMM is weeks-to-months for a competitive one.

- **Switch CPU attention from `softmax_outputs[H, T, T]` to LSE-only.**
  Current CPU `attention` saves the full softmax matrix for backward
  (~2 MB per call at the bench shape). Flash Attention v2 saves only
  `lse[H, T]` (~8 KB) and recomputes attention weights in backward.
  Same algorithm we use on GPU already; would unify the two scratch
  layouts and matter at long sequence lengths.

## Test coverage

- **`_big` variants for every parity test.** Existing template:
  `linear_big`, `rmsnorm_big`, `attention_gqa_big`, `attention_window_big`.
  Worth adding for `batched_matmul`, `softmax`, `layernorm` so any
  future tile-size tuning trips the suite immediately.

- **GPU stress test for races on `linear_backward` style accumulation.**
  The CPU `linear_backward` two-pass fix solved the race; the GPU
  version should be audited the same way and have a stress test that
  runs the same input many times and asserts bit-identity.

## Other deferred items

- **`select` bf16 path.** SmolLM2 fine-tune in mixed precision keeps
  `token_embeddings` as F32 master and casts the lookup output to Bf16;
  a bf16 select would shave one cast per step but is not blocking.

- **F16 GPU support for `cast_to`.** CPU is full {F32, F16, Bf16}; GPU
  is F32 ↔ Bf16 only. Gemma 4 is bf16, so this can wait until something
  asks for it.

- **mmap safetensors loader.** Currently reads the whole file into RAM.
  Defer until 8B-class checkpoints make it matter.

## Style / housekeeping

- `ml.fill_normal` / `ml.fill_value` use Odin's RNG (`core:math/rand`)
  seeded per process. PyTorch parity tests sidestep this by uploading
  PyTorch-generated weights. If you ever want bit-identical RNG between
  CPU and GPU paths, you'll need a deterministic in-library RNG.

- Bench `_checksum` downloads 1 float as a sync token. Cute, but a
  `ml.sync()` proc that just submits + waits without a download would
  express intent more clearly.

- The `attention_fused` vtable flag was removed once both backends had
  a fused Attention. If a third backend is added that wants the old
  composition path, resurrect `_attention_compose` (it's in git history)
  and the flag.
