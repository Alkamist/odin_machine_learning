# Audio generation — plan and open work

Notes captured while pivoting from text training to the audio target.
Everything below is a plan, not a commitment — revisit before each step.

## Goal

Two tasks the model should eventually support, both via the **same decoder**:

1. **Text → audio.** Caption / instruction in, audio out.
2. **Text + audio → audio.** Caption / instruction + reference clip in, transformed clip out (style transfer, denoise, voice swap, "make this happier", etc).

Both fit one architecture: a Llama+QK-norm decoder over discrete tokens, exactly
like the TinyStories trainer, with a richer token layout.

## Architecture

- Decoder-only LM. Reuse `networks/llama` with `use_qk_norm = true`.
- Inputs are integer token IDs in a shared vocabulary that mixes text and audio.
- All conditioning is via prefix tokens (no cross-attention, no encoder).

Why decoder-only over diffusion / encoder-decoder:

- Same training loop, same parity tests, same sampler. No new infra.
- Matches MusicGen / VALL-E / Bark / AudioLM family — most modern audio LMs.
- Cleanly handles both tasks (1) and (2) as different prefix layouts.
- Diffusion gives better unconditional quality at parity FLOPs but needs new
  ops (denoising loss, schedules, classifier-free guidance) — punt for v2.

## Codec

- **EnCodec 24kHz** as the audio tokenizer. Available via
  `transformers.EncodecModel.from_pretrained("facebook/encodec_24khz")`.
- 4 codebooks @ 3 kbps for the first PoC (smallest reasonable; 8 / 6 kbps when
  scaling up).
- Frame rate is fixed at 75 Hz. 1024 entries per codebook.
- 10 seconds of audio at 4 codebooks → 750 frames × 4 codes = 3000 codebook
  entries (or 754 with delayed pattern, see below).

Alternatives if EnCodec is unsatisfactory at 3 kbps:

- **DAC** (Descript): higher fidelity at lower bitrate, slightly less common.
- **SoundStream / Mimi** (Moshi's codec): also good, smaller community.
- **Stick with EnCodec for v1.** Swap codecs only if reconstruction quality
  blocks us — verifying that with `audio_decode.py` is a 5-minute test.

## Multi-codebook handling

Each audio frame is K codebook entries, not one token. Three established
strategies; we'll use the second:

1. **Flatten** — sequence becomes K× longer. Wasteful, never used in modern
   models.
2. **Delayed pattern** (MusicGen) — codebook k is shifted by k frames so all K
   codes per frame can be read off from one position's logits via K output
   heads. Effective seq_len = T + K. **This is the plan.**
3. **Parallel heads** (VALL-E / Stable Audio) — K heads predict all K codes per
   position simultaneously without delay. Slightly more efficient than (2) but
   training is trickier (masking, ordering of inputs vs targets).

Delayed pattern in practice:

- During training, the input/target tensors are constructed so codebook k is
  offset by k positions vs codebook 0.
- Single output head, single softmax per position; the model "looks" at the
  position offset by k to predict codebook k.
- We do the reordering at training time (in the Odin trainer), not in the .bin
  file — keeps the on-disk format codec-canonical.

## Token vocabulary layout

One shared softmax / embedding table for everything:

```
0 .. 49151                     SmolLM2 BPE text tokens
49152 .. 49152 + 1024 - 1      audio codebook 0
49152 + 1024 .. 49152 + 2048-1 audio codebook 1
49152 + 2048 .. ...            audio codebook 2
49152 + 3072 .. ...            audio codebook 3
...                            special tokens at the very top
  TEXT_END
  AUDIO_START
  AUDIO_SEP
  AUDIO_END
```

For 4 codebooks: total vocab = 49152 + 4096 + (~4 specials) ≈ 53,252.
Manageable. Tied embeddings still work.

## Sequence formats

Task 1 — text → audio:

```
[text_tokens] [TEXT_END] [AUDIO_START] [audio_tokens with delayed pattern] [AUDIO_END]
```

Task 2 — text + source → output:

```
[text_tokens] [TEXT_END] [AUDIO_START] [source_tokens] [AUDIO_SEP] [target_tokens] [AUDIO_END]
```

Loss-mask everything before the *generated* portion (text + first AUDIO_START
for task 1; text + source + AUDIO_SEP for task 2). Same SFT-style masking
trick that any instruct LM uses.

## Datasets

For task 1 only (start here):

- **LibriTTS-R** — ~585 h clean speech with transcripts. Best for early
  iteration: structure is simple, you can hear semantic correctness easily,
  and pretrained baselines exist for comparison.
- **AudioCaps** — ~50k 10-sec clips with general-sound captions. Good for the
  "describe a sound, generate it" demo.
- **MusicCaps** — ~5k music clips with descriptions. Tiny but evocative.

For task 2 (instruction-guided editing):

- **Synthetic pairs.** Take any clean audio, apply a known transform
  (pitch shift, EQ, time-stretch, reverb, additive noise, vocoder swap),
  label it with text describing the operation. Mechanical, infinite data.
  This is what the InstructME / AUDIT family does.
- **VCTK / LibriTTS** for parallel-speaker voice swap pairs.

Path of least resistance: **train task 1 first** on LibriTTS-R, validate the
whole pipeline (codec → LM → audio out → it sounds like speech), then add
task 2 by mixing synthetic edit pairs into the same training set with the
right prefix layout.

## Tools already in tree

- `tools/audio_dump.py` — encode a directory of audio (+ optional .txt
  captions) into `.bin` + `.idx` paired with SmolLM2 BPE token IDs. Documented
  format in the file header.
- `tools/audio_decode.py` — round-trip a single example back to a `.wav`.
  Use this **before any model training** to confirm:
  - EnCodec at chosen bandwidth sounds OK on your data,
  - the binary format round-trips losslessly,
  - the BPE tokenizer round-trips your captions.

## Work to do (Odin side)

1. **Trainer for paired audio data.** Mostly clone of the TinyStories trainer
   with these additions:
   - Read `.bin` + `.idx`, randomly sample one full example per step (no
     midstream window slicing — we want whole songs / sentences for now).
   - Apply the codebook offset (`code + 49152 + k * 1024`) when building the
     token stream.
   - Apply delayed-pattern shift on input and target tensors.
   - Loss-mask the prefix portion.
2. **Audio-aware sampler.** During generation, decode K codes per audio frame
   from the appropriate (delayed-pattern-shifted) positions in the rolling
   logits, undo the offsets, hand back to EnCodec for the wav.
3. **Generation tool** `tools/audio_generate.py` (or extend the chat REPL):
   take a text prompt, run the trained model, write a `.wav`.

## Work to do (Python side)

1. **Sanity-check EnCodec quality on real data.** Encode + decode a handful
   of representative clips at 3 kbps, listen. If 4 codebooks is too lossy for
   the target task, bump to 8 (6 kbps) and accept 2× longer sequences.
2. **Dataset prep scripts** for whichever first dataset we pick (LibriTTS-R is
   easiest — `datasets.load_dataset("mythicinfinity/libritts_r", "clean")`,
   stream wav + text into the existing `audio_dump.py` directory layout).
3. **Synthetic edit-pair generator** for task 2 (pitch / EQ / reverb / noise
   transforms with auto-generated captions). Don't write this until task 1
   is working end-to-end.

## Open questions / decisions to revisit

- **Bandwidth.** Start at 3 kbps (4 codebooks). Bump if reconstruction is
  unacceptable on our target audio.
- **Model size.** First run probably ~80–150M params. Audio needs more
  capacity per token than text — TinyStories at 50M was right because each
  token is ~1 word; here each token is ~13 ms of audio, much less semantic
  content per token.
- **Sequence length.** With delayed pattern at 4 codebooks @ 75 Hz, 10 s
  audio = 754 tokens, 30 s = 2254. Start with 10 s clips. Need bf16 weights
  and possibly larger ACCUM_STEPS to fit longer contexts.
- **Text conditioning depth.** For v1, just prepend BPE tokens. If quality is
  bad, swap in a frozen T5-small encoder feeding cross-attention — but that's
  a new op (no-cross-attn currently in the repo) so save it for v2.
- **Classifier-free guidance.** Standard trick: occasionally drop the text
  prefix during training (replace with a NULL token), then at inference
  interpolate logits between conditioned and unconditioned. Cheap to add,
  meaningful quality bump. Add once the basic loop is working.
- **Whether to share text and audio vocab.** Sharing is simpler (one embedding
  table, one softmax). Splitting (separate text head + audio head) is what
  some recipes use. Start shared; revisit if the lm_head softmax of 53k
  becomes the bottleneck.

## Likely first run, when we get there

- Codec: EnCodec 24kHz, 4 codebooks (3 kbps).
- Data: LibriTTS-R "clean" subset (~245 h), text→audio only.
- Vocab: 53,252 (49152 text + 4096 audio + ~4 specials).
- Model: 8-layer Llama+QK-norm, embedding 640, intermediate 2560,
  n_q_heads=10, n_kv_heads=2, head_size=64. Roughly 75M params.
- Sequence: 1024 tokens (covers ~12 s of audio with text prefix).
- Schedule: cosine LR, peak 4e-4, warmup 500, ~50k steps. Estimated 4–6 GPU
  hours on the 3090 Ti.
- Sample protocol: every N steps, generate from a fixed list of prompts and
  write .wavs to disk so progress is audible.
