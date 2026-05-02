Next round (per-dispatch tax, ~75 tok/s target):

1. Pre-recorded command buffer for forward_cached. Push constants moved to a
   uniform buffer that the host updates per token (cache_position, etc.).
   Today we re-record ~1500 dispatches per token at ~2 ms CPU cost; replaying
   a recorded CB sidesteps that and cuts queue submission overhead too. See
   llama.cpp's `ggml_vk_graph_compute` for the multi-CB pattern with split
   submits at boundaries that depend on host-side data (e.g. cache length).
2. Q4_K coopmat path. Mirror `linear_bf16_coopmat.comp` for in-shader
   Q4_K dequant feeding a coopmat tile accumulator. Largest remaining wedge
   per the gap table (~5-7 ms/tok).

Done:
- mmvq (Q8_1 + integer dot) for Q4_K decode: 37.2 -> 43.8 tok/s (+17.8%, ~4.1 ms/tok).
- rmsnorm+rope fusion (q/k_norm path, 60 dispatches/tok eliminated):
  43.8 -> ~44.5 tok/s.
- add+rmsnorm fusion (post-attn residual, 30 dispatches/tok eliminated):
  GPU-work -11%, wall wash. Kept; compounds with #1.
