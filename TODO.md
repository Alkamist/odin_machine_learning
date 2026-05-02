Next round (per-dispatch tax, ~75 tok/s target):

1. Pre-recorded command buffer for forward_cached. Push constants moved to a
   uniform buffer that the host updates per token (cache_position, etc.).
   Today we re-record ~1500 dispatches per token at ~2 ms CPU cost; replaying
   a recorded CB sidesteps that and cuts queue submission overhead too. See
   llama.cpp's `ggml_vk_graph_compute` for the multi-CB pattern with split
   submits at boundaries that depend on host-side data (e.g. cache length).
2. Prefill batching for arbitrary M. Q4_K/Q6_K coopmat tile shaders are
   M%64==0 only; chat REPL chunks prefill into a 64-aligned head and a
   single-token tail. To get full prefill speed for any prompt, either pad
   the trailing batch internally (with a row-mask in the coopmat shader)
   or grow the tile to a smaller-M variant.

Done:
- mmvq (Q8_1 + integer dot) for Q4_K decode: 37.2 -> 43.8 tok/s (+17.8%, ~4.1 ms/tok).
- rmsnorm+rope fusion (q/k_norm path, 60 dispatches/tok eliminated):
  43.8 -> ~44.5 tok/s.
- add+rmsnorm fusion (post-attn residual, 30 dispatches/tok eliminated):
  GPU-work -11%, wall wash. Kept; compounds with #1.
- Q4_K + Q6_K coopmat tile shaders (BM=64, BN=64, BK=32). In-shader Q4_K /
  Q6_K dequant -> bf16 in shared memory, coopmat fp32-acc matmul. Mirrors
  llama.cpp's mul_mm path for quantized weights. Wires into prefill (M>1);
  decode (M=1) keeps mmvq. Real-tensor parity vs bf16-staged reference:
  Q4_K max_abs=0.002, max_rel=0.007; Q6_K max_abs=0.004, max_rel=0.006.
- Chat REPL ported to Q4_K_M GGUF (--gguf path or default model.gguf).
  Prefill chunked into 64-aligned coopmat batches + single-token tail.
