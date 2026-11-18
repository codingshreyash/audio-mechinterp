# Bounded audio-conditioned Jacobian cost pilot

**Status: completed 16 of 2,048 target coordinates on one audio prompt.** This measures a small part of a transport fit and validates the gradient path. It is not a fitted J-lens, a readout evaluation, or a security finding.

## Exact experiment

The [benchmark script](benchmark_jacobian.py) loaded the pinned Qwen2.5-Omni-3B **Thinker** and the [5.5-second Apple Samantha clip](sample_apple_5p5s.wav) on local Apple Silicon MPS. The checkpoint and Transformers pins, audio SHA-256, torch version, position masks, and per-batch measurements are in [jacobian_16coords.json](jacobian_16coords.json). No paid GPU was used.

The source states were post-block text residuals at layers 0 and 17. The target was the post-block residual at layer 35, before final normalization and unembedding. We selected 16 evenly spaced target dimensions (`0, 128, …, 1920`), grouped into two batches of eight. Each batch replicated the **same audio input** eight times, placed a separate one-hot cotangent on each target dimension at every non-padding prompt position, and computed one vector-Jacobian product. For each source layer, gradients were averaged over the 137 audio placeholder positions and saved as float32 rows. This follows the reference J-lens's batched coordinate-gradient approach, with an explicit audio-position source mask. The causal text decoder makes earlier target positions contribute zero gradient to later source positions. This source-position average is a **cost and plumbing estimator**, not directly a terminal-position J-lens readout. A semantic audio-to-text lens needs a chosen source/target position policy and held-out validation.

## Measured result

| Batch | Coordinates | Forward | Backward | MPS driver allocation after backward |
| --- | ---: | ---: | ---: | ---: |
| 1 | 8 | 1.943 s | 0.661 s | 19.414 GiB |
| 2 | 8 | 0.858 s | 0.449 s | 19.601 GiB |

Total measured model work for 16 coordinates was **3.910 seconds**; cached model loading took about 3 seconds separately. The run stayed below its 30 GiB driver-memory guard and the Mac's approximately 37.4 GiB MPS recommended maximum. Both saved `16×2048` row tensors are finite; their overall L2 norms were 4.416 (layer 0) and 4.313 (layer 17). The first batch includes warmup effects. A naive linear multiplication by `2048/16` gives **500 seconds, about 8.3 minutes**, for one full-width prompt with the same two source layers. This is an **upper-bound-ish arithmetic scale check**, not a runtime prediction: this pilot reruns the forward pass for every eight-coordinate batch, while the upstream estimator can reuse a graph across backward batches. It is not a measured full fit or a forecast for 100 prompts or 36 layers. The memory guard checks allocation after model operations; it cannot prevent an instantaneous out-of-memory failure inside a kernel.

The script atomically checkpoints each batch. Rerunning a completed checkpoint skipped both batches. To test interruption recovery, we made a copy with only the first batch marked complete; the script computed only the second batch and recreated both row tensors **bit for bit** (`max_abs_row_delta = 0.0`). The resumability records are [jacobian_resume_check.json](jacobian_resume_check.json) and [jacobian_partial_resume_check.json](jacobian_partial_resume_check.json). The original [checkpoint](jacobian_16coords.pt) retains the 16 measured rows per source layer.

## Limits and next gate

The selected input was one synthetic voice and one request. Finite rows and a short runtime say nothing about whether token readouts are meaningful or whether the directions causally control output. The source-position mask is specific to Omni's audio placeholders; a text-only Anthropic fit uses a different position distribution. The full 2,048-coordinate matrix, other layers, and corpus fit remain unmeasured. The next design step is to fix the source/target position policy for a semantically interpretable lens, then run held-out readout and causal controls under [the evaluation protocol](../../eval/PROTOCOL.md) on a small frozen set before expanding compute.
