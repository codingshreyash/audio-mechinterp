# Qwen audio resource and read-only parity pilot V1

## Scope and question

This is a **technical NON-STUDY pilot**, not a calendar-security result. It measures whether the pinned Qwen2.5-Omni-3B Thinker can process one 16-second audio input on a bounded H100 80 GB allocation and whether passive hooks at shared Thinker post-block residuals 17 and 34 preserve the output. It does not score LIST/ASK/CANCEL, fit directions, compute gradients, patch activations, or make any claim about the six development (D_*) or locked (H_*) clips. Those target clips remain blocked by the real 18/18 two-listener QA and independent review in `audio_candidate_v1.py`.

## Frozen stimulus and measurement

`models/qwen_omni/sample_apple_5p5s.wav` is PCM16 mono 16 kHz, 88,000 samples, SHA-256 `ba226916abdd36bef212a0b478d49eb63028fb541d84a7934f50e4871fe6d21d`. Append exactly 168,000 zero PCM samples and write a canonical WAV with `wave`; final file `nonstudy_apple_zero_pad_16s.wav` is 256,000 samples, 512,044 bytes and SHA-256 `b84ce83e90ad498e93063a27e8cba9a3a860985e5801c55385a95e0483735973`. The fixed ID is `NON_STUDY_APPLE_ZERO_PAD_16S`; the system instruction is `Respond briefly to the user's audio.` No text is appended to the user audio. Recompute all bytes before inference and in saved audit.

Use the pinned `adapter.prepare_audio_batch` and `audio_candidate_v1.validate_audio_batch` shape/placeholder logic directly. Do not call `source_plan`, `reviewed_qa`, or `prepare_development_case`; they belong to the separate D_* study. A no-weight pinned-processor check on the frozen WAV gave input IDs/mask `[1,423]`, audio features `[1,128,30000]`, feature mask `[1,30000]`, **1600 active mel frames**, 800 encoder rows, and 400 projected rows at placeholder positions 17–416. Freeze those dimensions as a pre-weight requirement. Save and audit all four processor tensors and the timing warning. A 16-second padded example measures input length/resource behavior, not the acoustics of the six calendar clips.

Load the exact cached checkpoint/revision with FP32 and TF32 disabled. On the same audio batch, perform one stock no-cache forward and one passive-hook no-cache forward. Capture full first-step vocabulary logits and prompt-last states at `text.residual.17` and `.34`. Passive hooks return `None` and must fire once at both sites. The full first-step logits must have maximum absolute difference at most `1e-5`. The states are descriptive; no causal interpretation is permitted.

Run deterministic greedy `max_new_tokens=12` stock and passive-hook generation with EOS 151645. Save full raw continuation token IDs, hook call shapes, terminal-EOS flag, and stop reason. Paths and stop reasons must agree exactly. Record a native response ending at EOS, a malformed/early-EOS path, or a length-censored path as observed; a non-label path is allowed. The pilot makes no answer-quality or task-correctness claim.

## One-shot execution gate and bounds

Source-only review, no-weight admission, independent executable review, and a later committed one-shot GO are required in that order. No admission, review/GO artifact, or inference is created by this source implementation. The fixed attempt is `runs/audio_resource_pilot_v1/attempt_001`; it must be absent before launch. The same-user parent pipe and nonce are liveness guards, not cryptographic provenance against a malicious local owner. The saved auditor requires raw parent, worker, resource, and sidecar evidence.

Run only inside a reviewed isolated single-node Slurm H100 80 GB allocation. Stage the repository, pinned venv, and exact snapshot in `/lustre/fsw/portfolios/nemotron/projects/nemotron_speechprod_asr/users/shrranjan`, because remote home has insufficient capacity. Bind the actual Slurm job ID, node, allocated GPU list, physical GPU UUID, runtime, and snapshot in admission and GO, and recheck the same identity at parent and worker start. No allocation is granted by this protocol. Caps: 900 seconds including terminal reserve, sampled physical GPU use below 60 GiB, PyTorch peak reservation below 60 GiB, process-tree RSS below 24 GiB, and whole attempt evidence below 64 MiB. Missing samples, cap crossings, worker failure, changed source/input, or broken receipt invalidate the attempt; do not rerun into the fixed path or silently increase caps.

## Saved audit and next decision

The independent saved auditor reconstructs the processor batch from frozen WAV bytes without model weights, checks every tensor and sidecar hash, exact stock/read-only first logits, hook census/shapes, generated token/stop parity, parent time/identity chain, and measured limits. A PASS means technical feasibility and read-only parity for this constructed input. It does not authorize D/H inference. If the pilot fails resource or parity gates, inspect the raw failure and design a new reviewed attempt before any study run.
