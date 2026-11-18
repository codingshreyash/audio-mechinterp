# Independent plan review: Whisper FIT spread-versus-use assay

18 November 2026. Review target: committed `WHISPER_FIT_SPREAD_CAUSAL_PROSPECTIVE_V1.md` at `8419a4b`. **Decision: GO for implementation only.** No model weights were loaded and no inference or intervention was run for this review. The runner, auditor, no-weight preflight, and source reviews remain prerequisites to the planned assay.

## Checks made independently

- Rehashed the corpus v1 rendered manifest: SHA-256 `adf8cd02d24a43ef576bcb993dc54bd2527440a62798e12fc788def3c063971f`. For the eight selected `fit_chosen` WAVs, recomputed file and raw PCM hashes from bytes, checked 16-kHz mono 16-bit format, equal length within each pair/voice, exact PCM equality outside the manifest word slot, and a nonzero difference inside it. All passed.
- Recomputed `W`, `P`, `Q`, and `N=ceil(WAV_frames/320)` from the manifest. The plan's four rows are exact: color/Piper `W[90,161) P[10,81) Q[171,242) N272`; color/Kokoro `W[101,184) P[10,93) Q[194,277) N287`; object/Piper `W[90,139) P[10,59) Q[149,198) N250`; object/Kokoro `W[101,175) P[10,84) Q[185,259) N279`. Each `P` and `Q` is the same width as its `W`, disjoint from it, and inside `[0,N)`.
- Read the previously saved, separately audited pre-block-0 capture arrays without running a model. In every pair/voice, donor and recipient were bitwise equal at all `C=[0,N)\W` rows and differed in `W`. Their saved post-block-0 states differed at row 0. This supports the plan's input-locality gate and numerical-spread motivation for these existing FIT clips; the new run must still capture and recheck its own states.
- Recomputed the matrix: eight unique baseline WAVs plus eight contexts × two sites × seven arms plus eight final-state full-donor arms = **128 decodes**. Recomputed the specified SHA-256 seed derivation for all 32 random arms; all 32 derived seeds were distinct. The future runner must verify the generated edit hashes are also distinct, since seed uniqueness alone does not check realized noise tensors.

## Interpretation boundary

The exact intervention sites and outcome are clear. The post-block-0 output includes the whole encoder block, including its self-attention and feed-forward computation; it is not an isolated attention-head output. `C` is far larger than `W`, and `Q` may contain later speech. A `C`, `P`, or `Q` result must be interpreted as sufficiency of that patched region in this synthetic context, not a unique word circuit. Failure of a region patch cannot establish that the region contains no lexical information. Both pairs and this question were selected after viewing prior FIT results, so any result is exploratory within one English ASR checkpoint and cannot establish a security vulnerability or cross-model transfer.
