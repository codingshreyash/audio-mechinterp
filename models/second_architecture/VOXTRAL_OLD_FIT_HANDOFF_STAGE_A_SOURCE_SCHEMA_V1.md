# Prospective old-FIT handoff Stage A source schema

This describes the source-only runner under independent review. It is not a
result or authorization to run a model. The fixed protocol is
`VOXTRAL_OLD_FIT_NUMERICAL_HANDOFF_PROSPECTIVE_V1.md` at SHA-256
`61a67ac87a4e4e17cb0b1f7ad972f25fda4e0d90e0b70798e7144b3fd6f2484b`.

Each arithmetic arm writes a new exclusive local attempt directory. Large
sidecars are moved after successful exclusive serialization to
`blobs/<sha256>.pt`; the result stores relative `probes_path` and
`details_path`, their SHA-256 and byte counts. No large sidecar belongs in Git.
`result.json` appears only after all planned work and checks complete.
`error.json` preserves any failed attempt. The output cap is 1.5 GiB per arm.

`details.pt` is a CPU tensor dictionary with `probes`, `clean_trace`,
`donor_trace`, `source_vjps`, `observations`, `lm0_controls`, `parity`,
`directions`, and `analysis`. `observations[site]` has 18 records keyed by
`cycle` and signed `alpha`; each contains paired `zero` and `edited` traces.
Every trace has selected prepatch source rows, full projector output, full
post-replacement LM input, first-answer logits, terminal states after LM
blocks 0/7/14/22/29, 375 audio-placeholder indices, native per-coordinate
ULP spacing, stage/source hashes and realized edit metrics. All cycle-0
traces also contain full prepatch and postpatch encoder and projector source
tensors. Full source hashes are recorded on the remaining cycles; equality
there is a model-forward provenance assertion, while cycle-0 full tensors
permit independent bitwise equality checks. `source_vjps` contains 44 FP32
selected-row VJP tensors for each source site. `lm0_controls` contains the 24
fixed matched cells (zero/edit), one self control and the fixed unit vector.
The six parity prompt states include the complete `[1,396,3072]` state and
first-answer logits; the six greedy suffixes are also recorded.

The prospective tensor-payload estimate is **711,725,056 bytes BF16** and
**1,407,672,320 bytes FP32**. It includes 72 paired traces, 25 full-source
captures (24 cycle-0 forwards and one clean graph), 44 VJPs, probes, first
answer logits, per-coordinate ULP arrays, compact LM0 controls and complete
parity states. The FP32 estimate leaves 202,940,416 bytes below the hard
1,610,612,736-byte cap for metadata and serialization overhead. The runner
checks this estimate before model load and enforces the cap while writing;
it preserves a failed attempt if exact serialization exceeds the cap.

The success status is `completed_old_fit_handoff_stage_a_native_only`.
`result.json` binds `meta.json`, `model_layout.json`, progress, probes and
details by SHA-256, plus the protocol/source hashes, planned and artifact
derived counts, parity outcome, resource receipts and stage diagnostics.
The independent CPU auditor owns a separate audit receipt; Stage B requires
that audit before it can read this attempt. These data are a one-case
numerical diagnosis, not evidence of a security vulnerability.
