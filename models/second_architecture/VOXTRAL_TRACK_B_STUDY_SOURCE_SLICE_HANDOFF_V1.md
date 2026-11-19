# Voxtral Track B study source slice handoff

This is a **source-only HOLD**, not an admitted 16-context run. `voxtral_fp32_track_b_study_runner_v1.py` has only a `--validate-only` CLI. It never loads a candidate checkpoint. The preflight still reports absent 32/32 human QA, measured V6 pilot, and independent post-pilot GO.

The slice freezes the 16 ordered directed contexts from the no-weight preflight, validates matched WAV/processor/alias identity, captures one natural projector state during each clean generation, and executes the seven generation arms and sequential singleton scores for an injected **one-context** model. Its call ledger checks exact labeled order, identity, durations between calls, and per-context count (7 generations; up to 72 no-grad and 12 grad alias forwards). The one-context return includes all complete suffixes, EOS/class/alias coverage, margins, signed local predictions, parity failures, selected gradient hashes, and raw/random switches. A fake CPU model exercises this path.

The following live pieces remain required before candidate inference:

1. Bind the final immutable V6 `pilot_measurement.json` and `calls.jsonl` hashes plus an external independent GO bound to that exact measurement and this runner SHA. Rehash QA, preflight, model/source, alias and maximum-geometry fields. The pilot worker has not yet frozen its final receipt schema.
2. Add a reviewed 16-context supervisor that loads the pinned FP32 checkpoint **only after** the above admission, captures both directions without extra forwards, persists each call before and after execution, writes immutable per-context records and gradient sidecars, and reports all 16 planned contexts including failures and missing slots. A natural donor baseline is the opposite direction's clean generation, not an eighth generation.
3. Enforce the 240-second/context and 3,840-second cumulative model-work caps, 30-GiB MPS driver cap, 2-GiB new-output and 12-GiB free-disk gates, AC/open-lid/ten-minute transition checks, UTC versus monotonic gap, and no retry. The in-memory `CallLedger` checks only **between** calls; a hung single forward requires a live external supervisor. It does not certify host state or total campaign resources.
4. Independently review the final live entry point and immutable result schema before any Track B model call. Do not credit this source slice as a completed causal experiment or security finding.

Local checks: fake CPU tests in `test_voxtral_fp32_track_b_study_runner_v1.py`; no-weight `--validate-only` prints a HOLD and the fixed 16-context/112-generation/1,152 no-grad/192 grad maximum ledger.
