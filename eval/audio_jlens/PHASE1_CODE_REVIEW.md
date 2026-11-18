# Independent static review: FP32 phase-1 fit runner

Reviewed 2026-11-18 before any phase-1 Qwen run. This review did not run MPS, Qwen inference, development audio, or locked-test audio.

## Decision

**No blocking code or protocol discrepancy found for the fit-only run.** The runner does not grant development access merely because eight cases finish: it applies the frozen fit stop rule and records `development_permitted_by_fit_gate`. A fit pass is still an empirical question.

## Provenance and checks

- Frozen plan commit `b171278b51e3ee577a31b83161a9b9b2dcc502ed`, SHA-256 `c5cc77a37ebaa69c0d231c83cd19e0956c7a5c8ab87e1aae067f3812b6a5b34b`, matches the runner's bound hash. That commit made the off-word preference precise: one attention chunk disjoint from **all** word-row chunks. `controls_rows` implements that rule and chooses the lowest-start eligible window.
- Reviewed runner SHA-256 `f1ae3e697221dc68ef21df847304b38adb1b303a7c65ddf336e2f2e3050cf113` and CPU test SHA-256 `d64b20c51a32693bae59819d0812e7d09578ae6ee633d1c298dceaf818d0b844`; both match `PHASE1_IMPLEMENTATION.md` at review time.
- Pinned Transformers commit, exact model revision, fit manifest, four fit WAV hashes, plan hash, FP32 dtype, and tokenizer answer/EOS IDs are checked at load. The script reads only the Piper/Kokoro red/blue fit pair. It runs two directions at connector and encoder31; encoder15 is absent as intended.
- `py_compile` passed. The CPU toy suite passed **6/6** tests, including candidate-plus-EOS score orientation, row-contribution direction, control construction, off-word selection, progress hash equality, norm-baseline formula, and fit stop-rule counts. These tests do not establish full-model behavior.

## Protocol match

- Teacher forcing appends all candidate IDs except its final EOS to the prompt, and scores logits beginning at `prompt_len - 1`. The candidate includes the word token and EOS. Both candidate scores use the same recipient audio batch. The hooked audio state is a detached FP32 leaf for the VJP, and the donor-minus-recipient gradient predicts the donor-oriented contrast. The runner uses the no-grad unpatched contrast for no-grad patched measurements and records the grad/no-grad intercept gap.
- Source and donor tensor shapes, source anchors from the two candidate forwards, row maps, and cached versus uncached prefill states are checked. Effective edits are computed **after** FP32 assignment. The runner records each word-row contribution and its sum, measured delta, edit norm, generated suffix, EOS, and generation hook count at every planned alpha. Alpha zero checks zero effective edit, score parity, and identical token IDs; three additional zero-score repeats set the numerical floor.
- Clean generated answers must be exact red/blue words after trim and case folding, with EOS, in all four fit clips. The fixed alpha grid and seven named controls run on every eligible directional site case. Off-word and zero-off-word-delta controls have explicit ineligible records. Random seeds and the row permutation seed are fixed; natural and scaled off-word norms are recorded.
- The fit gate requires all four clean clips, four complete alpha-one donor answers **per site**, and at least 75% sign agreement among determinate small-step effects. The floor is `max(1e-4, 5 × max repeated-zero departure)` across the four routes at each site. An empty determinate set fails. The per-site fit-only norm coefficient uses exactly four routes × three small alphas and is saved with the fit progress SHA.
- MPS allocation is capped through the allocator fraction and checked against 30 GiB in a memory sidecar after load and each named VJP, prefill, zero repeat, alpha, and control. Existing result/sidecar paths are refused. The completed JSONL is compared to all eight in-memory results before its SHA is bound into the report and baseline sidecar.

## Limits to preserve in reporting

- The runner is **fit only**. Its status cannot establish the development or locked-test locality gates, a reusable Jacobian Lens, or a security finding.
- `score_with_patch` asserts exactly one hook call, but the saved score records do not separately expose that count; generated answers expose their generation hook count. This is a small observability gap relative to the plan's broad “hook call count” wording, not a scoring-path ambiguity.
- The progress JSONL is an audit trail, not a resume mechanism. If a run aborts, the no-overwrite guard requires a new output stem or deliberate handling of the partial files. The final status is written only after all eight cases or after a clean-answer failure.
- Row windows are nominal time indices, not isolated word features. Post-encoder attention can spread word information outside those rows; the report records outside-word state differences, and any control that flips an answer must constrain a word-local explanation.
