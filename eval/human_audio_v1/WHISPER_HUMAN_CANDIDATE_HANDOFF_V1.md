# Authentic-speech Whisper candidate runner V1 — source-only handoff

**HOLD. No target inference has been run by this candidate.** This code is a
development-split candidate for the frozen prospective plan. The A/B blinded
listener exports, any required C export, the final intake receipt, a reviewed
admission, and a separate executable GO are not present. The candidate cannot
run without them. It does not open the sealed-test split.

## What the code binds

`whisper_human_candidate_gate_v1.py` derives the admission from the frozen
16-WAV plan and `whisper_human_preflight_v1.py`. The latter rechecks the source
WAV hashes and geometry, proxy evidence, pinned model and processor files,
installed library versions, and authentic human exports through the V2 intake.
The admission additionally binds exact source hashes, the fixed development
attempt path, decode settings, limits, and call census. No weight is loaded by
admission. The later GO gate checks committed, clean bytes and enforces this
order: source commit, admission commit, independent review commit, GO commit.
It compares every bound byte with the snapshot in the GO commit. The GO is for
one attempt only; an existing attempt directory blocks reuse.

The runner records eight development calls, two base repeats, all four
development pair decisions, whole-mel differences, and fresh unpatched replay
results for every pair that meets the earlier eligibility criteria. For each
eligible pair it executes both directions, two sites, and the five frozen arms.
Each call records full output token path, prefix and detected language, EOS
status, first-step raw and processed vectors, source WAV provenance, mel/mask
sidecars, encoder hook counts, and, for a patch, native and edited generation
site states. The run ledger retains failed screens and ineligible pairs. It
reports control-qualified cells only after comparing the complete token paths
and the full-donor first raw vector. All output remains pending a separate
saved-evidence audit.

## Source review required before any GO

An independent reviewer must inspect the runner and tests, confirm the pinned
Transformers 5.20.2 hook topology and feature extractor attributes against the
local source, check the conservative 106-call / 1 GiB / 4 GiB / 7200-second
caps, and verify that the native/edited sidecars satisfy the protocol. The
reviewer must also validate the genuine A/B (and, if needed, C) listening
exports and the final intake receipt. Record an explicit review receipt with
the schema and fields checked in `validate_go`; commit it after the admission.
The independently authorized GO has a separate schema and commit. Do not
derive either receipt from a model outcome. In particular, the current source
commit is not an independent review or an authorization to run.

The target path is `python -m eval.human_audio_v1.whisper_human_candidate_runner_v1
run`. This command checks GO **before** `_target`, where model loading occurs.
The source-only admission command requires actual reviewer paths. Running it
with absent or fake exports fails the V2 intake. The tests use mocks only to
check gate behavior; they contain no listener judgments and invoke no model.

## Limits of this candidate

This version covers development only. A sealed-test run needs a new frozen
method and code version, an independent source review and a new GO. The
runner's code has not been validated against a target forward pass in this
state; a reviewer must resolve any source-level or no-model test issue before
GO. An eligible pair count of zero yields no causal conclusion. Any model or
resource failure leaves a partial attempt, which must be audited as invalid
and cannot be retried under the same GO.
