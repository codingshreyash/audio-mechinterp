# Independent V5 four-context launcher source review — HOLD

18 November 2026. This is an independent, source-only review of the committed
Voxtral non-beaver FP32 launch gate. **Decision: HOLD for the combined runner
and launch-gate source. No campaign reservation or model launch is authorized.**

## Exact reviewed source

- Gate: `voxtral_fp32_nonbeaver_linearity_launch_gate_v1.py`, SHA-256
  `6a33709943fa601c9939fb82bb6373a6099c1140173c8e69ac88db179360273a`.
- Gate tests: `test_voxtral_fp32_nonbeaver_linearity_launch_gate_v1.py`, SHA-256
  `032836412dfbc62f06f0f34020b347704e375d2261b1154db97533fd4b0f9d89`.
- Both were committed in `9e49eb78bc3d0bcda346f70ad4c9fb11b10ed926`.
- Runner: SHA-256
  `91e622f7fd8f4d9dac00ca7a0baaf02d74faf9326109d10674ebf5673310bc04`;
  its separate runner-only source review remains distinct from this combined gate review.
- V5 protocol SHA-256
  `99de95003190de6b5fb55c5ba8bed446dfea7cdd8cf53c0569d304bcb942b49b`;
  V5 conceptual review SHA-256
  `e5e706d8536754209c79eb4f458e8f7ecc90527e84a1b4198e0a0895fc13d6a3`.

All 11 gate CPU tests passed when run from the pinned `models/second_architecture/.venv`
interpreter and module directory. The real read-only `--check-freeze` failed
only at the absent combined source-review file, before any reservation or
model load. The committed source sets four ordered contexts, a 300-second
worker watchdog, a 30-GiB MPS runner limit, 2-GiB per-context and 8-GiB
campaign output caps, a 12-GiB free-space prerequisite, and a 120-second
CPU-audit deadline. These checks do not clear the defects below.

## Blocking defects

1. `_check_prior_entries` accepts a minimal earlier `launch_finish.json`
   containing only `status`, `context_index`, `attempt_id`, and
   `model_work_elapsed_seconds`. It does not verify the earlier finish's
   context label, freeze hash, start/claim chronology, result/audit hashes
   for a completed entry, or its prior/cumulative model-work fields. The
   existing `test_prior_entry_must_finish_and_later_entry_must_not_start`
   demonstrates the minimal accepted shape. An independent mutation placed
   a purported successful index-0 finish with elapsed time `0`, no start,
   claim, worker, result, or audit; `_check_prior_entries(1, ...)` returned
   `0.0` and would allow the next context's admission to proceed. This
   defeats the campaign's ordered evidence and cumulative-time provenance.
   Require a full, internally consistent prior receipt and mandatory
   start/claim/result/audit checks appropriate to successful versus failed
   entries before admitting index `i+1`. Add negative tests for forged or
   incomplete prior finishes and changed elapsed time.

2. `_reservation` verifies only the status, IDs, labels, and freeze hash.
   It does not compare the recorded `prior_model_work_seconds` to the
   independently recomputed upstream value, nor verify that its UTC time
   follows the reviewed freeze. An independent mutation wrote the otherwise
   valid reservation with `prior_model_work_seconds=-999999.0` and
   `wall_utc=2020-01-01T00:00:00+00:00`; `_reservation()` accepted both.
   The later `launch_context` uses a separate upstream value, so model-time
   enforcement may still work, but the reservation evidence is false and
   the campaign chronology is not verified on admission. Bind these fields
   to `verify_freeze()` and test both mutations before a source GO.

The external freeze schema does not explicitly include the 8-GiB campaign
and 12-GiB free-space limits. Their constants are still bound indirectly
by the frozen gate hash; include these fields in the final external freeze
if the launcher source is revised.

The separate saved-data auditor is under independent review. No model
weights, fresh-v2 corpus outputs, or live pilot result were opened here.
