# Independent V5 four-context launcher source rereview (V2) — HOLD

18 November 2026. This is an independent, source-only rereview of the corrected
Voxtral non-beaver FP32 launcher. It preserves the immutable V1 HOLD review
`a609b49803c21a9cfd58b91fb718a10b5d92270a`.

**Decision: HOLD for the combined runner and launch-gate source. No campaign
reservation or model launch is authorized.**

## Exact reviewed source

| Item | SHA-256 |
| --- | --- |
| Launch gate, `voxtral_fp32_nonbeaver_linearity_launch_gate_v1.py` | `c99673b39faf434519195ab0757a67c3079dd1f26975519beb4f1a7ead70748d` |
| Gate tests, `test_voxtral_fp32_nonbeaver_linearity_launch_gate_v1.py` | `fc203d69ab24e70e46034ad4aa98aac3f39cde170ec35ccea879ff1f98f9a1b9` |
| Runner, `voxtral_fp32_nonbeaver_linearity_runner_v1.py` | `91e622f7fd8f4d9dac00ca7a0baaf02d74faf9326109d10674ebf5673310bc04` |
| Runner tests | `a4d9600fcd0371f1c0cf11a6ea6133d9f5ee7d097d126ca6fe685e16a3749463` |
| V5 prospective protocol | `99de95003190de6b5fb55c5ba8bed446dfea7cdd8cf53c0569d304bcb942b49b` |
| V5 CPU preflight | `5ef97e47e66c37b6c8ecb64b366daf4a713cfea1dfdee2f7f59ca75324b08401` |
| V5 byte budget | `53d19e065669cce63256f39c3b1112bdbbdb52691f2fe5ae475f865ad6191a7f` |
| V5 byte-budget probe | `f666e1df888e7bd8c87b1772b8031bfbf3c819345b607ed3e165ddaf979666ec` |
| Saved-data auditor | `70a1ac3c4e4da92bb1fa1034bd866195024dcd423ea35a2386e233bce5dca8e2` |
| Auditor tests | `3ffaa2224ca0c29ed5a49a985cf58b230aaaca9faa539214fa26548ca978c15a` |

The corrected gate and tests were committed in `5703ac49afa4e73d2382dbc304a16b5ed0dbe2d7`;
the subsequent gate binding of the auditor V2 rereview was committed in
`6d275de5ff8f4a4cf5d92b8f8ac36f2e68b17915`. The distinct independent
auditor V2 GO is `a142526f2f089786fd05e1264b1a7bb2ca4acb73`.

## Checks completed

- All 14 gate CPU tests passed in the pinned `models/second_architecture/.venv`.
  The read-only `--check-freeze` stopped at the absent combined source GO file,
  before any reservation or model load.
- The earlier V1 HOLD mutations are rejected: a minimal prior finish without
  start/claim/result/audit evidence, a changed parent-measured elapsed time,
  a negative prior-work reservation, and a reservation dated before the freeze.
- The source binds the four ordered Piper/Kokoro silver/teal contexts, exact
  selected WAV hashes, processor batches, masks, basis, V5 protocol, CPU
  preflight and byte-budget receipts. It has a 300-second parent worker
  watchdog, 30-GiB MPS limit, 2-GiB per-context output cap, 8-GiB campaign
  output cap, 12-GiB free-space reservation prerequisite, and 120-second
  auditor watchdog. It retains previous audited model-work time and the
  all-planned denominator. These checks do not clear the defect below.

## Blocking defect

`_check_prior_entries` still accepts a purported successful prior context
without the saved evidence that the independent auditor must read. A direct
temporary-directory mutation made a complete-looking index-0 start, claim,
finish, worker streams, and self-consistent `result.json`/`audit.json` hashes.
It omitted the tensor sidecar, `meta.json`, and `progress.jsonl` entirely.
The result and audit JSON held only the small subset of fields checked by the
gate. Nevertheless `_check_prior_entries(1, reservation)` returned `8.0`,
admitting context 1. The committed
`test_completed_prior_requires_claim_and_matching_result_audit` fixture
also accepts this same sparse success shape.

This is material to a four-context one-shot campaign: a forged success can
advance the next context without any independently auditable model result.
For a `FINISH_OK` prior entry, require the full saved-data file set and
validate the prior audit against it, or run the independent saved-data
verification read-only before allowing the next context. Bind its complete
result/audit schema, sidecar and supporting file hashes, and fail closed on
missing or mutated files. A genuinely failed prior attempt should still
advance when its failure receipt and elapsed work are corroborated; it must
remain in the planned denominator and cannot be retried.

This HOLD concerns source admission only. It makes no claim about model
behavior, linearity, or an observed live attack. No model weights or live
four-context campaign were opened or run for this rereview.
