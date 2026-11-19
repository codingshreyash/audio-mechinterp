# Independent source review: Qwen blind-listening QA intake V1

19 November 2026. **Decision: HOLD on accepting a human-QA pass receipt from `human_qa_v1.py`.** The already audited 18-clip reviewer forms remain suitable for collecting real blind listening records. This review used only synthetic test dictionaries; no human response exists here, no Qwen weights were loaded, and no model inference was run. The intake's own reports always set `candidate_model_inference_authorized: false`.

I reviewed committed source `79f6dc7` against the [real technically audited bundle](INDEPENDENT_SPEECH_RENDER_RESULT_REVIEW_GO_V2.md). `eval/tts/.venv/bin/python -m unittest models.second_architecture.qwen_action_method_v1.test_human_qa_v1 -q` passed **8/8** synthetic no-weight tests; `py_compile` passed. The real source/bundle read-only audit still reports 36 copied WAVs and two intact forms, with **zero human reviews**.

## Checks that passed

- Each first export must have the expected schema, queue, bundle binding, distinct syntactically valid reviewer ID, 18 complete rows in the exact opaque queue order, and finite 0–16-second phrase times. Missing/swapped rows, altered binding, duplicate JSON keys and nonfinite times are rejected. The intake re-runs the source and bundle byte audits before using actual review paths.
- Frozen scripts and labels are consulted only after both full first exports validate. The comparison records every case, preserves failed wording/intent/date/event/clarity reasons, and refuses to turn a partial or ambiguous review into a pass. It requires the two first reviewers to agree; a third review records disagreements but cannot retroactively clear a failed first pair.
- A sparse third bundle is built only for flagged cases, with hashes of both first exports, copied-WAV hashes, a fixed opaque queue, and a distinct third reviewer ID. An unneeded third review is rejected. These are source-level checks; I created no production third bundle.
- The form hashes, reviewer IDs and timestamps are recorded. They cannot prove that two distinct people listened independently or that a browser export was immutable. The handoff correctly requires separate coordinator attestation and a later independent review of actual human records.

## Chronology HOLD

`timestamp()` validates UTC-Z syntax but has **no current-time upper bound**. `_load_bound()` requires first exports to be after the source render; it does not reject an export locked far in the future. In a synthetic 18-case test, setting both otherwise complete first exports to `2099-01-01T00:00:00.000Z` left `first_pair_all_pass=True` with no third review required. An unchanged complete no-disagreement path could therefore receive `human_qa_pass_needs_independent_review` with impossible lock times. `validate_third_mapping()` likewise accepted a synthetically valid third bundle with `created_utc` in 2099, and the third-review validator checks only that the third export is after that creation time.

The altered bundle binding was rejected in the same synthetic test. The problem is specifically the missing **upper** chronology bound, not a failure of source/bundle hashes or review-row completeness. The source tests do not cover future first or third times.

Before a production QA receipt is accepted, version the intake so first lock times, third-bundle creation, and third-export lock are no later than an explicitly frozen small clock-skew allowance after the auditor's current UTC time. Retain the lower bounds against the terminal source receipt and prior review locks. Add no-weight tests for far-future first and third timestamps and for modified binding/queue bytes. Record file hashes and keep the coordinator's separate identity/blinding attestation requirement; clock plausibility alone cannot prove real independent listening. Preserve this HOLD and any actual human exports unchanged while correcting the software.
