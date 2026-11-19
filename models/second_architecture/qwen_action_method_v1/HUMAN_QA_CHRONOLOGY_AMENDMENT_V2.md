# Qwen action listening QA V2: clock and coordinator binding

19 November 2026. **Source-only correction; human reviews remain 0/18.** The
independent [V1 HOLD](INDEPENDENT_HUMAN_QA_SOURCE_REVIEW_HOLD_V1.md) found that
the first-review and sparse third-review timestamps could be set in 2099 and
still pass. The original V1 source, its tests, and the HOLD memo remain
unchanged. Use [`human_qa_v2.py`](human_qa_v2.py) for any future actual
reviews; V1 cannot issue an accepted receipt.

V2 freezes a **300-second future-skew cap** relative to the auditor's UTC
clock. It applies to each first export lock, the sparse third bundle's creation
time, the third export lock, and the coordinator statement time. The lower
bounds also remain: first locks must follow the audited source render; a third
bundle must follow both first locks; its review must follow bundle creation;
the coordinator statement must follow both first locks. Timestamps and local
file mtimes are plausibility evidence, not cryptographic proof that a person
listened at that time. A bad local clock causes HOLD until independently
diagnosed; do not widen the cap after viewing model results.

V2 also requires a **separate coordinator attestation** for the final QA
report. After both reviewers export their blind JSON, create an unsigned
template with exact export hashes, reviewer IDs and bundle mapping hash:

```sh
eval/tts/.venv/bin/python -m models.second_architecture.qwen_action_method_v1.human_qa_v2 \
  --review-a /absolute/path/review_A.json --review-b /absolute/path/review_B.json \
  --make-attestation-template --output /absolute/new/coordinator_template.json
```

The coordinator must fill `coordinator_id`, set the three factual statements
`independent_people`, `separate_blind_folders` and
`no_shared_answers_before_both_locks` to `true`, add a nonempty note, and
write the current UTC `signed_utc` in ISO form ending `Z`. An unsigned or
altered binding fails. A coordinator may also be reviewer A, but the two
reviewer IDs must represent two different people. The software verifies the
statement's structure, hashes and chronology; it cannot establish a person's
identity or the truth of a self-report. An independent reviewer must inspect
actual reviewer and coordinator records before any later model admission.

`--inspect-first` remains a read-only way to see whether a sparse third review
is required. `--make-third` is allowed only for flagged cases, and the third
reviewer must have a distinct ID. A third record documents disagreement but
does not rescue a failed first pair. The final audit requires `--attestation`
along with both first exports, optional required third export, and a new
absolute `--output`. It retains the full 18-case denominator and sets
`candidate_model_inference_authorized: false` even when every human source
passes. No real export or attestation has been produced by this amendment.

Run no-weight checks:

```sh
eval/tts/.venv/bin/python -m unittest models.second_architecture.qwen_action_method_v1.test_human_qa_v1 models.second_architecture.qwen_action_method_v1.test_human_qa_v2 -q
eval/tts/.venv/bin/python -m py_compile models/second_architecture/qwen_action_method_v1/human_qa_v2.py
```

The six V2 synthetic tests cover the exact 2099 first/third counterexamples,
the 300-second boundary, changed binding, required coordinator claims and a
non-authorizing final receipt. These tests supply synthetic dictionaries only;
they do not stand in for actual listening.
