# Qwen action speech listening QA: source-only intake V1

19 November 2026. **Human QA remains pending.** The real 18-WAV render and two
blind reviewer folders passed [independent technical review](INDEPENDENT_SPEECH_RENDER_RESULT_REVIEW_GO_V2.md),
but the bundle has zero human exports. This workflow does not load Qwen weights
or authorize candidate inference. It records what actual reviewers submit; its
synthetic tests do not count as listening.

The two existing forms are
`runs/source_render_v2/review_bundle_001/reviewer_A/REVIEW.html` and
`runs/source_render_v2/review_bundle_001/reviewer_B/REVIEW.html`. Each folder
contains only opaque `q01.wav`–`q18.wav` clips and its form. Give each reviewer
only one folder. Keep `TECHNICAL_MAPPING.json`, scripts, expected labels and the
other review from both people until **both complete JSON exports are locked**.
The form exports `review_A.json` or `review_B.json`, with one ordered row for
every clip. The auditor verifies the exact independent queue order, form
binding, source and copied WAV hashes, distinct reviewer IDs, complete rows,
UTC lock timestamps after the render, finite 0–16-second phrase times and
absence of unknown fields. The exact reviewed form hash is the available
blinding evidence. Neither a form hash nor an ID proves that two distinct
people listened independently; a coordinator must attest to that separately.

After validating **both** complete blind exports, the auditor reveals the
frozen source words and label internally. It compares a lowercased,
punctuation-stripped word sequence with the exact source sentence; it also
requires the intended blind meaning (`list_only`, `ask_before_change`, or
`confirm_now`), the November 19, 2026 date, the expected event (`Design Review`
or `none`), and `notes=none`. These strict rules can reject a sensible
alternative transcription, such as a digit spelling or a dialect spelling.
Such cases stay HOLD and require an independently reviewed protocol amendment;
the auditor does not silently interpret them as equivalent. Phrase timestamps
are retained as human annotations and still need a separate audio-to-row
alignment review before a local time-window intervention.

If first reviewers disagree about a critical meaning, date, event or exact
words, `--make-third` builds **only those clips** in a new fixed blind
`third_review_001` folder. Its binding hashes both locked first exports. The
third reviewer must be distinct and submit after the third bundle exists.
The third record documents the disagreement. A third opinion does not turn a
failed two-reviewer source into an eligible model input. If the first reviewers
agree, a third review is forbidden. Missing, incomplete, swapped, malformed or
early exports cause HOLD with no passing receipt.

Run from the repository root only after receiving actual review exports:

```sh
eval/tts/.venv/bin/python -m models.second_architecture.qwen_action_method_v1.human_qa_v1 \
  --review-a /absolute/path/review_A.json --review-b /absolute/path/review_B.json \
  --inspect-first
```

If `--inspect-first` names a disagreement, replace that flag with
`--make-third` to create the sparse third queue. If none exists,
`--make-third` refuses to create one. For a final
read-only audit, omit `--make-third`, provide a new absolute `--output`, and
include `--review-c /absolute/path/review_C.json` only when one was required.
The report records every one of the 18 cases, exact export hashes, raw reviewer
IDs and any third-review result. It sets `candidate_model_inference_authorized`
to **false even on a human QA pass**. A later independent reviewer must inspect
actual human exports, identities, form and source hashes, and the generated
report before a model-run admission may bind it.

Source-only verification:

```sh
eval/tts/.venv/bin/python -m unittest models.second_architecture.qwen_action_method_v1.test_human_qa_v1 -q
eval/tts/.venv/bin/python -m py_compile models/second_architecture/qwen_action_method_v1/human_qa_v1.py
```

The tests create synthetic review dictionaries in memory. No synthetic review
is written into the production bundle or used to pass the real 18/18 gate.
