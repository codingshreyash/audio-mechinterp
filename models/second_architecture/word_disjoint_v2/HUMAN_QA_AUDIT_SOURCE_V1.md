# Human listening audit, source stage

The 32 synthetic recordings passed a technical PCM audit. This module checks future blinded listening exports; it does not listen to audio or create human review records. The private case map and this audit report contain reference words and must stay out of the listener-facing directory.

Two independent reviewers each submit all 32 opaque clips through the listener form. Each export must retain its first response for every clip and bind the public queue SHA-256 and individual WAV SHA-256 values. A third independent reviewer may submit the same 32-clip form if the first two differ in their raw full transcript, raw middle-word field, or clarity for any clip. Normalization is used only for correctness scoring. The audit uses one fixed Unicode NFKC, casefold, punctuation/underscore-splitting parser. Each clip passes only if two independent reviewers report the exact complete script, exact middle word, and clear audio. If the initial reviewers disagree, the third may provide the second agreeing assessment. The audit retains all raw responses, including failed and unused third responses.

After real reviewer exports exist, run:

    python3 -m models.second_architecture.word_disjoint_v2.human_qa_audit --primary /private/path/reviewer_a.json --secondary /private/path/reviewer_b.json --tertiary /private/path/reviewer_c.json

Omit --tertiary only when the two initial reviewers agree on every normalized answer and clarity judgment. The default private report is rendered_v2/QA_HUMAN_AUDIT_V1.json; the command refuses to replace it. Its status is human_qa_32_of_32_pass only if all 32 pass. A HOLD report records all missing third reviews and unclear or incorrect clips. A separate independent human-evidence review and candidate-runner freeze are required before model inference.

Source-only tests use fabricated in-memory responses and create no QA receipt:

    python3 -m unittest models.second_architecture.word_disjoint_v2.test_human_qa_audit -v

The code can verify file integrity and distinct reviewer IDs. It cannot prove that a person listened, that IDs belong to different people, or that a reviewer was genuinely blinded. Those are procedural checks for the human reviewers.
