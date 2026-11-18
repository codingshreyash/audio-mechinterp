# Independent review: Voxtral v2 FIT pilot

**18 November 2026; read-only audit.** I independently checked the saved [v2 run](runs/voxtral_corpus_v1_fit_pilot_v2/20261118T084011020385Z/result.json) and [gate artifact](runs/voxtral_corpus_v1_fit_pilot_v2/20261118T084011020385Z/v2_gate.json) against the frozen [v2 selection](VOXTRAL_FIT_V2_SELECTION.json), [draft protocol](VOXTRAL_FIT_V2_DRAFT.md), and pinned [runner](voxtral_fit_pilot_capability_v2.py). I loaded only the local **processor/tokenizer** to independently decode generated token IDs; I did not load model weights, run inference, or open development/test WAVs. The result JSON SHA-256 is `6cc84dc1a4ced9ce17380eab3836079c462c9974f31337f146416730b5536ccf`.

## Artifact and provenance checks

| Check | Independent result |
| --- | --- |
| Frozen code/selection | `run_meta.json` runner SHA-256 matches current v2 script `7ac333dadd4f645c4b173fd23a56fcee4bf45728abfaaabc91b87d3207d068a3`; reused v1 script matches `1d4d65a528d558e5bdda3b733eaab6e7e849ab9456d6842d5ee178d3efdfb8d8`; v2 selection matches `f897d360dbcec9c09f1813f9c070294753fbabc25c6213ba6bdac34b1d79b56b`. The recorded source index, source manifest, v2 draft, and requirements hashes match current files. |
| Checkpoint and source | Rehashed actual local revision tree, config, safetensors index, **both weight shards**, processor config, tokenizer asset, and generation config against the values recorded in run metadata. Rehashed installed Transformers Voxtral model/processor and Whisper frontend source files. All match. This checks current on-disk bytes against saved metadata; the runner's literal pins are the pre-run root of trust. |
| Every planned clip | All **32 unique** `(pair ID, engine, template, expected word)` keys exactly match the selection, eight clips each for `animal_08`, `color_05`, `nature_09`, and `object_05`. Every selected WAV's current SHA-256 matches the frozen selection; all result rows say `fit_pool`. The run did not include a v1 pilot, development, or locked-test clip. |
| Immutable progress | Parsed all 32 `progress.jsonl` lines; they equal the ordered 32 `result.json` records exactly. Saved run-meta/progress hashes match the files; `v2_gate.json` points to the exact result hash. |
| Token-level answer | Independently decoded every suffix with the pinned processor. Each generated sequence ends with a **single EOS ID 2**, contains no other special token, and decodes to the expected single word under the frozen case/whitespace rule. Raw decoded strings and decoded-with-special-token strings match stored values. Each prompt has 375 audio placeholders. |
| Canonical score | For each of 64 teacher-forced candidate entries, the saved complete-reply log likelihood equals the sum of its token log probabilities, and the token IDs equal the pinned tokenizer's initial-capital word sequence plus EOS. Every stored other-minus-expected margin recomputes correctly and is negative (range **−18.108 to −8.277**). These canonical scores are secondary; they are not a four-alias aggregate. |
| Resource cap | Observed maximum MPS driver allocation **9.935 GiB**, below the frozen **30 GiB** cap. The saved post-condition allocation samples are also below cap. |

## Frozen gates and comparison with v1

| Gate | Planned threshold | Audited v2 result |
| --- | ---: | ---: |
| Complete exact answers, all clips | ≥24/32 | **32/32** |
| Per synthetic voice | ≥12/16 each | **Piper Amy 16/16; Kokoro Bella 16/16** |
| First-template natural endpoints, all eight pair/voice records | 16/16 | **16/16** |

The v2 `intervention_gate_pass` flag is therefore arithmetically correct. The four categories each contribute eight clips; there is no selected-case exclusion. The earlier **v1** pilot remains **28/32 strict** and failed its separate Stage-2 gate at **14/16** because `deer` was output as `dear`/`Dear`, an orthographic homophone ambiguity. V2 deliberately selects fresh **fit-pool** words under a rule written after that outcome, so its clean pass is a **fit-informed repair**, not a retroactive v1 success or a held-out generalization.

## Interpretation

This establishes a clean natural-response precondition for the next **fit-only** feature and hidden-state interventions under a separately frozen implementation. It does not establish where Voxtral represents a word, that an activation patch will change a full answer, robustness to human voices or noise, or a security violation. The 12-pair development and 12-pair locked-test Voxtral candidate-answer pools remain unopened by this audit. **No blocking discrepancy found** in the saved v2 pilot or gate calculation.
