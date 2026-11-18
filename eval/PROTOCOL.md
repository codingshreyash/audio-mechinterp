# Evaluation and evidence protocol (draft, frozen before candidate fitting)

This protocol separates three claims: **a signal is detectable**, **a readout is useful**, and **the signal causally drives behavior**. They require different evidence. Every experiment records failures as well as successes.

## Reproduction gate

- Pin repository commit, model checkpoint revision, lens artifact revision/hash, library versions, dataset revision, prompt/audio bytes, and random seeds.
- Run upstream tests or validation scripts when available. Rerun one published evaluation using its metric and exclusions. Compare local output with the reported value or qualitative pattern; document discrepancies without relabeling them as successes.
- For a pre-fitted lens, check model/lens identity. Matching hidden width is insufficient.

## Readout gate

- Define concept labels and source/target time positions before inspecting candidate readouts. Keep speakers, wording, and recordings in train and held-out splits disjoint where the task permits.
- Compare a candidate with at least raw logit lens, a suitable existing J-lens/J++ baseline, and a task-appropriate probe or audio SAE when available.
- Report the same metric for all methods: concept rank or retrieval on a frozen candidate set, per layer and per audio-time region, plus false positives on matched controls. Include non-speech and silence where relevant.
- Show the full distributions and per-item examples; a top-token visualization alone does not establish a useful or faithful readout.

## Causality gate

- First establish the unmodified full output on repeated runs, using the exact serving path to be claimed.
- Apply a candidate-specific intervention and record the complete generated speech/text/native action. Include self swap, irrelevant-time or layer swap, and random or norm-matched-direction controls.
- Test both addition and removal when possible. Check collateral effects on unrelated words, audio quality, and model behavior.
- Validate on held-out recordings and at least one independently generated or human-recorded audio source before a general claim.
- A changed rank, logit margin, or refusal score alone is not evidence that a complete decision changed.

## Novelty and security gate

- Search primary papers and code at the time of any novelty claim. State what the method adds beyond Anthropic J-lens, Audiolens, Audio Jacobian Lens, audio SAE, and published audio-LLM patching work.
- For a security claim, state the attacker-controlled input, protected property, output/action consequence, and model versus runtime/harness contribution. Keep tools inert until there is a justified end-to-end test.

This is a draft until the first baseline and candidate models are fixed. Subsequent changes must be versioned and explained; candidate results cannot silently change the held-out test.
