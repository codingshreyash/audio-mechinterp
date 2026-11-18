# Voxtral fit-pilot capability runner handoff

**Ready for review; no model inference run.** [Runner](voxtral_fit_pilot_capability_v1.py) is bound to the frozen [CPU index](voxtral_corpus_v1_index.json) SHA-256 `bc739cf1f0d265426cc760ddd6a42223554b1097cab6ba87c2c8b81129d1b971` and [protocol](VOXTRAL_CORPUS_V1_PROTOCOL.md). Current runner SHA-256: `1d4d65a528d558e5bdda3b733eaab6e7e849ab9456d6842d5ee178d3efdfb8d8`. It selects exactly 32 `fit_pool` clips from the four fixed pilot pair IDs and never opens a development or locked-test WAV. No model weights are loaded without `--run`.

CPU validation command:

```bash
models/second_architecture/.venv/bin/python models/second_architecture/voxtral_fit_pilot_capability_v1.py --validate-only
```

This passed: 32 clips, 16 pair records, `[1,128,3000]` features, 375 audio placeholders per clip, matched prompt token IDs for both words of each pair, and pinned snapshot/source hashes. The **six** [CPU structural tests](test_voxtral_fit_pilot_capability_v1.py) passed, covering fit-only selection, rejection of a mislabeled development row, strict full-answer/EOS classification, canonical sequence IDs, source-manifest binding, and detection of same-size corruption in both weight shards, their revision tree, the safetensors index, and processor/tokenizer/generation assets. The preflight rehashes the actual source manifest, the **literal-pinned revision-tree SHA-256** before trusting its LFS hashes, both ~9.4 GB weight shards, the **literal-pinned safetensors index SHA-256**, processor config and tokenizer against the frozen index, and generation config against a pinned SHA-256. The computed hashes enter `run_meta.json`.

After review and resource coordination, the explicit inference command is:

```bash
models/second_architecture/.venv/bin/python models/second_architecture/voxtral_fit_pilot_capability_v1.py --run
```

It loads the pinned BF16 Voxtral checkpoint once on MPS with a 30 GiB process cap. Every clip gets one greedy `max_new_tokens=32` complete generation; strict primary success requires the exact spoken word (case-insensitive, surrounding whitespace ignored), EOS, and **no other special token** in the generated suffix. The secondary score teacher-forces the **initial-capital canonical** token sequence plus EOS for both words and reports other-minus-expected log-likelihood. This is one frozen surface per word, not the four-surface logsumexp proposed for the later assay. Raw generated token IDs, decoded text with and without special tokens, prompt IDs, WAV hashes, candidate token log probabilities, timing, model/source/version hashes, and MPS allocation samples are saved. The run creates a unique directory under `runs/voxtral_corpus_v1_fit_pilot/`, writes `run_meta.json` once, appends and fsyncs each `progress.jsonl` record, then writes `result.json` once; failures get `error.json` and preserve progress. It refuses existing output directories and does **not** resume a partial run.

The fit-only gate is ≥24/32 exact complete answers and ≥12/16 for each voice. A passing gate licenses the next predeclared fit-only frontend/patch checks, not a held-out or security claim. A failed gate stays in the report. No current Voxtral development/test answer has been opened by this runner.
