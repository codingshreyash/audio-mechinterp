# Voxtral v2 FIT pilot runner handoff

**Implementation ready; no v2 model inference.** [Runner](voxtral_fit_pilot_capability_v2.py) SHA-256 `7ac333dadd4f645c4b173fd23a56fcee4bf45728abfaaabc91b87d3207d068a3` is bound to [selection](VOXTRAL_FIT_V2_SELECTION.json) SHA-256 `f897d360dbcec9c09f1813f9c070294753fbabc25c6213ba6bdac34b1d79b56b`, source index SHA-256 `bc739cf1f0d265426cc760ddd6a42223554b1097cab6ba87c2c8b81129d1b971`, and reviewed [v1 runner](voxtral_fit_pilot_capability_v1.py) SHA-256 `1d4d65a528d558e5bdda3b733eaab6e7e849ab9456d6842d5ee178d3efdfb8d8`. It reuses v1's byte-hash checks for local checkpoint shards and assets, processor, BF16 MPS generation, strict complete-answer/EOS classifier, canonical initial-capital sequence score, 30 GiB cap, and immutable output/progress writer. It also rehashes the actual corpus manifest and selected 32 fit WAVs before MPS. No development/test WAV is opened.

CPU validation passed:

```bash
python3 -m unittest models/second_architecture/test_voxtral_fit_pilot_capability_v2.py
models/second_architecture/.venv/bin/python models/second_architecture/voxtral_fit_pilot_capability_v2.py --validate-only
```

The five tests cover exactly 32 unopened-by-Voxtral fit clips, rejection of old pilot/nonfit and changed-WAV selection, and independent capability/Stage-2 gating. Processor preflight found 16 pair records, matched prompt token IDs per pair, `[1,128,3000]` audio features, and 375 audio placeholders per clip. `--validate-only` loads no checkpoint weights and does not allocate MPS.

After independent code review and coordinated MPS availability, the explicit run command is:

```bash
models/second_architecture/.venv/bin/python models/second_architecture/voxtral_fit_pilot_capability_v2.py --run
```

The v2 run creates a fresh directory under `runs/voxtral_corpus_v1_fit_pilot_v2/`. The inherited `result.json` reports the original preregistered **≥24/32 overall and ≥12/16 per voice** capability gate over all 32 clips; a separate immutable `v2_gate.json` computes the **16/16 exact natural endpoints** in the lexicographically first fit template for each pair/voice. Interventions are permitted only if **both** gates pass. An observed semantically plausible alternate spelling still fails strict exact matching. Failed or partial runs keep append-only progress and are not silently resumed. This is a fit-informed v2 after the v1 `deer`/`dear` ambiguity, not a retroactive v1 success or a held-out result.
