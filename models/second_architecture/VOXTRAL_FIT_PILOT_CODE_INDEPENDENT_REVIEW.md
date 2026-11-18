# Voxtral corpus-v1 FIT pilot: independent code review

**Verdict:** No remaining static blocker before the bounded FIT-only MPS pilot. The final reviewed runner SHA-256 is `1d4d65a528d558e5bdda3b733eaab6e7e849ab9456d6842d5ee178d3efdfb8d8`. This review used only fit-pool metadata, the local processor, and CPU tests. It did not load Voxtral weights for inference or open development/locked-test WAVs.

The runner selects the lexicographically first preselected map-fit pair in each category (`animal_06`, `color_03`, `nature_00`, `object_02`), yielding exactly **4 pairs × 2 voices × 2 fit templates × 2 words = 32 clips** from 16 matched records. It rejects a non-fit row marked pilot. CPU validation confirms all 32 selected WAV hashes and format, `[1,128,3000]` feature tensors, 375 audio placeholders per clip, and equal prompt token IDs within each matched word pair. Five CPU tests and `--validate-only` passed.

Generation uses the frozen prompt, greedy decoding, `max_new_tokens=32`, and pinned EOS ID 2. Its primary count requires one exact expected whitespace word after surrounding whitespace removal/casefolding and a terminal EOS. The secondary score teacher-forces the one frozen initial-capital answer sequence **including EOS**, taking logits at `prompt_len−1+offset`, summing each token's full-vocabulary log probability, and reporting other-minus-expected. This is a single canonical surface, not a four-surface class probability. The gate keeps all 32 answers and requires at least 24/32 overall and 12/16 for each voice.

The intended MPS path sets a 30 GiB per-process memory fraction before model loading, samples allocation during and after the run, and writes a unique run directory with fsynced per-case progress, error preservation, run metadata, and a hash-bound final report. It intentionally **does not resume**; a failed run needs a new directory and retains partial evidence.

## Provenance and output corrections verified

- The runner now verifies the **literal pinned revision-tree SHA** before trusting its LFS hashes, then hashes both complete model shards. It also verifies the safetensors index, processor config, tokenizer asset, generation config, model config, frozen CPU index, and actual corpus manifest. Verified hashes are recorded in `run_meta.json`. A same-size mutation of any of those assets fails the CPU corruption test.
- Generated token IDs and the decoded string **with** special tokens are saved. Strict exact success rejects nonterminal EOS or other generated special IDs, so `skip_special_tokens=True` cannot silently hide them.

The corrected 6/6 CPU tests and `--validate-only` passed independently. Validation reported 32 clips, 16 pair records, `[1,128,3000]` features, 375 placeholders, and no model weights loaded. The current code has no obvious selection, teacher-forcing, hash-binding, or strict-output defect. This verdict licenses only the FIT feasibility run, not a held-out or security claim.
