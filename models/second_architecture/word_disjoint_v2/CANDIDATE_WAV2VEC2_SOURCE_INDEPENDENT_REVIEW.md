# Independent source review: Track B Wav2Vec2 candidate assay

19 November 2026. **Decision: GO for the exact Wav2Vec2 runner and saved-data auditor source bound below; HOLD for candidate inference until the shared QA admission review, real 32/32 human listening receipt, and separately committed hash-bound preflight/freeze pass.** I reviewed source commits `b604ebd`, `9b79ae9`, and the narrow freeze-binding change `cdcc42a` against the [frozen V2 plan](../WORD_DISJOINT_TIME_STATE_VALIDATION_PLAN_V2.md), prior reviewed FIT and exploratory development Wav2Vec2 runners/auditors, and the technically audited 32-WAV fixture. I loaded no candidate weights, opened no locked-test WAVs, and ran no candidate inference. This review does not approve the independent human QA records or a model result.

## Source and no-weight checks

- `prepare` enumerates **32 fixed WAVs, 16 pair/engine cells, 64 directed S/C cells, two pinned sites and six arms**, or **768 patched forwards plus 64 clean forwards**. It uses the first feature-projection tuple tensor and first tensor after encoder block 11, with the fixed PCM-receptive-field window `[ceil((s−399)/320), floor((e−1)/320)+1)`. It rejects an invalid window or unequal pair geometry before patching. The recipient waveform and all unedited rows remain fixed.
- Clean capture hooks require exactly one tuple output at each site. The reused, previously reviewed patch helper checks the live recipient activation against the saved clean state before replacing the selected tensor, removes its hook in `finally`, and requires one invocation. The six arms are self, donor word W, donor outside O, full donor, separately seeded random W and random O. The random edit helper draws over the whole selected region, concatenating O left then right, and scales to the corresponding natural donor-minus-recipient L2. Seeds `26048191 + 4·direction + 2·site + arm_offset` are distinct across all random arms.
- The runner saves **complete** clean and patched CTC argmax ID sequences and decoded transcripts. Strict intended-word eligibility requires both complete clean transcripts to equal their full fixed reference token lists; all planned cells run even if clean ASR is wrong. Self and full compare complete logit tensors exactly with the recipient and donor baselines. Control failures stay in the records.
- The independent saved-data auditor requires the full **834-row** order (preflight, 64 clean, 768 patched, complete), rehashes every expected sidecar and rejects extras, reconstructs S/C inputs, verifies C-mode outside-word input equality, reconstructs all six patches from saved clean states and seeds, compares patch hashes and norms, re-decodes every complete CTC output, and rechecks self/full exact logits. It uses the pinned processor but loads no model weights. It can report a valid saved-data audit with `mechanism_gate_pass=false` if a self/full control failed, preserving that failure for interpretation.
- The no-weight byte ledger is **217,901,056 bytes**, including clean input/two site states/logits, 768 patched logit arrays, `.npy` headers, 32 MiB metadata and 64 MiB contingency, below 512 MiB. The runner keeps no patched-state sidecars. CPU float32 eval, four Torch threads, 30-minute, 4-GiB process RSS, 512-MiB output and forward headroom checks are in source. A failure appends a failure row and cannot satisfy the auditor's complete matrix.

I reran the nine no-weight candidate-source tests (`9/9` passed), including the fake two-site live-identity/hook-removal test and rejection of stale source GO bindings. `py_compile` passed for the Wav2Vec2 runner, auditor, and shared gate. The earlier `prepare` returned 64 clean/768 patched cells and the bound above with `admitted_for_candidate_inference=false`; the actual `preflight` failed closed because the real 32/32 QA receipt and independent review are absent. No model output or source-selected successful subset was seen.

The `cdcc42a` diff adds source bindings to the preflight and freeze receipts. The shared gate requires this GO to contain one exact machine-readable binding for the runner, auditor, shared gate, FIT helper, and no-weight test source. It checks committed cleanliness, verifies every blob exists in this later independent review commit at the bound SHA-256, and requires each source commit to precede this review and the review to precede the final frozen receipt. The Wav2Vec2 model execution and saved-data auditor were unchanged; no result-dependent analysis or candidate inference was added. The current source hashes and last-change commits are:

```json
{
  "schema": "word_disjoint_source_go_v1",
  "decision": "GO",
  "source_bindings": {
    "runner": {"path": "models/second_architecture/word_disjoint_v2/run_wav2vec2.py", "sha256": "a14d13e29746b62c29d05c2d82538c29ee8151bce370e7a312626cc99886a2a5", "commit": "cdcc42a759059ccf58ddc56432ca71e83d29458a"},
    "auditor": {"path": "models/second_architecture/word_disjoint_v2/audit_wav2vec2.py", "sha256": "fc826aabf704c6eab9c72d6074fdc03045519035b7ef197ee474af8bbae512b3", "commit": "b604ebdaf50864c2838f0f37b35cb02e2f6c2bc8"},
    "common": {"path": "models/second_architecture/word_disjoint_v2/candidate_common.py", "sha256": "0ad9268ceae5bd54aa24edfbcf31d00844fd6ff1556ca58bbe778d23f4f70390", "commit": "cdcc42a759059ccf58ddc56432ca71e83d29458a"},
    "helper": {"path": "models/second_architecture/wav2vec2_fit_complement_patch_v1.py", "sha256": "79932d09efbadf784b53f28e744bb7397d3908e2da43c41f1dd0240d5c3f9202", "commit": "ec894107e1d0165ca66b9f23fb50478e17f678ec"},
    "test": {"path": "models/second_architecture/word_disjoint_v2/test_candidate_source.py", "sha256": "b360b23dd882f0f2f8cbb5bde925571fc340a581df5a2d82265b7715df25576a", "commit": "cdcc42a759059ccf58ddc56432ca71e83d29458a"}
  }
}
```

## Interpretation boundary

A source GO does not establish that the checkpoint recognizes the new words, that any internal patch transfers a word, or that an outside speaker can produce the same activation. S/C modes, opposite directions, sites and arms remain correlated. If the shared human-QA gate or a control fails, retain every row and report the failure. A speech-assistant or security claim would need a separate unmodified-model, attacker-reachable action or disclosure test.
