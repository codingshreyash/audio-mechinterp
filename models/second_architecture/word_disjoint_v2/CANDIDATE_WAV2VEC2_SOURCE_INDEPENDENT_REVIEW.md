# Independent source review: Track B Wav2Vec2 candidate assay

19 November 2026. **Decision: GO for the exact Wav2Vec2 runner and saved-data auditor source bound below; HOLD for candidate inference until the shared QA admission review, real 32/32 human listening receipt, and separately committed hash-bound preflight/freeze pass.** I reviewed source commits `b604ebd`, `9b79ae9`, the freeze-binding change `cdcc42a`, and the local-dependency closure changes `b82907d` and `9450849` against the [frozen V2 plan](../WORD_DISJOINT_TIME_STATE_VALIDATION_PLAN_V2.md), prior reviewed FIT and exploratory development Wav2Vec2 runners/auditors, and the technically audited 32-WAV fixture. I loaded no candidate weights, opened no locked-test WAVs, and ran no candidate inference. This review does not approve the independent human QA records or a model result.

## Source and no-weight checks

- `prepare` enumerates **32 fixed WAVs, 16 pair/engine cells, 64 directed S/C cells, two pinned sites and six arms**, or **768 patched forwards plus 64 clean forwards**. It uses the first feature-projection tuple tensor and first tensor after encoder block 11, with the fixed PCM-receptive-field window `[ceil((s−399)/320), floor((e−1)/320)+1)`. It rejects an invalid window or unequal pair geometry before patching. The recipient waveform and all unedited rows remain fixed.
- Clean capture hooks require exactly one tuple output at each site. The reused, previously reviewed patch helper checks the live recipient activation against the saved clean state before replacing the selected tensor, removes its hook in `finally`, and requires one invocation. The six arms are self, donor word W, donor outside O, full donor, separately seeded random W and random O. The random edit helper draws over the whole selected region, concatenating O left then right, and scales to the corresponding natural donor-minus-recipient L2. Seeds `26048191 + 4·direction + 2·site + arm_offset` are distinct across all random arms.
- The runner saves **complete** clean and patched CTC argmax ID sequences and decoded transcripts. Strict intended-word eligibility requires both complete clean transcripts to equal their full fixed reference token lists; all planned cells run even if clean ASR is wrong. Self and full compare complete logit tensors exactly with the recipient and donor baselines. Control failures stay in the records.
- The independent saved-data auditor requires the full **834-row** order (preflight, 64 clean, 768 patched, complete), rehashes every expected sidecar and rejects extras, reconstructs S/C inputs, verifies C-mode outside-word input equality, reconstructs all six patches from saved clean states and seeds, compares patch hashes and norms, re-decodes every complete CTC output, and rechecks self/full exact logits. It uses the pinned processor but loads no model weights. It can report a valid saved-data audit with `mechanism_gate_pass=false` if a self/full control failed, preserving that failure for interpretation.
- The no-weight byte ledger is **217,901,056 bytes**, including clean input/two site states/logits, 768 patched logit arrays, `.npy` headers, 32 MiB metadata and 64 MiB contingency, below 512 MiB. The runner keeps no patched-state sidecars. CPU float32 eval, four Torch threads, 30-minute, 4-GiB process RSS, 512-MiB output and forward headroom checks are in source. A failure appends a failure row and cannot satisfy the auditor's complete matrix.

I reran the twelve no-weight candidate-source tests (`12/12` passed), including the fake two-site live-identity/hook-removal test, relative-import closure tests, and rejection of stale source GO bindings after a transitive helper changes. `py_compile` passed for the Wav2Vec2 runner, auditor, and shared gate. The earlier `prepare` returned 64 clean/768 patched cells and the bound above with `admitted_for_candidate_inference=false`; the actual `preflight` failed closed because the real 32/32 QA receipt and independent review are absent. No model output or source-selected successful subset was seen.

The `cdcc42a` diff adds source bindings to the preflight and freeze receipts. The `b82907d` and `9450849` diffs add a bounded AST walk of repo-local imports, including relative imports and package initializers. The resulting Wav2Vec2 dependency closure has exactly 12 roles: the runner, auditor, shared gate, FIT helper, no-weight tests, and seven additional local sources. The shared gate requires this GO to contain one exact machine-readable binding for every role. It checks committed cleanliness, verifies every blob exists in this later independent review commit at the bound SHA-256, and requires each source commit to precede this review and the review to precede the final frozen receipt. The Wav2Vec2 model execution and saved-data auditor remain unchanged through these revisions; no result-dependent analysis or candidate inference was added. The current source hashes and last-change commits are:

```json
{
  "schema": "word_disjoint_source_go_v1",
  "decision": "GO",
  "source_bindings": {
    "auditor": {
      "path": "models/second_architecture/word_disjoint_v2/audit_wav2vec2.py",
      "sha256": "fc826aabf704c6eab9c72d6074fdc03045519035b7ef197ee474af8bbae512b3",
      "commit": "b604ebdaf50864c2838f0f37b35cb02e2f6c2bc8"
    },
    "common": {
      "path": "models/second_architecture/word_disjoint_v2/candidate_common.py",
      "sha256": "01e3ece95a8e3c1202369185061c29d6b5307761299df3f23115300164b18312",
      "commit": "9450849ed2695ea00de97f712812afa0b513d24d"
    },
    "helper": {
      "path": "models/second_architecture/wav2vec2_fit_complement_patch_v1.py",
      "sha256": "79932d09efbadf784b53f28e744bb7397d3908e2da43c41f1dd0240d5c3f9202",
      "commit": "ec894107e1d0165ca66b9f23fb50478e17f678ec"
    },
    "local:models/second_architecture/wav2vec2_dev_word_disjoint_audit_v1.py": {
      "path": "models/second_architecture/wav2vec2_dev_word_disjoint_audit_v1.py",
      "sha256": "d106d1c03f5bc6410d49ea5b05eeaa78cf7a45a92e84dbc91da4a3c1bd127141",
      "commit": "d255534366645200a14ec218a6cf6a89d86c0e44"
    },
    "local:models/second_architecture/wav2vec2_dev_word_disjoint_v1.py": {
      "path": "models/second_architecture/wav2vec2_dev_word_disjoint_v1.py",
      "sha256": "ea1f949aa039ff3d5cf96ca14932384f4f5197801976ab95b8bddacb66462f74",
      "commit": "da1e6b3cad3c905dfd123313ba3e5c6f0c15e113"
    },
    "local:models/second_architecture/wav2vec2_fit_baseline_stage_capture_v1.py": {
      "path": "models/second_architecture/wav2vec2_fit_baseline_stage_capture_v1.py",
      "sha256": "89497c216aea0437b8e5076dfdb76fa002c69800daf7a8ae231b8457b15486d2",
      "commit": "fe75bea4531faa5fe362bfa743dfeabdea6ab4fb"
    },
    "local:models/second_architecture/word_disjoint_v2/__init__.py": {
      "path": "models/second_architecture/word_disjoint_v2/__init__.py",
      "sha256": "46050500ba491a71951c429e0bd5e69eadb4136118974c5b8475fc04b43aee6c",
      "commit": "49c0f7dfecaeae691aa05ebca040a43ee6f41eec"
    },
    "local:models/second_architecture/word_disjoint_v2/audit.py": {
      "path": "models/second_architecture/word_disjoint_v2/audit.py",
      "sha256": "6411c2961ba31700f3c3e099124644c80c38e52c925c4ef02cb9d01173cd831f",
      "commit": "8c65d8ebdc18fde10e21774721bb65f4d790f4b1"
    },
    "local:models/second_architecture/word_disjoint_v2/queue.py": {
      "path": "models/second_architecture/word_disjoint_v2/queue.py",
      "sha256": "1744be6270352f86fb1990f9ef7223734514dd05a9ec14b717876c8b28d15e3c",
      "commit": "8c65d8ebdc18fde10e21774721bb65f4d790f4b1"
    },
    "local:models/second_architecture/word_disjoint_v2/render.py": {
      "path": "models/second_architecture/word_disjoint_v2/render.py",
      "sha256": "f6524f30f2655c9f62fdc4898be5e480c20c5e0ba84c5e19427ad437656d044c",
      "commit": "8c65d8ebdc18fde10e21774721bb65f4d790f4b1"
    },
    "runner": {
      "path": "models/second_architecture/word_disjoint_v2/run_wav2vec2.py",
      "sha256": "a14d13e29746b62c29d05c2d82538c29ee8151bce370e7a312626cc99886a2a5",
      "commit": "cdcc42a759059ccf58ddc56432ca71e83d29458a"
    },
    "test": {
      "path": "models/second_architecture/word_disjoint_v2/test_candidate_source.py",
      "sha256": "6a9961ce3c15e001900ea6e1b0d3502df26d72a8b6a779a9780dd44aca5f9ff9",
      "commit": "9450849ed2695ea00de97f712812afa0b513d24d"
    }
  }
}
```

## Interpretation boundary

A source GO does not establish that the checkpoint recognizes the new words, that any internal patch transfers a word, or that an outside speaker can produce the same activation. S/C modes, opposite directions, sites and arms remain correlated. If the shared human-QA gate or a control fails, retain every row and report the failure. A speech-assistant or security claim would need a separate unmodified-model, attacker-reachable action or disclosure test.
