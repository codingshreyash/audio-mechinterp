# Executable handoff: non-speech Whisper phase and time interventions

19 November 2026. **HOLD for independent executable review. No causal candidate-model forward has run.** This source implements the independently reviewed prospective design, its pre-inference [V2 donor-state amendment](WHISPER_NONSPEECH_CAUSAL_DESIGN_AMENDMENT_V2.md), and its [V3 exact-generation-argument amendment](WHISPER_NONSPEECH_CAUSAL_DESIGN_AMENDMENT_V3.md). The exact machine-readable schedule is 328 calls: four true no-hook/read-only silence donor baselines, 180 Stage A final-encoder calls, and 144 Stage B early-encoder calls. All 18 non-silence inputs remain in the denominator. The schedule SHA-256 is `a4a77079c9381758720712f7b5420186e0b08a18c8f3fbbf1a99beb4926d087b`.

The [worker](causal_worker_v1.py) owns one pinned CPU FP32/eval checkpoint load, exact phase-tagged `detect_language` and transcription hooks, full native donor/recipient state sidecars from read-only calls, all 328 generated outputs, random edits, per-call tensor hashes, first-step raw logits, and full per-step raw float32 logits for the two true no-hook silence donors and 36 `donor_both` controls. The [parent](causal_parent_v1.py) owns a fixed exclusive `runs/causal_whisper_v1/attempt_001` directory, AC/open-lid/power logs, process-tree RSS/evidence/wall caps, and an exact durable call census after any worker failure or kill. The [saved auditor](causal_audit_v1.py) is read-only over the attempt and runs outside the candidate model; it rehashes files, recomputes output IDs and tokenizer text, source/donor edits and random seeds, full-donor raw-logit parity, parent host/RSS evidence, and censored calls. The worker's own parity flags never substitute for that saved audit or independent result review.

## Source and admission sequence

Run all commands from the repository root with `models/second_architecture/.venv/bin/python`. Before any model load, independently review the exact committed `causal_worker_v1.py`, `causal_parent_v1.py`, `causal_audit_v1.py`, `causal_plan_v1.py`, their tests, this handoff, the V1/V2/V3 prospective design/amendments, descriptive asset pins, and the audited host-gate source. Run the no-weight tests and fake-model phase/patch/censor tests. An unresolved source issue is **HOLD**, including a mismatch between `attention_mask` and the actual pinned V2 descriptive generation arguments, a missing donor read-only state, or incomplete raw-logit sidecar binding.

Only after source review, create a **new no-weight** admission with:

```sh
models/second_architecture/.venv/bin/python -m models.second_architecture.nonspeech_probe_v1.causal_worker_v1 admit --output models/second_architecture/nonspeech_probe_v1/CAUSAL_WHISPER_ADMISSION_V1.json
```

The admission hashes the committed study source, pinned model/processor/tokenizer assets, installed Transformers/Torch/NumPy source, valid V2 descriptive record bytes, and exact 328-call schedule; it does not deserialize weights. The independent executable reviewer then checks that admission, commits a distinct later GO JSON at `models/second_architecture/nonspeech_probe_v1/CAUSAL_WHISPER_EXECUTABLE_GO_V1.json` with schema `nonspeech_whisper_causal_executable_go_v1`, status `GO`, exact `admission_sha256` and exact `sources` from the admission. Both parent and worker recompute the admission and enforce GO ancestry. The design-only review does **not** authorize execution.

After and only after the independent executable GO, the fixed one-shot launch is:

```sh
models/second_architecture/.venv/bin/python -m models.second_architecture.nonspeech_probe_v1.causal_parent_v1 --admission models/second_architecture/nonspeech_probe_v1/CAUSAL_WHISPER_ADMISSION_V1.json --go models/second_architecture/nonspeech_probe_v1/CAUSAL_WHISPER_EXECUTABLE_GO_V1.json
```

The parent checks 1,800-second wall time, a sampled 4-GiB process-tree RSS ceiling, a 2-GiB evidence ceiling, and 120-second post-worker host-probe lag. Its sampled RSS cannot rule out a peak between polls. The worker additionally enforces a 128-MiB evidence reserve before forwards and writes. A failed or incomplete attempt is immutable and not retried under this source/GO. A parent kill still leaves an ordered attempted, completed and censored call census from `call_progress.jsonl`.

After a clean parent completion, run the separate saved-data auditor with an output path outside the attempt:

```sh
models/second_architecture/.venv/bin/python -m models.second_architecture.nonspeech_probe_v1.causal_audit_v1 --attempt models/second_architecture/nonspeech_probe_v1/runs/causal_whisper_v1/attempt_001 --admission models/second_architecture/nonspeech_probe_v1/CAUSAL_WHISPER_ADMISSION_V1.json --go models/second_architecture/nonspeech_probe_v1/CAUSAL_WHISPER_EXECUTABLE_GO_V1.json --output models/second_architecture/nonspeech_probe_v1/runs/causal_whisper_v1/attempt_001_audit.json
```

Then obtain an independent **result** audit over the saved tensors, full sequences, both phase interventions, and controls before any causal interpretation. A full-donor parity failure, changed self/read-only output, censored generation, or unresolved numerical disagreement remains a result, not a reason to rerun. Even a clean encoder-state intervention is a white-box causal result on already inspected engineered sounds; it does not demonstrate an attacker-controlled security exploit, held-out audio transfer, or an assistant action.
