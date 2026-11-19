# Attempt 002 handoff: non-speech Whisper causal study

19 November 2026. **HOLD for independent executable review; no attempt-002 candidate model forward has run.** Attempt 001 is immutable. It failed before model load because the worker rejected the two planned repeat-silence case records in the valid descriptive V2 ledger. The [prospective repair](WHISPER_NONSPEECH_CAUSAL_ATTEMPT_002_REPAIR_PROSPECTIVE_V1.md) was frozen before this V2 worker was written.

The [V2 worker](causal_worker_v2.py) parses all 42 SHA-256-pinned descriptive rows before model load. It accepts exactly 20 ordered unique cases and two ordered silence repeats, checks the repeat outputs and WAV/PCM hashes against their originals, and returns only the 20 unique outputs as baselines. The 328-call causal schedule SHA-256 remains `a4a77079c9381758720712f7b5420186e0b08a18c8f3fbbf1a99beb4926d087b`: four donor calls, 180 final-encoder calls, and 144 early-encoder calls over all 18 non-silence inputs. The [V2 parent](causal_parent_v2.py) fixes the one-shot path at `runs/causal_whisper_v1/attempt_002`. The [V2 saved auditor](causal_audit_v2.py) revalidates the full descriptive ledger, 328 outputs, donor state provenance, raw-logit parity, EOS/censoring, edit reconstruction, and host/resource evidence.

Before any candidate load, independently review every V2 source file, this handoff, the repair amendment, the unchanged V1 design and planner, the actual-42-record and fake-model tests, and the no-weight admission. A design review or the attempt-001 executable GO is insufficient. The executable reviewer must commit a distinct V2 GO after the V2 admission, binding its exact SHA-256 and source hashes. The worker and parent require clean committed fixed-path admission and GO files, with GO ancestry later than each source and admission. Any unresolved issue is HOLD.

From repository root, with `models/second_architecture/.venv/bin/python`:

```sh
models/second_architecture/.venv/bin/python -m unittest models.second_architecture.nonspeech_probe_v1.test_causal_plan_v1 models.second_architecture.nonspeech_probe_v1.test_causal_worker_v2 models.second_architecture.nonspeech_probe_v1.test_causal_parent_v2 -v
models/second_architecture/.venv/bin/python -m models.second_architecture.nonspeech_probe_v1.causal_worker_v2 admit --output models/second_architecture/nonspeech_probe_v1/CAUSAL_WHISPER_ADMISSION_V2.json
```

Commit that admission. After independent executable review, the reviewer commits `CAUSAL_WHISPER_EXECUTABLE_GO_V2.json` with schema `nonspeech_whisper_causal_executable_go_v2`, status `GO`, exact `admission_sha256`, and exact `sources` from the admission. Only then can the fixed one-shot attempt launch:

```sh
models/second_architecture/.venv/bin/python -m models.second_architecture.nonspeech_probe_v1.causal_parent_v2 --admission models/second_architecture/nonspeech_probe_v1/CAUSAL_WHISPER_ADMISSION_V2.json --go models/second_architecture/nonspeech_probe_v1/CAUSAL_WHISPER_EXECUTABLE_GO_V2.json
```

The 4-GiB sampled process-tree RSS, 2-GiB evidence, 1,800-second worker wall, 128-MiB worker evidence reserve, and audited AC/open-lid/no-sleep host gates are unchanged. A failed or interrupted attempt 002 remains immutable. After a clean completion, run the V2 saved-data auditor outside the attempt directory, then obtain an independent result review before causal interpretation:

```sh
models/second_architecture/.venv/bin/python -m models.second_architecture.nonspeech_probe_v1.causal_audit_v2 --attempt models/second_architecture/nonspeech_probe_v1/runs/causal_whisper_v1/attempt_002 --admission models/second_architecture/nonspeech_probe_v1/CAUSAL_WHISPER_ADMISSION_V2.json --go models/second_architecture/nonspeech_probe_v1/CAUSAL_WHISPER_EXECUTABLE_GO_V2.json --output models/second_architecture/nonspeech_probe_v1/runs/causal_whisper_v1/attempt_002_audit.json
```

Even a complete clean intervention is a white-box causal result on already inspected engineered sounds. It does not itself establish an attacker-controlled audio exploit or assistant action.
