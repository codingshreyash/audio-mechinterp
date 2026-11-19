# Executable handoff: FIT normalization causal study

19 November 2026. **HOLD: source for independent review; no pinned-model
candidate inference has run.** The earlier 80-forward plan and review are
superseded by the [88-forward amendment](WAV2VEC2_FIT_NOISE_NORMALIZATION_CAUSAL_88_FORWARD_AMENDMENT_V1.md)
(commit `2442cdf`, separately reviewed in `927c265`). This handoff freezes
the implementation entry points and the future review/admission sequence.

## Source and one-shot schedule

Run all commands as Python modules from the repository root with
`models/second_architecture/.venv/bin/python`. The no-weight execution
plan is `wav2vec2_fit_noise_norm_causal_88_plan_v1.py`; the worker,
launcher, host gate, and read-only saved-result auditor are the files
named `wav2vec2_fit_noise_norm_causal_{worker,launcher,host,audit}_v1.py`.
`test_wav2vec2_fit_noise_norm_causal_worker_v1.py` exercises all 88
calls with a randomly initialized fake CPU network and a complete fake
saved-result audit. It never loads the pinned Wav2Vec2 checkpoint.

The 88 attempted forwards are exactly 32 Stage I scored, 12 Stage I
no-hook stock, 12 Stage I read-only stock, 16 Stage II scored, eight
Stage II no-hook shared-input, and eight Stage II read-only captures.
Every unique clean, word-noise, and outside-noise stock waveform gets
its own no-hook/read-only pair. Clean stock controls are run once per
source case and shared only across that case's two noise contrasts.
The worker writes scored rows after each call and stock rows after each
no-hook/read-only control pair. On failure, it records the exact ordered
completed names, the attempted failed call if any, completed calls lacking
a row, and every later never-attempted call name. Each attempted call also
gets a durable `call_progress.jsonl` start entry before inference and a
finish entry after it succeeds. If the parent kills the worker, it uses
that file to distinguish the in-flight call from later censored calls.
It never retries. Its fixed
output directory is supplied by the launcher at
`runs/wav2vec2_fit_noise_norm_causal_v1/attempt_001/worker_output`.
The parent refuses an existing attempt directory, even if it failed.

## Admission and independent GO

After an independent reviewer approves the exact committed worker,
launcher, host gate, auditor, test and handoff sources, create a **new,
no-weight** admission receipt with:

```sh
models/second_architecture/.venv/bin/python -m models.second_architecture.wav2vec2_fit_noise_norm_causal_worker_v1 admit --output models/second_architecture/WAV2VEC2_FIT_NOISE_NORM_CAUSAL_ADMISSION_V1.json
```

`admit` hashes all committed study sources, the prior FIT input/result
bytes, the 88-forward plan, the local checkpoint, processor/tokenizer
assets and pinned modeling source. It does not construct or run the
model. Freeze that JSON and obtain a **later, independently committed**
GO JSON with schema `fit_norm_causal_executable_go_v1`, status `GO`,
`admission_sha256` equal to the exact admission file SHA-256, and
`source_binding` equal to the admission's complete source-binding object.
The worker and parent both reject missing, changed, uncommitted or
non-descendant GO receipts. The GO must explicitly review the 88-forward
amendment and these executable sources. The earlier source-only GO does
not unlock this worker.

Only after that review, the fixed parent entry point is:

```sh
models/second_architecture/.venv/bin/python -m models.second_architecture.wav2vec2_fit_noise_norm_causal_launcher_v1 --admission models/second_architecture/WAV2VEC2_FIT_NOISE_NORM_CAUSAL_ADMISSION_V1.json --go models/second_architecture/WAV2VEC2_FIT_NOISE_NORM_CAUSAL_EXECUTABLE_GO_V1.json
```

The parent checks AC power, open lid, a stable ten-minute prior power
log, and no sleep/wake/thermal events through the run. It starts one
CPU worker process group, polls its process-tree RSS during execution,
records every sample and the sampled peak, and kills the group at 4 GiB
or after 1,200 seconds. The sampled peak is a bounded observation, not a
claim that sub-poll transients are impossible. The parent preserves
the only attempt directory. The worker enforces four Torch threads,
one pinned checkpoint load, 88 attempted calls, 4-GiB peak RSS,
2-GiB evidence, and complete logits/CTC paths. The parent records
pre/post host evidence and its own run result. A failed or incomplete
attempt is an invalid/censored observation; it is not relaunched.

The saved-result auditor is a separate read-only step after a successful
parent return. Give it an output path **outside** the attempt directory:

```sh
models/second_architecture/.venv/bin/python -m models.second_architecture.wav2vec2_fit_noise_norm_causal_audit_v1 --attempt models/second_architecture/runs/wav2vec2_fit_noise_norm_causal_v1/attempt_001 --admission models/second_architecture/WAV2VEC2_FIT_NOISE_NORM_CAUSAL_ADMISSION_V1.json --go models/second_architecture/WAV2VEC2_FIT_NOISE_NORM_CAUSAL_EXECUTABLE_GO_V1.json --output models/second_architecture/runs/wav2vec2_fit_noise_norm_causal_v1/attempt_001_audit.json
```

The auditor checks the complete parent RSS sample log against its recorded
peak and hard cap, and reconciles the durable call-progress file with the
parent's census. It does **no full candidate forward**. It validates all 88
ordered logits files and complete CTC paths, original pilot stock
endpoints, Stage I input statistics, and Stage II donor provenance. It
rebuilds the first convolution and GroupNorm from pinned checkpoint
tensors and saved inputs. Its numerical screen records changed-frame
top-two margins against the natural-diagonal reproduction error. A
screened flip remains inconclusive, and even an unscreened flip does
not establish unique mediation, fresh-audio transfer, attacker control,
or a security vulnerability.

## Review focus

Check source-binding/GO ancestry, the 12 unique Stage I stock controls,
read-only hook parity, GroupNorm donor capture under common outside-word
input normalization, exact natural CTC path parity, fixed numerical
tolerances, the exact attempted/censored call-name ledger, process-tree RSS
sample/kill path, CPU/host caps, and the saved auditor's
independent first-convolution reconstruction. The Stage I
feature-projection and post-block-0 time traces are only hashed summaries;
the auditor cannot reconstruct those later tensors without a full model
forward and reports that limitation explicitly.
