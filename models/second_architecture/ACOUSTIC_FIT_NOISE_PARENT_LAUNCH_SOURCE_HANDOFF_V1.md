# FIT acoustic pilot parent launcher: source-only handoff

18 November 2026. This [parent launcher](acoustic_fit_noise_parent_launch_v1.py) is a separate execution gate for the frozen [pilot worker](acoustic_fit_noise_pilot_v1.py) and [protocol](ACOUSTIC_FIT_NOISE_PILOT_PROSPECTIVE_V1.md). **No candidate inference has run under this launcher.** It makes one CPU process per named model, uses an external 1,800-second subprocess deadline, and kills the process group on timeout. A child-process exit or timeout is retained in an exclusive attempt directory and is never retried in place.

Before it creates an attempt directory, `run` recomputes the worker's no-weight preflight, requires the worker's separately committed GO receipt, and requires a **second, separately committed launcher GO** bound to the source hashes, existing preflight receipt and worker review. Thus this source handoff and fake tests do not authorize a model load. The required launcher review JSON schema is `acoustic_fit_noise_parent_source_go_v1`, `status=GO`, with `source_binding` exactly equal to `validate-only` output and `preflight_sha256` / `worker_review_sha256` equal to the named immutable files. The review commit must descend from, and differ from, the launcher/test/handoff source commits. Reviewers should independently inspect that JSON and all source bytes before issuing GO; the file's fields are admission bindings, not a self-review.

The host probe reuses the already reviewed Voxtral continuation gate's `_host_probe(require_stable=True)`: external power, open clamshell, ten-minute power-log coverage and no recent Sleep/DarkWake/Wake/Thermal event. It archives pre/post raw power logs losslessly in bounded gzip files. The post probe additionally checks coverage and transitions from launch start through worker finish. UTC and monotonic elapsed times must agree within five seconds. A host-check failure, timeout, nonzero worker exit, missing or duplicate case row, or >16 MiB **parent-only** capture sets `invalid_or_partial`. The worker separately limits its JSONL output to 32 MiB and checks its own RSS/time between forwards. The parent timeout covers a blocked model forward; it does not stop the Mac from sleeping. An undetected host transition is therefore an environmental validity risk, handled by the post-interval log check rather than assumed away.

No command should be issued before independent GO. Source-only checks:

```sh
models/second_architecture/.venv/bin/python -m unittest models/second_architecture/test_acoustic_fit_noise_parent_launch_v1.py -v
models/second_architecture/.venv/bin/python models/second_architecture/acoustic_fit_noise_parent_launch_v1.py validate-only
```

After a separate GO, run a single model with a new exclusive attempt path, for example:

```sh
models/second_architecture/.venv/bin/python models/second_architecture/acoustic_fit_noise_parent_launch_v1.py run \
  --model whisper \
  --receipt models/second_architecture/ACOUSTIC_FIT_NOISE_PREFLIGHT_V1.json \
  --worker-review models/second_architecture/ACOUSTIC_FIT_NOISE_WORKER_SOURCE_GO_V1.json \
  --launcher-review models/second_architecture/ACOUSTIC_FIT_NOISE_PARENT_SOURCE_GO_V1.json \
  --attempt models/second_architecture/runs/acoustic_fit_noise_parent_v1/whisper_attempt_001
```

The proposed review filenames are placeholders, not existing approvals. Check `launch_result.json` plus the 20 case rows; a `complete` receipt is an execution-integrity result, **not** an acoustic-mechanism or security result. The experiment is FIT-only, synthetic English speech with a single Gaussian noise setting. It has no human intelligibility record, state intervention, or attacker-reachable consequential behavior. Report the full model outcomes and failures before choosing a later causal test.
