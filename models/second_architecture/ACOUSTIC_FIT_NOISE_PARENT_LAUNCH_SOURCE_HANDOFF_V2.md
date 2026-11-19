# V2 Wav2Vec2 acoustic parent launcher: source-only handoff

19 November 2026. This [V2 parent launcher](acoustic_fit_noise_parent_launch_v2.py) is a separate, one-shot gate for the [V2 worker](acoustic_fit_noise_pilot_v2.py) and [capture-topology amendment](ACOUSTIC_FIT_NOISE_PILOT_PROSPECTIVE_V2.md). It does not alter the frozen V1 worker, V1 parent, V1 protocol, completed Whisper V1 attempt or failed Wav2Vec2 V1 attempt. The V1 Wav2Vec2 failure is byte-bound by three SHA-256 values before V2 source validation or admission. Only `runs/acoustic_fit_noise_parent_v2/wav2vec2_attempt_002` is an admitted new attempt; no replacement path or retry.

V2 is **Wav2Vec2 only**. The worker and parent both reject Whisper. V2 retains the reviewed parent's process-group kill on a hard 1,800-second timeout, pre/post AC and open-lid checks, ten-minute preflight power-log coverage, complete slot power-log coverage, UTC/monotonic chronology, full 20-row/clean-repeat/metric audit, evidence SHA map, byte caps and read-only saved audit. The parent source is copied from V1 with only V2 worker/protocol/test/handoff bindings, V2 schemas, Wav2Vec2-only admission and exact failed/new attempt paths. V2 capture itself changes only the installed Transformers 5.20.2 Wav2Vec2 block-output selection: feature projection tuple first; block outputs full batch-preserving Tensor. No acoustic stimulus or model checkpoint changed.

The restricted no-weight V2 worker preflight is [committed here](ACOUSTIC_FIT_NOISE_PREFLIGHT_V2_RESTRICTED.json), SHA-256 `df639af27950d5acb3f8b905ca30f83a3c8c099d749d9dc11464c5748aaeb03c`; it must be independently recomputed and reviewed. Worker GO uses `acoustic_fit_noise_review_go_v2`; parent GO uses `acoustic_fit_noise_parent_source_go_v2`. Each GO must bind exact current source, protocol and preflight hashes, and the parent GO must be in a strictly later separate commit than worker GO. Neither a V1 review nor the previous V2 preflight receipt authorizes this worker. The V2 parent `validate-only` and fake tests load no candidate weights. Do not run the candidate until the independent reviewer accepts source and both distinct GO receipts exist.

Source-only checks:

```sh
models/second_architecture/.venv/bin/python -m unittest models/second_architecture/test_acoustic_fit_noise_pilot_v2.py models/second_architecture/test_acoustic_fit_noise_parent_launch_v2.py -v
models/second_architecture/.venv/bin/python models/second_architecture/acoustic_fit_noise_pilot_v2.py validate-only
models/second_architecture/.venv/bin/python models/second_architecture/acoustic_fit_noise_parent_launch_v2.py validate-only
```

After independently reviewed GO, a separate controller can run `wav2vec2_attempt_002` with the V2 preflight and V2 GO files. The resulting 20-case matrix is FIT-only descriptive evidence. A failed or partial V2 attempt stays failed; source validation and byte-bound host receipts cannot be relabeled as a model finding.
