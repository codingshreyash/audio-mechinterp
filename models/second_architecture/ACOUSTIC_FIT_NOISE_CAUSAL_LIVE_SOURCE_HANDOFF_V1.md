# Held live causal assay: source handoff

This package implements the already frozen, post hoc FIT-only Kokoro/teal Wav2Vec2 experiment in `ACOUSTIC_FIT_NOISE_CAUSAL_FOLLOWUP_PROSPECTIVE_V1.md`. It has **not** run on the candidate model. The earlier V1 failed attempt and V2 selection evidence are re-audited by the no-weight helper before admission. The result cannot establish generalization or a security finding.

`acoustic_fit_noise_causal_worker_v1.py` captures two no-hook baselines, then runs eight fixed feature-projection patch arms in each direction. It records complete CTC frame IDs and transcripts, baseline states, every logits tensor, edit hashes, hook counts, elapsed time and RSS. Self and full-donor logits must match the corresponding baselines exactly. `acoustic_fit_noise_causal_parent_v1.py` binds the fixed attempt path, checks AC power and open lid, owns the 1,800-second child watchdog, records pre/post power logs, caps all evidence at 128 MiB, and audits saved output without model weights. One attempted path only; no in-place retry.

## Review and machine GO order

1. Independently review the corrected causal helper/protocol and the no-weight preflight V2 receipt. Preserve the earlier HOLD review as history.
2. Independently review this worker, parent and fake tests. Run the no-model tests and source-only validation. Commit a worker GO JSON **after** the reviewed worker source, helper, protocol and preflight commits. The worker GO must contain `schema: acoustic_fit_noise_causal_worker_go_v1`, `status: GO`, and exact `source_sha256`, `helper_sha256`, `protocol_sha256`, `preflight_sha256` fields.
3. Commit a parent GO JSON in a **later separate commit** with `schema: acoustic_fit_noise_causal_parent_go_v1`, `status: GO`, the exact `source_binding` object printed by `acoustic_fit_noise_causal_parent_v1.py validate-only`, and exact `preflight_sha256`, `worker_go_sha256` fields. The parent source binding includes the earlier host/deadline helper.
4. Only then, on a machine showing AC power and an open lid, the parent `run` command may create `runs/acoustic_fit_noise_causal_v1/kokoro_teal_attempt_001`. Save `audit` output afterward. Do not infer, interpret, or publish if any parity or saved audit gate fails.

Source-only checks (no candidate checkpoint):

```sh
models/second_architecture/.venv/bin/python -m unittest discover -s models/second_architecture -p test_acoustic_fit_noise_causal_live_v1.py -v
models/second_architecture/.venv/bin/python models/second_architecture/acoustic_fit_noise_causal_worker_v1.py validate-only
models/second_architecture/.venv/bin/python models/second_architecture/acoustic_fit_noise_causal_parent_v1.py validate-only
```

The public `run` paths remain on HOLD while either GO receipt is absent.
