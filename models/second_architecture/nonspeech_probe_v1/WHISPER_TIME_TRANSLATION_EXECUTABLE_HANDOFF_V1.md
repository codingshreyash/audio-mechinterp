# Whisper time-translation executable handoff, V1

**Source-only handoff, 19 November 2026. No target-model inference has run for this protocol.** The independent executable reviewer must inspect the committed worker, saved auditor, parent, gate, their tests, the frozen plan, and the rendered-source receipt before any admission or GO is made. The design review at commit `15af9b7` passed the protocol and plan only; it did not authorize execution.

## Frozen question and schedule

The experiment uses only the already-open engineered `d08_beep_train_low` zero/noise pair. It compares a 0.5- or 1-second zero **prepend**, which shifts the sound, with an equal-length zero **append**, which holds the original sound at its old time. The original stage has 2 baselines and 26 arms (28 calls); each offset stage has 4 baselines and at most 58 arms (62 calls), for at most 152 calls total across three separate attempts. Each offset stage has both layouts. A layout without a complete clean zero/noise contrast skips its 29 prespecified arms while still saving its two baselines. The original stage stops before arms if the V4 contrast is absent. There is no replacement input or posthoc output-length adjustment.

The prior qualified patch used the **union** of two boundary windows. The original stage checks union and both isolated windows. The translated stages test fixed versus shifted union at `post_conv_pos` and `post_block0`, and isolated source/tail at `post_conv_pos`. The latter site is before self-attention; `post_block0` is coordinate-specific. Every cell has identity, unrelated donor, displaced equal-norm, and full-donor controls. Each translated cell also has an equal-norm correct-delta relocation into the competing window. Full token paths through EOS, first-step vectors, and unchanged language detection are audited. A strict crossover is descriptive in a single translated attempt until the separate original-stage viability is joined. Isolated source/tail tracking cannot be claimed if its original isolated window fails, regardless of a union result.

## Source, launch, and audit contract

The pinned manifest is `runs/whisper_time_translation_v1/source_v1/manifest.json`, SHA-256 `eec732fb24a4b238ad0666d52d168f5ba41ec38c9d315a057a39e0d53be1435b`. The frozen plan is `WHISPER_TIME_TRANSLATION_PLAN_V1.json`; its source builder refuses a changed manifest or inherited original contrast. Independent no-weight preprocessing receipt `WHISPER_TIME_TRANSLATION_INDEPENDENT_PREPROCESSING_RECEIPT_V1.json` documents exact +50/+100 mel-frame interior transport for prepend, fixed interior mel for append, and equal masks with 1050/1100 active frames. The live worker repeats those saved-feature checks before any arm, but a feature qualification failure does not select a new arm or pair.

The intended launcher is the gated parent. After a separate committed admission, independent executable review, and stage-specific committed GO, its stage command is:

`python -m models.second_architecture.nonspeech_probe_v1.whisper_time_translation_parent_v1 --stage original --admission <fixed path> --go <fixed path>`

The same command shape applies to `offset_8000` and `offset_16000`. The gate requires tracked clean source hashes and checks admission, review, and GO commit order before the worker imports model weights. The parent creates a new fixed attempt and sends a nonce-bound live heartbeat. **This is an accidental-run guard under a trusted local-owner assumption, not cryptographic proof of parent provenance:** a deliberate same-user local launcher can fabricate the launch file and pipe. The saved auditor therefore requires the completed parent receipt and checks its resource and host records; those records provide consistency checks, not an adversarial local-owner guarantee. The Mac host must have an open lid, no thermal warning, at least 80% battery initially and above 70% thereafter. The whole-attempt deadline is 1800 seconds, tree RSS below 4 GiB, evidence below 1 GiB, and power-probe gap at most 15 seconds. If any gate or cap fails, preserve the partial evidence and label the attempt invalid; never retry the same GO. AC power is preferred but the checked battery floor remains required.

An attempt writes `progress.jsonl` with attempted/completed pairs, `records.jsonl` with baselines, interventions, and a terminal skip list, hashed `.npy` sidecars, `host_samples.jsonl`, `worker_rss_samples.jsonl`, `post_power_log.txt`, `launch_start.json`, and `parent_result.json`. The saved auditor is read-only and loads no model weights:

`python -m models.second_architecture.nonspeech_probe_v1.whisper_time_translation_saved_audit_v1 --stage original --admission <fixed path> --go <fixed path>`

Run it for each stage only after the parent says `completed_pending_saved_audit`. It verifies the parent receipt and host/resource traces, plan order and skips, exact sidecar bytes and geometry, complete EOS paths, original V4 replay, translated mel shift, unchanged detection and input, and every edited hook row against the frozen donor/recipient states. `PASS` means the saved evidence is internally consistent. It is not an affirmative mechanistic or security finding. The three audit outputs must be joined under the prospective decision rule; if any stage is missing or fails audit, no cross-stage claim is available.

Save the three `PASS` JSON outputs in original, 8,000, and 16,000 order, then run the no-weight joint decision:

`python -m models.second_architecture.nonspeech_probe_v1.whisper_time_translation_saved_audit_v1 --join-reports <original-audit.json> <offset-8000-audit.json> <offset-16000-audit.json>`

This requires all three matching source identities and reports the four union/site/offset cells and two isolated source/tail/offset cells separately. Original isolated viability gates the latter; the original tail/end result remains separate. It will not label a security vulnerability.

## Source-only verification

Run no-weight tests with the environment that has NumPy and psutil:

`/Users/shrranjan/Garak/garak/.venv/bin/python -m unittest models.second_architecture.nonspeech_probe_v1.test_whisper_time_translation_worker_v1 models.second_architecture.nonspeech_probe_v1.test_whisper_time_translation_saved_audit_v1 models.second_architecture.nonspeech_probe_v1.test_whisper_time_translation_gate_v1 models.second_architecture.nonspeech_probe_v1.test_whisper_time_translation_parent_v1 -v`

These tests check conditional arm counts, direct versus relocated delta geometry, strict crossover interpretation, and parent/gate failure branches. They do not establish target-model behavior. Preserve the old V1/V2/V3 causal attempts and human/locked corpora unchanged.
