# Prospective v2 ASR and listening outcome auditor: source handoff

**Prepared before any v2 ASR or human-listening outcome.** `audit_asr_listening_v1.py` is a separate, standard-library-only CPU auditor. Its `--validate-only` mode independently rebuilds 480 case identities from the pinned manifest, verifies all 480 WAV/PCM hashes and the reference-free ASR projection, and regenerates the blind queue order. It does not load a recognizer, read a listener response, or inspect Qwen/Voxtral output.

The outcome mode requires both ASR output directories, the SHA-256 of each externally sealed `complete.json`, a listening directory, an externally sealed `responses.jsonl` SHA-256, and a new output path. It checks the ASR freeze receipt, runner/model/source provenance, CPU/four-thread decoding settings, each `complete.json` and raw file hash, exact ordered 480-row schema for **each** recognizer, and no reference fields in raw ASR rows. It checks the listening run's source/version provenance, exact 480-case blind queue, response hash chain, original primary order, stage owners, duplicate initial responses, independent reviewers, correction links, timestamps, and browser user-agent/version. Missing or invalid evidence writes an explicit `stop` receipt. The private report retains all case-level comparisons, original responses, revisions, failure reasons, and aggregate counts.

The listener collector now records `server_version: BlindListening/1` in `run_meta.json` and the HTTP `User-Agent` in each hash-chained response row. The auditor rejects a missing/changed server version and a missing, placeholder, or malformed browser user-agent. A user-agent identifies the reported browser version; it does not prove a human heard the clip. The human review and reviewer independence still require operational verification.

The separately versioned [middle-slot amendment](ASR_MIDDLE_SLOT_FLAG_AMENDMENT_V1.md) is pinned by SHA-256 in **both** the ASR runner and auditor. The runner rejects a changed amendment at source preflight and a freeze receipt missing `amendment_sha256` before either `--run` mode can load a model. The auditor requires a second listener whenever **either** ASR normalized full-script transcript differs from the normalized reference. Merely finding the target word somewhere in an ASR transcript cannot clear the flag: `Otter say other now` contains `otter` once, but does not match `Say otter now`. Whole-token occurrence remains a descriptive metric. A synthetic regression covers this case with both ASRs making the same error, so ASR disagreement cannot accidentally supply the flag. The amendment adds human review without changing the frozen corpus or original protocol hash.

## Conservative decisions requiring protocol clarification before a `go`

- A correction preserves the original answer. The prospective protocol does not specify how to establish that a correction is genuinely typographical or whether a corrected answer may replace an original vote. The auditor reports every correction and **stops** pending independent review/versioned resolution; it never silently substitutes the correction into the quality count.
- The collector permits an `adjudication` after an initial tertiary response. The prospective protocol specifies three blinded listeners, but does not define how a later adjudication changes that vote. The auditor preserves such rows and **stops** pending independent review/versioned resolution.
- A flagged clip needs a second independent response; a disagreement in word or clarity needs a third. Quality confirmation after a flag requires two independent original responses giving the exact target and marking it clear. An unflagged extra second response or an unrequired third also stops, so the population cannot be expanded or selected from outcomes without a new review plan.

Before any ASR inference, independently review this auditor and the updated listening collector, then commit a freeze receipt binding `script_sha256` for the ASR runner, `auditor_sha256`, `queue_sha256`, `server_sha256`, `html_sha256`, `app_sha256`, `amendment_sha256`, and the six source hashes already required by the runner. The runner accepts additional receipt fields; the auditor requires them. Freeze the listening interface and response file hashes before comparison. Run the two ASR passes serially after the active Voxtral run ends. No synthetic response in the tests is a human-listening result, and no current QA result authorizes Qwen v2 inference.

Source SHA-256 values for the pending freeze receipt:

| Source | SHA-256 |
| --- | --- |
| ASR runner | `6cc4e6eaa3cf93c3cdf1617589fb1d2ad5eb23890f376aafc5124029474ddcb5` |
| Outcome auditor | `2b39238dcad768a428b184a185b07217f592db4fcb5ebb5276fa5a0958857573` |
| Middle-slot amendment | `9a6c12199e591ac623ee74938d13a4a6b11dbbbadce21e79cb75bd888176246d` |
| Listening server | `05c9629d40f36103b61a6ad05a0a97e8db007d0af957be88484559546cfb38a7` |
| Listening HTML | `3c912bf01cac53f1c6b019e441a54a98294addbd2dcd814f4c7e27e675347cc3` |
| Listening JavaScript | `9053555674e5a6eac6f0160fc5a85900dd38e9ba4a3dd8a8abf4a0d479575962` |

Verification: `pytest -q test_audit_asr_listening_v1.py test_listening_server_v1.py test_asr_screen_v2.py` passed **18 tests and 2 subtests**. Both the auditor and runner `--validate-only` modes passed on all 480 frozen WAVs. Neither command performed ASR inference.
