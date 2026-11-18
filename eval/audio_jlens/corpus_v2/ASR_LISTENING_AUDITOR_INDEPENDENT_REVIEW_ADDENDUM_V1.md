# Independent re-review of the middle-slot flag amendment

**Reviewed commit:** `0fe16cb` (18 November 2026), source and synthetic fixtures only. This addendum follows [the first review](ASR_LISTENING_AUDITOR_INDEPENDENT_REVIEW_V1.md) and its [saved counterexample](ASR_LISTENING_AUDITOR_COUNTEREXAMPLE_V1.json). No ASR inference, human response, Qwen output, or active Voxtral result was opened.

## Closed: the original false-go case

The new [prospective amendment](ASR_MIDDLE_SLOT_FLAG_AMENDMENT_V1.md) requires a second blinded listener for any recognizer's normalized full-script mismatch. The auditor now flags `asr_script_mismatch` for `Otter say other now.` against reference `Say otter now.` even though `otter` occurs once elsewhere. Replaying the saved synthetic counterexample against the new `audit()` returned `status=stop`, `requires_secondary=true`, `flag_reasons=["asr_script_mismatch"]`, and `missing_required_secondary`. The new regression checks this condition. When a second independent clear `otter` response is supplied in the synthetic test, the case can pass under the unchanged two-vote rule. This is a logic test, not a corpus outcome. [Auditor flag](audit_asr_listening_v1.py#L424-L450); [regression](test_audit_asr_listening_v1.py#L113-L137).

The amendment is separately pinned in the auditor's `frozen_cases()` and `verify_freeze()`, preserving the original protocol/case-map/queue hashes. `--validate-only` still reconstructed and hashed **480/480** cases and queue entries; the production auditor's two ASR inputs still require **480 raw rows each**. `python3.12 -m unittest discover -s eval/audio_jlens/corpus_v2 -p 'test_*v1.py' -v` passed 16 source/synthetic tests. [Auditor source checks](audit_asr_listening_v1.py#L94-L187); [ASR log checks](audit_asr_listening_v1.py#L190-L263).

## Remaining pre-freeze blocker

The amendment says its SHA-256 must be bound in the freeze receipt **before any ASR inference**. The outcome auditor enforces `amendment_sha256` in `verify_freeze()`, but the unchanged [ASR runner](asr_screen_v2.py#L165-L177) checks only the six original source hashes and its own script hash. In a synthetic no-inference test, `asr_screen_v2.check_freeze()` accepted a receipt that had all those original fields but **omitted** `amendment_sha256`. The later auditor would stop, but the runner can start ASR without the new pre-inference binding. This contradicts the amendment's stated execution gate. Add the amendment path/hash to the runner's frozen source check and require `amendment_sha256` in its freeze check; update the auditor's pinned runner SHA and tests, then re-review before creating a real freeze receipt. No current freeze receipt exists.

## Exact reviewed SHA-256 values

| File | SHA-256 |
| --- | --- |
| Amendment | `9a6c12199e591ac623ee74938d13a4a6b11dbbbadce21e79cb75bd888176246d` |
| Auditor after fix | `53344372bd8b06400768dac114e915fe6e8473298e8b63e48b5d1c56e60e0732` |
| Original protocol | `30285a6c897e742eff59f593b3d12a1bd92d62585cd72171837c3e18fabe662e` |
| Private cases | `d9a08a61b76e8ab56ab3a9b2d72a4aedc5871555487c7a33b712097fe3206ed8` |
| Blind queue | `19319ac394a24f8d742285d4bb3ecfd78020b1d131fa57def5b3362fd0b73873` |
| ASR runner before runner-side fix | `7df490c768d929e63845e00397ae1f0f77ba5b7f03722479fa8f5826527b4b78` |
| Manifest / technical binding | `15395c7d132eb3f1c1a6b2b7d555146faa83be4ee2b77fba4f481cf70f26761d` / `858f8f68fce65ee60e328b1536b34608ced6b3e84d725f229460c6376e14af93` |
| ASR input projection / model pins | `74f38f6704d1995b8e7040b329f7c0c0ea68b3cae36e774b875b4c358ae76d33` / `1136baf167244172d3058262ae484497f8028f02112c6a582a4267fa87b2bc1b` |

**Disposition:** The false-go logic is fixed. The runner-side freeze omission remains a blocker to the first v2 ASR pass under the amendment's own rule. No quality `go` or model inference is authorized by this review.
