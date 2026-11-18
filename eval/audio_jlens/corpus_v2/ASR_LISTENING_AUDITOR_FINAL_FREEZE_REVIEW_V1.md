# Final independent source review for the v2 ASR/listening freeze

**Reviewed commit:** `2a8519c` (18 November 2026). This follows the [initial review](ASR_LISTENING_AUDITOR_INDEPENDENT_REVIEW_V1.md) and [amendment addendum](ASR_LISTENING_AUDITOR_INDEPENDENT_REVIEW_ADDENDUM_V1.md). This is a **go for creating the source freeze receipt**, not an ASR outcome, listening result, or permission to run Qwen v2 inference. ASR checkpoint files were hashed without model loading or inference; active Voxtral results were not inspected.

## Previously blocking gates now close

- The [ASR runner](asr_screen_v2.py#L67-L79) pins and checks the [middle-slot amendment](ASR_MIDDLE_SLOT_FLAG_AMENDMENT_V1.md) during source preflight. Its [pre-inference `check_freeze()`](asr_screen_v2.py#L168-L181) also requires `amendment_sha256`. The [outcome auditor](audit_asr_listening_v1.py#L176-L190) requires the same field and pins the runner's new SHA-256. A synthetic freeze receipt with all 14 required fields was accepted by both; removing only `amendment_sha256` made **both** reject it. The runner's regression also confirms rejection occurs before an ASR output directory is made.
- Replaying the saved [synthetic false-go counterexample](ASR_LISTENING_AUDITOR_COUNTEREXAMPLE_V1.json) now yields `stop`, `asr_script_mismatch`, `requires_secondary=true`, and `missing_required_secondary`. It still retains `target_exactly_once=true` descriptively and `normalized_script_exact=false`. The second-listener requirement is therefore enforced when the target word appears in the wrong slot. [Auditor flag](audit_asr_listening_v1.py#L424-L454).

## Source and population verification

- The auditor's `--validate-only` returned **480 cases and 480 blind-queue IDs**, rebuilding identities and checking every WAV/PCM hash. The frozen population is 60 pairs: 288 fit, 96 development, 96 locked test, with no reserve activation. Its ASR outcome verifier still requires two complete, ordered **480-row** raw transcript logs and their external completion hashes. [Case verifier](audit_asr_listening_v1.py#L94-L172); [ASR verifier](audit_asr_listening_v1.py#L193-L265).
- The runner's `--validate-only` in the pinned `reproductions/audio_prior/.venv` returned **480 clips**, verified both cached model revisions/files and source pins, and loaded neither ASR model. The same environment ran `python -m unittest -v test_asr_screen_v2.py test_audit_asr_listening_v1.py test_listening_server_v1.py`: **18 tests passed**. A response-reader unclosed-file `ResourceWarning` remains a cleanup issue, not a gate bypass.
- The original protocol, private cases, queue, manifest, technical binding, ASR input projection, model pins, and listener sources have unchanged SHA-256 values. The only auditor edit in `2a8519c` updates its expected runner hash; the runner changes are the amendment source and freeze checks.

## Exact hashes for the pending freeze receipt

| Receipt field | SHA-256 |
| --- | --- |
| `script_sha256` | `6cc4e6eaa3cf93c3cdf1617589fb1d2ad5eb23890f376aafc5124029474ddcb5` |
| `auditor_sha256` | `2b39238dcad768a428b184a185b07217f592db4fcb5ebb5276fa5a0958857573` |
| `amendment_sha256` | `9a6c12199e591ac623ee74938d13a4a6b11dbbbadce21e79cb75bd888176246d` |
| `manifest_sha256` | `15395c7d132eb3f1c1a6b2b7d555146faa83be4ee2b77fba4f481cf70f26761d` |
| `binding_sha256` | `858f8f68fce65ee60e328b1536b34608ced6b3e84d725f229460c6376e14af93` |
| `cases_sha256` | `d9a08a61b76e8ab56ab3a9b2d72a4aedc5871555487c7a33b712097fe3206ed8` |
| `queue_sha256` | `19319ac394a24f8d742285d4bb3ecfd78020b1d131fa57def5b3362fd0b73873` |
| `inputs_sha256` | `74f38f6704d1995b8e7040b329f7c0c0ea68b3cae36e774b875b4c358ae76d33` |
| `pins_sha256` | `1136baf167244172d3058262ae484497f8028f02112c6a582a4267fa87b2bc1b` |
| `protocol_sha256` | `30285a6c897e742eff59f593b3d12a1bd92d62585cd72171837c3e18fabe662e` |
| `server_sha256` | `05c9629d40f36103b61a6ad05a0a97e8db007d0af957be88484559546cfb38a7` |
| `html_sha256` | `3c912bf01cac53f1c6b019e441a54a98294addbd2dcd814f4c7e27e675347cc3` |
| `app_sha256` | `9053555674e5a6eac6f0160fc5a85900dd38e9ba4a3dd8a8abf4a0d479575962` |

Set `status` to `independently_reviewed_asr_screen_frozen` in the receipt and commit it before any ASR pass. **Verdict: go for that freeze, with no remaining source blocker found in this re-review.** Subsequent inference and human listening remain separate work governed by the prospective protocol and any active-run scheduling constraint.
