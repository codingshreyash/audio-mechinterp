# Follow-up source review: VoiceChat calendar fixture `355fb5b`

18 November 2026. Scope: source inspection and local Python probes only. No TTS, model inference, cloud run, or real calendar action. Compared `355fb5b` with the [first fixture review](VOICECHAT_CALENDAR_FIXTURE_SOURCE_REVIEW.md), the [dual-head protocol](VOICECHAT_DUAL_HEAD_ASSAY_PROTOCOL.md), and pinned official [helper](https://raw.githubusercontent.com/NVIDIA-NeMo/Speech/097dfe9e2f55baf653b83035868bdc89849f1b47/nemo/collections/speechlm2/inference/utils/offline_voicechat.py) and [template](https://raw.githubusercontent.com/NVIDIA-NeMo/Speech/097dfe9e2f55baf653b83035868bdc89849f1b47/examples/speechlm2/function_calling/template.jinja).

## Disposition of the five earlier blockers

| Earlier issue | Result in `355fb5b` |
| --- | --- |
| Self-asserted QA receipt could authorize C | **Closed at the dispatcher trust boundary.** `verify_external_grant` requires an HMAC from a separately held 32-byte-or-longer key and checks controller ID, case C, the supplied WAV hash, receipt hash, exact authorized date, and `DR19`. The fake pulse receipt with positive booleans is rejected. A/B/D and literal controls cannot borrow a valid C grant, and a model-written `confirmed=true` is insufficient. |
| Calendar event aliasing | **Closed.** Both list events and dry-run `would_cancel` events are deep-copied. Mutating either returned list leaves `CALENDAR` and later responses unchanged. |
| Replay wrapper differed from official helper | **Closed for the stub response object.** `formatted_response` equals `<TOOL_RESPONSE>[` + default `json.dumps(response)` + `]</TOOL_RESPONSE>`, as in the pinned helper. This does not yet verify actual injected bytes or token IDs in a model run. |
| Missing two-pass adapter | **Still open and explicitly documented.** There is no adapter that captures pass 1, selects a call-specific response, then captures pass 2. The stock helper takes a prewritten `api_response` before pass 1 and returns only pass 2. A model-level dual-head claim is therefore still gated on the adapter and its identity tests. |
| Parser mismatch for non-JSON strings | **Closed for individual parsed calls.** `normalize_official_call` maps a string such as `"nonsense"` to `{}` and retains the call; a malformed string starting with `{` is marked parse-failed and is not replayed. It retains raw and normalized arguments. Full raw-call extraction, position retention, and multi-call handling remain adapter work. |

## Exact limits of the signed grant

The signature authenticates a **controller assertion**, not the waveform's audible content. In an independent probe, a trusted key signed an invented 64-character WAV hash and receipt hash; `dispatch` returned `authorized_dry_run` with `executed:false` without opening any WAV. That is consistent with the documented trust model: an external reviewer must listen, review independent ASR, bind the actual final WAV and receipt hashes, and protect the key. The future adapter must compute the hash from the same bytes it submits to the model; accepting a caller-supplied hash alone would make that binding only asserted. The receipt contents are not recomputed by the stub, and no source-only test proves a human heard the C phrase. A signed grant is reusable for the same case/WAV/date/ID; it is not a one-use execution credential, and the dispatcher remains inert.

The passed `confirmation_state` is included in returned evidence. Its signature is not a secret, but downstream records should preserve the exact grant and independently audited receipt, rather than treating the returned Boolean as proof of audible confirmation. The `confirmed` model argument is logged only: a valid external grant authorizes the dry-run even when the model writes `confirmed:false`. That is appropriate if the external reviewer, rather than the model, owns authorization.

## Local verification

`python3 -m unittest discover -s models/second_architecture/voicechat_calendar_fixture_v1 -p test_fixture.py -v`: **8/8 passed**. Separate probes confirmed the valid-grant path, rejection for wrong key/WAV/case/controller and tampered date/event/receipt/WAV fields, deep-copy list behavior, exact default-JSON wrapper, and the two string-argument parser fallbacks. A signed arbitrary hash demonstrated the explicit reviewer trust boundary above. No test executed a real action.

**Readiness:** source fixture is suitable for building the adapter and obtaining independent audio QA. It does not yet produce pass-labelled VoiceChat evidence or a security finding.
