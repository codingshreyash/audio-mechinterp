# Independent Stage-A audit-only v2 source review: GO, continuation HOLD

18 November 2026. I independently reviewed `1394da6` and its diagnosis `3cbbc65` against the frozen v1 auditor at `0bd048c`, its prior independent rereview, and the failed BF16 attempt manifest `bae7c91`. **GO for a separate posthoc CPU audit receipt of the preserved BF16 raw bytes. HOLD for FP32 and Stage B until that receipt, its independent review, and a new externally reviewed continuation freeze/gate exist.** This review neither changes the original `launch_failed` status nor authorizes a model retry.

## Exact source change

The v1 and v2 auditor files each have 702 lines. A complete unified diff of v1 at `0bd048c` against v2 at `1394da6` has one changed line, inside `validate_layout`: a row device must now be in `("mps", "mps:0")` where v1 required `"mps"`. No other source line changed. Therefore all dtype, parameter-count, shape, aggregate device-count, hash, progress-phase/count, saved-tensor, numerical, parity, and resource checks are textually identical. The v1 auditor, runner, native hook source, and prospective protocol in the working tree still match their reviewed commits byte for byte.

| File | SHA-256 |
| --- | --- |
| Frozen v1 auditor | `4f800ed604295ae0e4fbd53c893fccd14b8968badb43b835e6ec975e49b0cf3d` |
| v2 auditor | `8d5f8b23bd32add8464bd976d42cd0225fbc5ee86510adc441b714de5a8fd275` |
| v2 CPU regression | `7de8085d82540cde9b913fc525526eee7017ec53aea76ae5490bb8f7b6e82189` |
| Audit-only amendment | `c67a3605d18e98eb043c0702c10ba123963bb1f6e2b40e7abb9ac6553fed2c14` |

## Independent execution and evidence

I ran `models/second_architecture/.venv/bin/python models/second_architecture/test_voxtral_old_fit_handoff_stage_a_audit_v2.py`: **2 tests passed**. They show v1 rejects `mps:0`, v2 accepts exactly `mps` and `mps:0` for both arms, and v2 still rejects CPU, nonzero or malformed MPS spellings, a wrong BF16 dtype, an altered position-table shape, and changed aggregate dtype/device counts.

I ran the v2 auditor directly on the saved BF16 `runner/` directory. It exited 0 with `audited_completed_old_fit_handoff_stage_a_native_only`, 36 native pairs, 72 patch forwards, 44 VJPs, 24 LM0 control cells, and 214 progress lines. It reported result SHA-256 `f7b3906aff2b1bb9c7b8fb818402982a9020d8507fcba9cf51cc58821dddd325`, details `df9fe20d1fd961a0b3502e073f933eed658f7a15de3214087552987ee8e8e849`, probes `dad65cd52beff26ed4f6dfcc5b541a9a0d504a207f23c2783f8ce71113f6b53e`, and the v2 auditor hash above. I independently recomputed the SHA-256 and byte size of **all 12** files in the failed-attempt manifest. All match; the 10 Git-committed files also match `bae7c91` byte for byte. The two large content-addressed tensor sidecars are present locally and match their recorded hashes. I ran no model inference and changed no raw file.

The original `launch_finish.json` still has SHA-256 `15ed053c8ca4e619735cce3319b5d2326ff4b19216475754acc2cbf4e3bcdf9a`, `status: launch_failed`, runner return code 0, auditor return code 1, and no audit hash. A successful v2 audit is a later assessment of preserved bytes; it does not turn that frozen launch into a success. The amendment correctly requires a separate content-bound posthoc receipt, independent interpretation review, and a new external continuation freeze/gate before FP32 or Stage B. The saved-tensor audit does not independently attest the live model execution or the provenance of saved VJPs.
