# Independent V5 saved-data auditor source rereview (V2)

18 November 2026. This is a separate rereview of correction commit `26521f5`. It preserves the earlier immutable HOLD review at `164be59`. No checkpoint, processor, audio corpus, or model forward was opened for this review.

**Decision: GO for non-beaver FP32 CPU auditor source only; model HOLD until external freeze.**

## Exact reviewed source

| Item | SHA-256 |
| --- | --- |
| `voxtral_fp32_nonbeaver_linearity_audit_v1.py` | `70a1ac3c4e4da92bb1fa1034bd866195024dcd423ea35a2386e233bce5dca8e2` |
| `test_voxtral_fp32_nonbeaver_linearity_audit_v1.py` | `3ffaa2224ca0c29ed5a49a985cf58b230aaaca9faa539214fa26548ca978c15a` |
| V5 prospective protocol | `99de95003190de6b5fb55c5ba8bed446dfea7cdd8cf53c0569d304bcb942b49b` |
| V5 CPU preflight receipt | `5ef97e47e66c37b6c8ecb64b366daf4a713cfea1dfdee2f7f59ca75324b08401` |
| V5 byte-budget receipt | `53d19e065669cce63256f39c3b1112bdbbdb52691f2fe5ae475f865ad6191a7f` |
| Runner | `91e622f7fd8f4d9dac00ca7a0baaf02d74faf9326109d10674ebf5673310bc04` |

## Independent checks

- All 23 CPU auditor tests passed (`python -m unittest -q test_voxtral_fp32_nonbeaver_linearity_audit_v1`, 0.271 seconds). The complete synthetic context tests the exact 96-forward order, 66 pairwise zero comparisons per site and mode, and 6 alpha cells per site and context, giving 24 cells per site across four contexts. The test suite checks G1/N0 and two-cycle arithmetic, four-probe contractions, realized fidelity and row nonuniformity, retained invalid and unresolved outcomes, saved tensor boundaries, signed-zero raw-byte parity, sidecar/result hashes, malformed fields, nonfinite values, and tampering.
- Independently mutated the original three false-handoff-flag cases: full LM-input-to-LM0, edited projector-to-LM-input, and edited LM-input-to-LM0. Each now raises `ValueError`. An edited LM0 digest disagreement and a changed retained full projector tensor with its digest recomputed also raise `ValueError`. The unmodified complete synthetic context remains accepted. These checks resolve the specific false acceptance documented in V1 HOLD.
- Ran a file-backed, production-shape synthetic audit: saved a 1,302,377,012-byte `.pt` with 96 forwards, reloaded it with `weights_only=True, mmap=True`, and called `audit_payload` against a custom production-dimension test spec. It produced 6 valid cells per site, 12 valid cells total, and zero computed invalidities in 2.019 seconds including construction and file I/O. The custom test basis and batch hashes establish feasibility of the saved-data computation, not validation of the frozen real inputs.
- Repeated the independent full-size serialization probe. The Piper sidecar was 1,302,437,429 bytes with SHA-256 `977e231a6fe90e748fb682796e232bce4bdc00f14aa9a04341f72e099e28e6ae`; Kokoro was 1,318,364,789 bytes. Reloaded the Piper file from disk, checked all 933 tensors for finiteness and 192 full tensor clean-to-zero boundary comparisons. The four-context sidecar headroom remained 3,348,330,156 bytes under the campaign cap. The probe completed in 1.677 seconds. These measurements are source-only synthetic checks, not model runtime predictions.
- Confirmed the V5 alpha-zero identity contract is represented in the auditor: full pre/post raw-byte equality including signed zero, all-zero effective rows, exact zero metrics, passive-recipient parity, and clean-versus-every-paired-zero full boundary parity. The auditor retains and reports invalid cells instead of silently dropping them. It checks the first-answer logits and LM0, LM14, and LM29 terminal tensors, plus edited handoff flags and matching digest claims.

## Scope and limit

The GO applies only to the corrected CPU auditor source and its saved-data checks. For edited forwards, full source/projector/LM-input tensors are not retained; their full hashes and handoff flags remain runner claims. Even for retained full LM input, the LM0-pre digest is a claim matched to that saved input rather than an independently reconstructed LM0 activation. The audit cannot attest that a model forward occurred, that the frozen audio was used, or that the model behaves linearly. The external executable freeze and model launch remain separate gates.
