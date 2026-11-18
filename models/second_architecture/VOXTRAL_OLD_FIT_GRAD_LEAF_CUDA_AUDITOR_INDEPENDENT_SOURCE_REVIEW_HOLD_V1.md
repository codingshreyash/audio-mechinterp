# CUDA old-FIT saved-data auditor: independent source review HOLD V1

18 November 2026. **Source-only review.** No CUDA worker, checkpoint load, or model forward was run for this review. The synthetic payloads establish auditor behavior only; their `same_pattern` labels are not experimental results.

## Reviewed source and contract

- Auditor commit `9173c8cdbd03d53f80bef82987007d4e75d4deef`; source SHA-256 `70780a3aeaf0790c9fb121571c0466e6640a6d46406101dab8116d1d5e6c2bd4`; original tests SHA-256 `c8ef2c858fd7846d4b0b40817ce6b2ba290b0a010e1d99567dfb08fb62abef99`.
- Runner commit `5663b94`; source SHA-256 `2bfb6be63102577dff06ef5a96330560a61a4584695bcfaba63318b41bbd8c9c`.
- V1 protocol SHA-256 `188925bc4022aa9841502d310899b5f9b28bf8ec0ba32ab7f7acb3fc45579050`; V2 amendment SHA-256 `c86537552b55f6542c88d2be9a7c71d5e1cecd589c6a11cc89eec514c49419ca`; result schema SHA-256 `c90f46adcf73a8466232be15ba37d9a1d18b0eca234f2b4648f480bff6b7ad4d`.
- Independent negative tests: `test_voxtral_old_fit_grad_leaf_cuda_audit_independent_review_v1.py`, SHA-256 `6626e9a2edcd7231dfa5c842cb0823e04f1f4a01a40c6f6da67b0726ec95f6ed`.

## Checks that pass

From `models/second_architecture`, `./.venv/bin/python -m unittest test_voxtral_old_fit_grad_leaf_cuda_audit_v1 test_voxtral_old_fit_grad_leaf_cuda_audit_independent_review_v1 -v` passed **9/9** tests: six original synthetic tests and three independent tests. The auditor enforces 32 saved factorial calls per case, two waterfall passive calls, the planned 66-forward/two-case/four-cell denominator, and 24 required direct/full raw-byte comparisons per cell (96 planned comparisons). A signed-zero difference in required logits is labeled `different_pattern` despite zero numeric changed coordinates. The tested source, passive, schedule, hook-count-zero, origin, and hashed-receipt tampering is rejected. A mixed valid/invalid synthetic campaign retains four cells and 66 planned forwards. These checks do not attest a live forward.

## Material gap: model and environment identity are self-asserted

V1 requires the independent auditor to verify model/environment identity. The current `audit_case` checks that `runner_sha256` agrees between `meta.json` and `result.json`, and that the checkpoint and environment receipt fields look like 64 hex characters. It does not compare any of the three with trusted values from an independently reviewed freeze or gate, nor does it open and hash the environment receipt. `model_layout.json` only needs `valid: true` and an empty `errors` list. The CUDA execution metadata needs some field names and control booleans, but the GPU name and software versions can be arbitrary strings; a driver version is not required by this auditor.

The independent tests demonstrate acceptance after changing the recorded runner SHA to `e` repeated 64 times, checkpoint tree SHA to `d` repeated 64 times, and environment receipt SHA to `f` repeated 64 times, then recomputing the self-referential `meta_sha256`. The result remains `audited_saved_data_only` with two synthetic `same_pattern` cells. A second test replaces the layout with `valid: true`, zero named parameters, zero parameter count, and an empty parameter list; refreshing `model_layout_sha256` still produces `audited_saved_data_only`. These are saved-record integrity failures relative to the protocol, not evidence that the runner executes the wrong model.

## Decision and required correction

**HOLD before executable freeze or CUDA model load.** Keep the independent auditor's identity check independent: pass exact, trusted runner/checkpoint/environment pins from the reviewed freeze or gate; compare the recorded fields to those pins; hash the actual environment receipt file; and verify the model layout's recorded structure against a separately pinned or otherwise independently justified expected structure. The runner currently expects 762 FP32 CUDA frozen named parameters, but this review does not assert that count as an independently verified checkpoint fact. Bind CUDA execution metadata to the reviewed environment receipt where the receipt supplies corresponding values. Keep test-only synthetic identity overrides explicit so production audit cannot silently accept them.

After a fix, rerun both suites and invert the two negative reproductions to require `invalid`. A separate reviewer should recheck the corrected source hash and the freeze before weights load. This review makes no claim of an audio-specific mechanism, security effect, or CUDA observation.
