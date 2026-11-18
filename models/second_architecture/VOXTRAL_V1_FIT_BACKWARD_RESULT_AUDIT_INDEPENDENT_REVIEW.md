# Independent source review: old-FIT Voxtral backward result auditor

**18 November 2026. Source-level go for one bounded old-FIT pilot, conditional on an external result hash anchor.** I reviewed committed auditor `e3a515c` against producer/protocol `fc2e702` and the saved `details.pt` schema. I did not load Voxtral, run a pilot, open the active development output, or inspect fresh v2/locked-test data.

The auditor's `--validate-only` passed with exact producer SHA-256 `cc48437b96d47c766f5ba9f86eb5789481fecd7d1469c09a503d225a42ed3be1` and protocol SHA-256 `8fa6aa224e150b2579e335d14f7dafddd542eacdf733bc92ab7f3ff281384148`. It checked the frozen old-FIT first record, selected `word_0` rows, saved natural suffix, checkpoint tree, and source pins without opening an attempt. All **six** synthetic auditor tests passed under `.venv/bin/python -m unittest -v test_voxtral_v1_fit_backward_result_audit.py`.

For completed attempts, the code confines public `--run-dir` paths to the old-FIT backward root after resolving symlinks. It requires a single result or error receipt, the exact completed phase sequence, bound meta/progress/sidecar hashes, pinned model/source/runtime claims, resource caps, source/target shapes and dtypes, suffix parity, and CPU loading of the sidecar with `torch.load(..., map_location="cpu", weights_only=True)`. It checks full zero/alpha terminal tensor hashes and recomputes the BF16-realized edits, selected VJP contractions, four-coordinate measured differences, and error metrics from saved tensors. These checks establish consistency of the saved result, not independent execution of the model graph.

I also tried adversarial edits to synthetic receipts and sidecars:

| Edit | Audit result |
| --- | --- |
| Change an unselected zero-terminal coordinate, keep original receipt | Rejected: `details.pt` SHA mismatch. |
| Change a selected alpha-terminal coordinate, update sidecar SHA only | Rejected: terminal tensor SHA mismatch. |
| Change a selected stored gradient, update sidecar SHA only | Rejected: recomputed prediction mismatch. |
| Change failed-attempt progress hash | Rejected: invalid error receipt. |
| Change an unselected zero-terminal coordinate and update both its tensor SHA and sidecar SHA | Accepted; the four frozen coordinates and their recomputed metrics were unchanged. |

The last case is an **authenticity limit**: the receipt hashes are self-contained and do not detect a coordinated rewrite of the saved files. The pilot output directory is exclusive at creation but is not cryptographically immutable. **Before invoking the auditor on the live attempt, commit the complete raw attempt directory and compare the audited `result.json`/`details.pt` hashes to that git commit.** Do not claim the auditor verifies unselected coordinates against a model forward, or that it independently proves saved tensors came from the stated run. The four selected coordinates remain the only numerical claim.

With that external anchor, I found no source-level blocker to the single old-FIT engineering pilot under its 15-minute/30-GiB caps. A failed attempt is a feasibility result and does not authorize rerunning a different case, relaxing the protocol, or making a v2 efficacy/security claim.
