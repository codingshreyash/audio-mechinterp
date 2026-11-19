# V6 one-shot launcher HOLD correction: source handoff

19 November 2026. **Source-only correction; live pilot remains HOLD.** This addresses the two blocking findings in [the independent review](VOXTRAL_TRACK_B_V6_ONE_SHOT_LAUNCHER_INDEPENDENT_SOURCE_REVIEW_HOLD_V1.md). It does not fill the external A/R pins, create a 32/32 human listening receipt, load candidate weights, or issue post-pilot GO.

## Exact changed bytes

| File | SHA-256 |
| --- | --- |
| `voxtral_track_b_v6_one_shot_launcher_source_v1.py` | `14de1226c1747a8df316f7d2536023602574ab8323c9522313518ed5fe6b74bc` |
| `voxtral_track_b_v6_composed_saved_auditor_v1.py` | `26a1fed4bf71d11e4e4bce33ac430db7b7cc1414e67b1b797155cded07997dce` |
| `test_voxtral_track_b_v6_one_shot_launcher_source_v1.py` | `d62f8e887dcb136d1ba2d53329c18ba1c958fc255b3114897e4a4a43ff418f8e` |
| `test_voxtral_track_b_v6_composed_saved_auditor_v1.py` | `39c62c65a7959de1b24c219dd2ca47b5a01cac818c94e40d277ba194c7f14332` |

The new launcher serializes the bounded sidecar **before** writing the post-host raw or sidecar files. It inventories the existing worker output, worker campaign claim, and parent watchdog claim, rejects symlinks, includes the exact pending raw and sidecar byte lengths, and reserves one MiB for the parent's later watchdog result. It checks free disk before writing, writes both files exclusively, verifies exact lengths and hashes, then repeats the aggregate cap check. A partial filesystem failure remains visible in the failed parent attempt; there is no retry. The former 150-byte fake cap reproduction now rejects before either external file exists.

The sidecar now names **L**, `voxtral_track_b_v6_one_shot_launcher_source_v1.py`, as its producer and stores its actual SHA-256. It separately records the older C composition shell SHA-256 as a **schema/reference source**. The composed saved auditor requires an externally supplied exact L hash, rehashes the actual L source, and checks the sidecar's producer fields against it. Its returned evidence also records L and C in these distinct roles. L pins the changed composed-auditor source hash. The worker W and old C source were not changed.

The caller-supplied expected L hash is not independently authenticated by this code. A future external executable freeze and independent review must supply and bind that exact final L hash and the changed composed-auditor hash before live use; an independent post-pilot reviewer must check those external bytes again. A/R are still `None`, so the public entry raises before a child is created. Any later edit that fills A/R changes L's hash and requires a new final L review.

## No-weight verification

- Four targeted fake suites: **24/24 pytest tests passed** using the project Python with isolated `/tmp` pytest; no candidate weights loaded.
- Composed saved-auditor `unittest`: **5/5 passed**, including missing/tampered L provenance and a fake complete HOLD receipt.
- `py_compile` and `git diff --check` passed.

An independent source rereview of the exact changed bytes is required. This correction is not a resource-pilot result, an answer-link result, or a security finding.
