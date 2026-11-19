# Independent V6 one-shot launcher accounting rereview

19 November 2026. **Decision: GO for the corrected source-only launcher/saved-auditor slice in `65130a1`; HOLD for any live pilot.** I independently rereviewed the correction against [HOLD V2](VOXTRAL_TRACK_B_V6_ONE_SHOT_LAUNCHER_INDEPENDENT_REREVIEW_HOLD_V2.md) and the prior cap/provenance HOLD. No candidate weights or Track B audio were loaded.

Exact SHA-256: launcher `49c1df4fc4567b34b40b4681cb5fe313d437765e077d0aa00945623929e81c79`; composed saved auditor `f12c64f2f64045b2b8aba1efc24a7bec8a37bb32ad7423baa7c79d8895524218`; composed auditor tests `38d9d9f76d3abb7d3b683f17657c1edcd7ebd05f92b5bc1bada176c93bde7b41`; correction handoff `3595ead5306be9c801abfcfb3b79caaf060a445621b614dbc83418616a8656f6`. The launcher’s `COMPOSED_AUDITOR_SHA256` equals the actual corrected auditor digest. The previous source-only 150-byte prewrite rejection and L-as-producer/C-as-schema checks remain in place.

## Recomputed accounting

The composed auditor now derives the watchdog claim and result paths from the admission-bound campaign claim. `base.read_json` rejects absent, symlinked, or non-object files. The auditor checks the watchdog claim schema, exact campaign/output paths and 240-second deadline, then reuses the reviewed parent-result validator for child completion, elapsed time, MPS cap and sampled lower-bound fields. It adds both watchdog file sizes to the cap and reports their paths, hashes and sizes. The launcher compares the resulting `full_attempt_output_bytes` with `freeze.audit_composition_sidecars(...)["total_attempt_bytes"]`; disagreement raises.

I independently repeated the former counterexample using the fake fixture: before extra bytes, the composed total was **42,863**. Appending 73 JSON-whitespace bytes to the watchdog claim and 91 to the result produced **43,027**, an exact **164-byte increase**. The separate freeze inventory was also **43,027**. The fake tests additionally reject a missing watchdog result, invalid parent result and wrong output path in the watchdog claim. This verifies the bounded saved-accounting correction, not a live resource measurement.

## Verification and remaining gates

The composed auditor unittest passed **8/8**; four launcher/freeze/batch/composition fake pytest suites passed **24/24** using project Python and isolated `/tmp` pytest; `py_compile` passed. The saved auditor still returns `HOLD_no_post_pilot_go`, and its expected L digest is caller supplied. A later external executable freeze and independent reviewer must authenticate the exact **final** launcher digest after A/R pins are filled; source self-pinning would be circular. Actual 32/32 human listening QA, external A/R, measured pilot, semantic host-log review and separate post-pilot GO remain absent. This source GO is not authorization to load candidate weights or claim a complete-answer mechanism.
