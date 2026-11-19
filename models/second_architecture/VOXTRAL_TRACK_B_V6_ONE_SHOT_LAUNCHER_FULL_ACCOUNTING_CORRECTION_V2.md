# V6 composed pilot receipt: watchdog accounting correction

19 November 2026. **Source-only correction; independent rereview required.** The first launcher correction fixed pre-write output limits and sidecar-producer provenance, but [independent rereview HOLD V2](VOXTRAL_TRACK_B_V6_ONE_SHOT_LAUNCHER_INDEPENDENT_REREVIEW_HOLD_V2.md) found that the composed saved receipt omitted the parent watchdog claim and result from `full_attempt_output_bytes`. No candidate weights were loaded for this correction.

The composed saved auditor now derives the watchdog paths from the admission-bound campaign claim. It requires regular, nonsymlinked claim and result files, checks the claim's schema, campaign path, output path and deadline, and applies the reviewed parent-result status, cap, timing and MPS sample validator. It includes both watchdog file sizes in the full output cap and reports their exact paths, sizes and SHA-256 hashes. The launcher compares that full saved total with the separate launch-freeze auditor's `total_attempt_bytes`; disagreement raises before any post-pilot result can be returned. L's pin of the composed auditor source was updated. The worker W and older composition shell C were not changed.

The fake regression adds exactly 73 bytes of JSON whitespace to the watchdog claim and 91 to the result. The saved full-attempt total increases by **164 bytes**. Separate tests reject a missing result, an invalid parent result and a watchdog claim pointing to the wrong output directory. The full saved total equals the independently computed launch-freeze inventory on a complete fake attempt.

| Changed file | SHA-256 |
| --- | --- |
| `voxtral_track_b_v6_composed_saved_auditor_v1.py` | `f12c64f2f64045b2b8aba1efc24a7bec8a37bb32ad7423baa7c79d8895524218` |
| `voxtral_track_b_v6_one_shot_launcher_source_v1.py` | `49c1df4fc4567b34b40b4681cb5fe313d437765e077d0aa00945623929e81c79` |
| `test_voxtral_track_b_v6_composed_saved_auditor_v1.py` | `38d9d9f76d3abb7d3b683f17657c1edcd7ebd05f92b5bc1bada176c93bde7b41` |

Verification: composed auditor **8/8 unittest**; four launcher/freeze/batch/composition fake suites **24/24 pytest**; `py_compile` and `git diff --check` passed. This is no-weight source verification only. A/R pins remain `None`, the 32/32 listening gate and external final-L freeze/review are absent, and live pilot/post-pilot GO remain HOLD.
