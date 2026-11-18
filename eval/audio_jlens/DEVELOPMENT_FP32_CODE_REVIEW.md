# Independent static review: FP32 development runner

Reviewed 2026-11-18 against `PLAN_FP32_V1.md` at commit `b171278`. I read the frozen runner and CPU tests, verified file hashes, and ran `py_compile`. I did **not** run development model inference, its input-validation command, MPS, or locked-test inputs.

## Decision

**No blocker found for the planned development run.** The runner's output must still be evaluated by its prespecified gate. A fit-only pass does not imply a development pass.

## Frozen inputs and scope

- Runner SHA-256 `4c2ac2c702126494f3d1d50ceaeddf69912a044720b8f216fda4e99d6083d77c` matches `DEVELOPMENT_FP32_HANDOFF.md`. Test SHA-256 is `2557952db01f29868c9359ec7cec1efa185c21040c4fd69292dc678103fa6d6e`. The bound plan, fit result, fit progress, fit norm baseline, and development manifest SHA-256 values all match the saved files. The runner additionally checks the fit status, permission flag, progress/baseline linkage, site floors, and the exact imported fit-helper source hashes before Qwen load.
- `verify_dev_manifest` fixes eight pairs across Piper/Kokoro × animal/weekday × chosen/remember, checks voices and word pairs, and verifies all 16 WAV hashes. `prepare_pair_inputs` checks matched row maps and prompt token IDs. The runner contains no path or loader for `test_stimuli_v1`; no locked-test audio is read by this program.
- Eight pairs × two directions × two primary sites produce **32 site cases**, or 16 directions per site. The seven fixed alpha values run on every case. Controls are generated only for `chosen`: eight directions per site, both directions for each category/voice. The off-word selection and control seeds come from the frozen fit helper.

## Measurement and gate

- The tokenizer check fixes `Cat=[26801,151645]`, `Dog=[48940,151645]`, `Monday=[35239,151645]`, and `Friday=[34620,151645]`. The inherited scoring helper teacher-forces the complete word-plus-EOS candidate at causal logits beginning at `prompt_len−1`. The no-grad unpatched contrast is the intercept for no-grad patched contrasts; the grad/no-grad intercept gap is recorded. FP32 effective edits, per-row VJP contributions, full generated IDs/answer/EOS, cache parity, and outside-word state differences are saved. Alpha zero must preserve score and generated token IDs.
- Clean eligibility requires **both** words of a pair to generate exact expected decoded words after trim/case fold and EOS. The code evaluates all cases, retains the all-planned 16-direction denominator per site, and restricts sign/error metrics to clean-eligible cases. Effects below the frozen fit floor remain in mean/median error metrics but not the determinate sign denominator. Alpha-one donor-answer flips have both all-planned and clean-eligible denominators, plus category, phrasing, and voice breakdowns.
- The gate requires ≥75% sign agreement among determinate clean small effects and Jacobian mean absolute error ≤0.75 times **each** nonzero zero-change and frozen norm-baseline MAE. Zero comparator error, no clean eligibility, or no determinate effects fail. Both sites must pass for `locked_test_permitted_by_development_gate=true`. The saved fit beta is 0 at both sites, so the two comparators coincide for this particular run; this is explicit in result fields rather than a code mismatch.
- The MPS allocator fraction and sidecar use the 30 GiB cap. Progress JSONL is compared with all 32 in-memory results before its SHA is written into the final report. Existing outputs are refused. `py_compile` passed; the handoff reports five CPU tests passed, and I statically checked their toy scoring, manifest/hash, denominator, and gate cases.

## Limits

- If an individual development case raises an invariant/gradient/memory error, the run stops with a partial progress JSONL rather than a final report containing a failed case row. Preserve that partial file and traceback as a failure; do not silently rerun with changed case selection or thresholds.
- Score hooks assert one call in the inherited helper, but saved score objects do not separately expose that count. Generation hook counts are saved.
- The program records clean-ineligible cases and excludes them from locality error/sign metrics as planned. A large ineligible fraction would narrow what a passed conditional gate means; always report both all-planned and clean-eligible counts.
