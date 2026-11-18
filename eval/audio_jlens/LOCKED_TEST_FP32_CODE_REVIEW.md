# Independent static review: frozen FP32 locked-test runner

Reviewed 2026-11-18 against the frozen `PLAN_FP32_V1.md` at `b171278`, committed runner `31b9fb5`, and the completed fit/development artifacts. I read source, ran CPU-only checks, and did **not** run Qwen inference or inspect locked-test model answers.

## Decision

**No blocker found for the one-shot locked-test run.** The code binds the precommitted model inputs, candidate answers, analysis rule, controls, and prerequisite results before loading model weights.

## Evidence

- The runner SHA-256 `dd9ea047caf3284844d2bb1226bedd807822eb66b11bd9d1e6312d0813a928d1` and CPU-test SHA-256 `bd3f7db3b31785beb2a595be8d8f42c581dfc419fc879800ed69a9ade5c0e62f` match the committed handoff. The frozen plan SHA-256 is `c5cc77a37ebaa69c0d231c83cd19e0956c7a5c8ab87e1aae067f3812b6a5b34b`; the test manifest SHA-256 is `88bb12c02bc000ffee7dee8329ac3aeddd57d5b44898786f8e6c88d63454f845`. The runner checks both and all 16 test WAV hashes.
- `verify_prerequisites` binds the passed fit result/progress/baseline and passed development result/progress/source hashes. It checks both gate statuses and permission fields, the fit numerical floors, the imported fit-helper hashes, and the pinned Transformers source revision before model load. No development or test result can change those gates without a hash mismatch.
- The locked manifest fixes eight pairs across Piper/Kokoro × direction/number × codeword/term, with the planned voices and opposite-word pairings. The runner runs two directions at connector and encoder31: **32 site cases, 16 directions per site**. Controls run for both directions of `codeword` only: eight directions per site. The seven alpha values and seven named controls are inherited unchanged from the fit/development code.
- Candidate token verification fixes title-case `North=[25221,151645]`, `South=[25018,151645]`, `Seven=[59528,151645]`, and `Eight=[59085,151645]`. The inherited score helper teacher-forces each complete word-plus-EOS answer from causal logits. Greedy generation requires exact decoded word under trim/case fold and EOS. The runner reports full generated IDs, score contrast, no-grad intercept, grad/no-grad gap, FP32 effective edit, row contributions, cache parity, and outside-word state difference; alpha zero must preserve score/token IDs.
- The locked gate uses the **frozen fit** site floor and β. It retains all-planned and clean-eligible direction denominators, counts only determinate clean small effects for sign, keeps indeterminate clean effects in MAE, requires ≥75% sign agreement and Jacobian MAE ≤0.75× **each** nonzero zero/norm comparator. Both sites must pass. Alpha-one complete donor-answer rates and category/phrasing/voice breakdowns are separate readouts. Since fit β is zero at both sites, the two prespecified error comparators coincide numerically here.
- The 30 GiB MPS allocator cap, driver memory log, progress JSONL, completed-progress equality/hash check, and no-overwrite output rule are present. A source diff against the frozen development runner shows only expected input/answer allocation, prerequisite, naming, and label changes; the core measurement loop and gate arithmetic are unchanged.
- `py_compile` passed; all **5 CPU toy tests passed**. `--validate-inputs-only` passed manifest/WAV/prerequisite hashes and tokenizer IDs without model inference. That input-only check emitted existing Transformers config-range warnings for unused Omni token fields; it did not fail candidate/EOS validation.

## Boundaries

The runner contains no locked-test answer lookup or adaptive threshold/site/alpha selection. Its input-only path does not generate test answers. The default output's no-overwrite guard supports a one-shot run; an alternate `--output` could technically permit another run, so the one-shot rule also depends on run discipline. A runtime invariant failure leaves partial progress rather than a completed gate report and must be reported as such. Some inherited exceptions still say “Development”; this is a message-label issue, not a change in logic.
