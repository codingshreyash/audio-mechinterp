# FP32 audio-Jacobian fit gate: passed, with localization limits

The [frozen FP32 plan](PLAN_FP32_V1.md) was committed in `dbef46b` and clarified in `b171278` before any development or locked-test Jacobian run. The [fit-only result](fit_fp32_phase1_v1_results.json), [progress log](fit_fp32_phase1_v1_results.progress.jsonl), [norm baseline](fit_fp32_phase1_v1_results.baseline.json), and [independent review](PHASE1_RESULT_INDEPENDENT_REVIEW.md) were committed together in `7abbc6b`. This gate uses only the original exact-PCM `red`/`blue` pair in Piper and Kokoro voices, both directions. It tests a per-input scalar VJP, **not** a reusable corpus-fitted Jacobian Lens.

All four unpatched FP32 replies were the expected complete color word plus EOS. At each primary site, a full donor-state replacement in nominal spoken-word rows flipped the complete answer in **4/4** directions. At the prespecified small steps `α=.005,.01,.025`, every one of the **12/12** score changes per site exceeded the frozen `1e−4` numerical floor and had the sign predicted by the local Jacobian. Alpha-zero score and answer controls were exact, and cached/uncached source states matched within the fixed tolerance. Peak observed MPS driver allocation remained below 21 GiB.

| Primary site | Fit small-step sign | Jacobian mean absolute score error | Zero-baseline error | Frozen norm-baseline coefficient |
| --- | ---: | ---: | ---: | ---: |
| Audio connector | 12/12 | 0.000834 | 0.025921 | 0 |
| Post-block encoder layer 31 | 12/12 | 0.004323 | 0.034757 | 0 |

The norm-only coefficient clipped to zero at both sites because the fit directions include positive and negative small score changes. It is therefore the same predictor as the zero baseline on this fit set; beating it adds no separate evidence here. The connector and late-encoder local predictions are accurate **near each recipient anchor**, while the full `α=1` effects are much larger and nonlinear. For the Kokoro routes, the small donor-oriented score movement is negative even though the complete answer eventually flips to the donor at full replacement.

The full-swap control outcomes limit a word-local interpretation. A row-shuffled donor edit flipped the answer in **3/4** connector routes. At encoder layer 31, a norm-scaled off-word edit flipped **1/4**; two fixed-seed random same-row edits each flipped **1/4**. Natural off-word edits and self patches flipped **0/4** at both primary sites. Thus a full donor-answer flip alone does not isolate a specific word representation; large edits can induce the answer through other directions. The small-step VJP result is a narrower, quantitative finding about the model's local response to a specified edit.

The plan's **fit stop rule passed**, authorizing one development evaluation with the committed baseline and no changes to the locked test. Any development failure remains a result. No security exploit, cross-model transfer, or corpus-level lens has been established by this fit gate.
