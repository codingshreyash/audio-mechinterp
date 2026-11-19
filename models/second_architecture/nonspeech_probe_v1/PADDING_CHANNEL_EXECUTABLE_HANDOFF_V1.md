# Source-only executable handoff: Whisper padding-channel factorial

19 November 2026. **No candidate checkpoint was loaded or run for this handoff. HOLD for independent executable review, admission, and a later committed GO.** The reviewed [V3 design](WHISPER_PADDING_CHANNEL_INDEPENDENT_DESIGN_REREVIEW_V3.md) fixes the 436-call schedule; the independently reviewed [V4 evidence amendment](WHISPER_PADDING_CHANNEL_INDEPENDENT_DESIGN_REREVIEW_V4.md) corrects processed-score `-inf` handling and first decoder-state shape. The executable source binds the latter review's SHA-256 `ec0220b7d93a986e10522315ada0b2994a397975f5e545e1a51aa25d4d74a00f` and retains the exact V3 schedule SHA-256 `d939618689e989a2dfe249cc6d4f0d3b32a532c184991680b77f11b4af6a0ace`.

## Source inventory and roles

| File | Role |
| --- | --- |
| `padding_channel_plan_v1.py`, `padding_channel_plan_v4.py` | Reviewed 436-call order, geometry, seeds, source pins, corrected evidence estimate. |
| `padding_channel_core_v1.py` | Pure float32 two-site active/pad splice algebra, seeded same-support equal-L2 random edits, finite/raw and suppression-mask parity functions. |
| `padding_channel_worker_v1.py` | Fixed admission/GO chronology, two-phase/two-site hooks, one-shot model run, binary evidence, live reserve and worker caps. Import and `admit` do not deserialize weights. |
| `padding_channel_parent_v1.py` | AC/lid/thermal gate, bounded process-tree RSS/wall/evidence monitor, exact call progress, exclusive attempt and post-run host log. Reuses the pinned bounded-monitor functions from `causal_parent_v2.py`. |
| `padding_channel_audit_v1.py` | Read-only replay of parent, sidecars, complete outputs/EOS, support/capture hashes, every edit and control parity. It loads the tokenizer/processor from local pinned assets but never model weights. |
| `test_padding_channel_*_v1.py`, `test_padding_channel_plan_v4.py` | No-weight schedule, fake hook/array, mask/decoder, source/evidence, parent census, and saved-sidecar tamper tests. |

All these sources, the V1–V4 protocols, the V3/V4 design reviews, the reused bounded monitor source, and the model/processor pins are included in the admission source identity. The worker requires every source to be committed and clean; the independent executable GO must be committed **after** the admission and every bound source. The fixed attempt is `runs/padding_channel_v1/attempt_001`; it must not exist before a run. Earlier causal attempts and descriptive files are read only or untouched.

## Evidence and acceptance map

The first four calls are separate true no-hook and read-only captures for the two silence donors. Each recipient/site then runs `RR`, `read_only`, `RD`, `DR`, `DD`, `self`, two active random, two padding random, detection-only padding, and transcription-only padding arms in that order. The read-only call saves both site tensors in both phases before any edit. Later duplicate read-only captures must exactly match the earlier tensors. Each call saves complete prefix/generated IDs, text, language, EOS/censor status, first raw logits and processed scores, full `[1,4,384]` first decoder state, selected scores, and hook/geometry records. Both silence no-hook calls and every `DD` call save every raw logit vector through EOS or the 128-token cap.

The first processed-score vector permits `-inf` at suppressed tokens, rejects `NaN`/`+inf`, and requires at least one finite value. Parity requires identical negative-infinity masks plus finite-entry max-absolute difference `≤1e-5`. All raw logits and encoder/decoder states must be finite. No-hook/read-only/self and cross-site `RR` calls require complete token and first-array parity; `DD` additionally requires complete sequence and every raw-step vector parity with its same-duration silence donor. The saved auditor independently rehashes sidecars, decodes text/EOS, reconstructs donor/recipient/random edits, and reports every length-censored arm. A score shift without an EOS-complete sequence change is not a behavioral result.

The planner's conservative core is **1,377,212,576 bytes**. The worker and parent enforce 1,800 seconds, 4 GiB sampled process-tree RSS and 2 GiB actual evidence, with a 128-MiB pre-forward/pre-write reserve and a 256-MiB auxiliary allowance. The parent requires AC power, open lid, and no thermal emergency in the ten-minute pre-run log; it fences the post-run log through worker exit and caps collection lag at 120 seconds. These are source rules, not observations of a run. The sampled RSS limit cannot rule out transient peaks between polls.

## Review sequence

1. Run only the no-weight tests and inspect this source. Verify the fake hooks cover both sites and phases, true no-hook paths, the exact splice axis for 2-s/8-s cases, random seed/norm/support, immutable donor/native arrays, score-mask mismatches despite identical text, nonfinite raw rejection, full decoder-state geometry, evidence reserve, and call census.
2. Once the executable source is frozen, run the worker's **`admit`** command to write the fixed `PADDING_CHANNEL_ADMISSION_V1.json`. It hashes local model weights and installed source bytes without deserializing weights. Commit that receipt after confirming it exactly matches a fresh `preflight_receipt()` recomputation.
3. A separate reviewer creates a committed `PADDING_CHANNEL_EXECUTABLE_GO_V1.json` binding admission SHA-256 and the entire source map. The GO commit must postdate all bound source/admission commits. Only then may the parent `launch` path start one live 436-call attempt, and only if the host gate passes. The worker refuses `run` without this GO.
4. After the one-shot attempt, run the read-only saved auditor and independently review its report before making any causal or security claim. A partial run, failed parity, missing evidence, or host event stays visible and prevents a clean interpretation.

The source is submitted for independent executable review. **Do not create a GO, run the parent, or load candidate weights from this handoff alone.**
