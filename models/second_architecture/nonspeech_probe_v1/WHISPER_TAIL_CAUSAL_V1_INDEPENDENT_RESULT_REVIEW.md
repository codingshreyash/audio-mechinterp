# Independent review: Whisper Tiny tail-state pilot

19 November 2026. **PASS for saved-evidence integrity and the bounded execution.** This is a post hoc causal test of one engineered non-speech pair. It is not a security finding or a result about human speech.

## What was checked

I ran [the separate read-only verifier](whisper_tail_causal_result_check_v1.py). It imports no candidate worker or auditor and loads no model weights. It used the frozen [plan](WHISPER_TAIL_CAUSAL_PLAN_V1.json), SHA-256 `0b9117123e407ba2045ac535d1edacbb3fbfb8eb1bc47c522c3272320fa3c4eb`, and the two raw attempts. It checked the ordered call plans, every referenced sidecar hash and shape, every edited encoder tensor against a fresh reconstruction of the specified patch, V4 baseline mel/mask/logit parity, EOS or length-censor status, unchanged input preprocessing and language-detection branch, host and resource records, and the compact auditor outcomes. The raw [SHA-256 inventory](WHISPER_TAIL_CAUSAL_V1_RAW_FILE_INVENTORY.json) lists all 806 attempt files (403 per checkpoint). The large raw sidecars remain local and untracked.

The frozen source was committed at `db1951f`, admissions at `e18ba28`, independent executable review at `4fb38ba`, and model-specific GO receipts at `a285a6d`. Both attempts used those receipts. The saved [multilingual audit](runs/whisper_tail_causal_v1/whisper_multilingual_attempt_001_audit.json), SHA-256 `5f4e5823475f2c2b752eb1acc15ffb0c363c33d10abf320717c4920e06a5ba3a`, and [English audit](runs/whisper_tail_causal_v1/whisper_english_attempt_001_audit.json), SHA-256 `aedac86775386254512870924f1f98ede7fc3aee4c6bbfa378c90ceb624bb2ab`, both report `PASS`. Each saved run has 56 calls and 394 sidecars. The independent verifier recomputed the same ordered output and control results.

| Checkpoint | V4 zero/noise baselines | EOS-complete calls | Control-qualified local target arms | Qualified arms matching donor answer | Whole attempt / peak tree RSS |
| --- | --- | ---: | ---: | ---: | --- |
| Multilingual | Different: zero `You`, noise `I` | 54/56 | 9/12 | 3/12 | 17.35 s / 911 MiB |
| English-only | Same token path | 56/56 | 12/12 | 12/12 | 13.20 s / 794 MiB |

The multilingual donor matches are all **noise-recipient → zero-donor rescues**: at the post-convolution/pre-block-0 boundary rows, the same site's tail-interior rows, and the post-block-0 boundary rows. Each of these three local patches produced the complete zero-tail `You` answer, kept the recipient language-detection result and English prefix, and passed identity, displaced source-row, norm-matched shifted-delta, and full-donor controls. All six full-donor controls reproduced the donor complete answer **and** first-step raw logits exactly. No zero-recipient → noise-donor local patch induced the complete `I` answer. Two multilingual control calls (indices 46 and 50) reached the 128-token cap; they are length-censored and cannot be counted as complete answers. Three multilingual local target arms failed their control qualification.

The English-only checkpoint had the *same* complete output for zero and noise tails. Its 12 donor matches are therefore tautological: matching the donor also means preserving the recipient. They do not demonstrate an answer transfer or resistance to this perturbation.

## Interpretation and limits

The multilingual result shows that replacing a small region of an internal audio state can **rescue this one observed answer change**. It narrows where a causal intervention works in this frozen case. It does not show that the tail information stays confined to those rows at later layers, that the reverse change can be induced, or that the same intervention transfers to another sound, seed, speaker, or architecture. The selected pair was chosen after inspecting the V4 results. The next meaningful check is the reserved second noise seed and an independent English-sensitive contrast, followed by natural human speech and an audio-language model with a consequential action endpoint. Keep this mechanism result separate from the earlier tool-call integrity observation.
