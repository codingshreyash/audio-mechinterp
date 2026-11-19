# Independent saved-result review: Whisper FIT acoustic-noise pilot

19 November 2026. **GO for this one saved Whisper engineering attempt; no causal or security finding.** I independently reviewed the frozen [protocol](ACOUSTIC_FIT_NOISE_PILOT_PROSPECTIVE_V1.md), the [worker](acoustic_fit_noise_pilot_v1.py), the [parent launcher](acoustic_fit_noise_parent_launch_v1.py), and `runs/acoustic_fit_noise_parent_v1/whisper_attempt_001`. I ran the launcher's read-only `audit` with the committed preflight and both independent GO receipts. It returned `complete_evidence_pass`, result SHA-256 `6b3c5736451ae91e1564ec389b81ccc1c5832b33f60255bbfbc3b2f5bf36c80f`, and worker JSONL SHA-256 `8c66d87457afddbb0cf2648e3d93e5b547f1efb6905a0b935e97f990158dc1c2`. I separately rehashed every file against the parent result inventory. No new model inference was run for this review.

The saved JSONL has the planned **20/20 case rows**: four previously inspected FIT source clips × clean, clean-repeat, full noise, word-slot noise and outside-slot noise. All 20 generated answers reached EOS without censoring. Clean-repeat matched each clean complete transcript, token-ID sequence and every saved state hash exactly. Every recorded hook fired once; metric values are finite, and the largest relative discrepancy in `whole_l2² = word_l2² + outside_l2²` was below `1.6×10⁻¹⁴` across saved stage/arm values. The parent recorded AC power, an open lid, no interval sleep/thermal event, zero worker errors and a complete saved-evidence audit. These checks validate the **saved attempt**, not independent replay of model weights.

## Complete transcript outcomes

Across the **12 noise arms**, nine complete transcripts were byte-identical to their own clean transcript. Three differed in punctuation/capitalization or pause rendering; their generated token IDs differed too:

| Source | Arm | Clean fragment | Noise fragment |
| --- | --- | --- | --- |
| Kokoro/teal | Word-slot noise | `Teal... What` | `Teal. What` |
| Piper/silver | Outside-slot noise | `is silver. What` | `is... Silver. What` |
| Piper/teal | Word-slot noise | `is teal. What` | `is... Teal. What` |

The intended **silver/teal word was present once, without substitution, in all 20 complete transcripts**, including every noise arm. Full-noise arms changed no complete transcript. The three text differences do not show a changed authorization, action, or semantic answer. Counts are descriptive for four correlated synthetic FIT clips and one frozen noise realization, not an attack success rate.

## First observed spread and its limit

For each of the four **word-slot-noise** arms, saved mel and post-convolution/position states had exactly zero L2 change in the nominal outside-word rows. The first saved outside-row difference was after encoder block 0: outside L2 was `4.8933` (Kokoro/silver), `4.6382` (Kokoro/teal), `7.9051` (Piper/silver), and `4.8215` (Piper/teal). Later saved states also differed outside the nominal window. For full-noise and outside-slot-noise arms, the earliest measured outside difference was already in the mel representation, as expected because those waveforms directly change audio outside the word. This identifies the first **measured** spread under the frozen window mapping; it does not isolate attention from other block operations or prove that outside-row changes influence the transcript.

Every noisy arm changed the final encoder state relative to its own clean recording (post-norm whole-state L2 ranges: full noise `215.8–229.3`, word-slot noise `102.7–138.3`, outside-slot noise `199.0–213.7`). These norms are within-model descriptive measurements, not a cross-model scale or a decision-relevant effect size. In this attempt, large internal changes mostly coexisted with a stable lexical transcript. A targeted state patch with matched random controls would be needed to test causal relevance; a fresh held-out speaker, blinded human intelligibility, and an unmodified assistant decision would be needed for broader or security claims. The separate Wav2Vec2 attempt stopped on its first forward because of a capture-hook shape mismatch and is **not** included in these Whisper counts.
