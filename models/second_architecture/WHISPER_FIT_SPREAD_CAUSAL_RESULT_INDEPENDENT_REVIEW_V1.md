# Independent result review: Whisper FIT spread-versus-use assay

18 November 2026. Review target: `runs/whisper_fit_spread_causal_v1/20261118T2321Z`, under the [prospective plan](WHISPER_FIT_SPREAD_CAUSAL_PROSPECTIVE_V1.md), [plan review](WHISPER_FIT_SPREAD_CAUSAL_INDEPENDENT_PLAN_REVIEW_V1.md), [source review](WHISPER_FIT_SPREAD_CAUSAL_INDEPENDENT_SOURCE_REVIEW_V1.md), and [frozen preflight review](WHISPER_FIT_SPREAD_CAUSAL_PREFLIGHT_INDEPENDENT_REVIEW_V1.md). I did **not** load or run the Whisper model, alter the raw attempt, or inspect corpus V2 or a locked test. **Decision: GO for saved-data consistency and the narrow, posthoc FIT causal observation.** This is not a held-out model result or a security finding.

## Evidence identity and checks

- The committed no-weight preflight SHA-256 is `23bfbb7542dbcb2d993b558a30bf2fefee9bd73943dc4aa2476db967af54e59c`. It binds the eight synthetic FIT WAV/PCM inputs, `openai/whisper-tiny.en` revision `87c7102498dcde7456f24cfd30239ca606ed9063`, checkpoint/assets, installed source hashes, CPU settings, runner commit, fixed regions, and 32 derived random seeds. The independent source and preflight GO reviews precede the live attempt in Git.
- Raw [`records.jsonl`](runs/whisper_fit_spread_causal_v1/20261118T2321Z/records.jsonl) SHA-256 is `aaf8220ab1e0af35b4951317133c90e937c00e8ad7a7b97280e38722f9b633cb`; saved [`audit.json`](runs/whisper_fit_spread_causal_v1/20261118T2321Z/audit.json) SHA-256 is `795f8b16eb2bb05507432ebd5fa5f1c4ad1a311d76d1132a73e57d79f723c38e`. I reran the reviewed auditor to a separate `/tmp` file; its JSON was byte-identical (`cmp` exit 0) and reported `independent_saved_data_audit_pass`.
- I independently rebuilt the 120-arm set from pair, voice, direction, site, condition and seed. All 120 identities are present exactly once. The raw file has 134 rows: one preflight, eight eligible baselines, four traces, 120 interventions, and one completion; there is no failure row. Every baseline has exactly one final EOS. Each saved arm records one valid hook call and finite CPU float32 state checks.
- I independently rehashed all 56 baseline state files and 56 trace sidecars, and recomputed every per-position L2 and maximum-absolute trace from the paired captured tensors within float32 tolerance. The 112 files total 118,670,336 bytes. SHA-256 of `filename<TAB>sha256<LF>` concatenated in lexicographic basename order is `5387fe0d738169f17e022f56777e3d3600741a90f7890da2ac284bc7a6226efc`.
- I recomputed all 32 random-arm seeds from the plan's SHA-256 derivation, regenerated every CPU Gaussian edit from the saved donor and recipient states, and matched all 32 saved edit hashes and target/realized norms. All seeds and edit hashes are distinct. The completion row reports 18.85 seconds of assay work and 940,834,816 bytes peak RSS, within the 30-minute and 4-GiB limits; these are runner-reported quantities, not separate host telemetry.

## Complete-transcript result

For every arm, I independently normalized the **full** saved transcript, replaced only the recipient word in its own baseline, and required one final EOS, one donor word, no recipient word, and all other words unchanged. My results match every saved primary flag.

| Patch region | After encoder block 0 | After final encoder normalization | Other check |
|---|---:|---:|---|
| Donor spoken-word window `W` | 8/8 target-only changes | 8/8 target-only changes | Complete donor token IDs in 15/16 combined arms; one differed only outside normalized words. |
| Donor active-audio complement `C` | 0/8 | 0/8 | All 16 retained the recipient's normalized words; 15/16 retained exact token IDs. |
| Equal-width prefix `P` | 0/8 | 0/8 | All 16 retained the recipient's normalized words. |
| Equal-width postword `Q` | 0/8 | 0/8 | All 16 retained the recipient's normalized words. |
| Recipient self `W` | 0/8 | 0/8 | All 16 matched recipient token IDs exactly. |
| Independent equal-total-norm random `C` | 0/16 | 0/16 | All 32 retained recipient normalized words; 29/32 retained exact token IDs. |
| Full donor final state | — | 8/8 target-only changes | All eight matched donor baseline token IDs exactly. |

The input-locality gate passed for all four pair/voice combinations: donor and recipient post-convolution-plus-position states are bitwise equal in `C` and differ in `W`. Log-mel differences first appear at frame 195 (Piper) or 233 (Kokoro); post-convolution differences first appear at row 97 or 116, all within the frozen `W`. After encoder block 0, the first nonzero difference is at row 0 in every combination, and the sum of `C` row L2 differences is nonzero (42.54–59.57 at that block; 379.01–435.15 at final normalization). Thus the captured state difference spreads beyond the word region, while the tested donor `C` patches alone do not change the complete transcription.

## Interpretation limits

This is a **posthoc exploratory** assay on two previously observed FIT lexical pairs, two synthetic English voice engines, and one tiny ASR checkpoint. The 16 directional outcomes share four pair/voice combinations and are not independent statistical trials. A failed `C`, `P`, or `Q` patch establishes lack of **sufficiency under these exact interventions and contexts**, not absence of lexical information outside `W`: interactions with unchanged `W`, edit scale, decoder path, or other regions may matter. The first post-block state includes attention and feed-forward computation; the study does not identify a single attention head or a unique circuit. The matched-magnitude random edits are useful specificity controls, not a significance distribution.

These patches require internal state access. No attacker-controlled waveform, natural voice, multilingual request, audio-language response/action pathway, or violated security property was tested. The saved-data audit verifies file consistency and independent recomputation; it cannot by itself prove that live hooks executed against the stated unmodified weights.
