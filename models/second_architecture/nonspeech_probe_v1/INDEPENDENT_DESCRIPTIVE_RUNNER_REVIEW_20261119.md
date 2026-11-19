# Independent descriptive runner source review

19 November 2026. **HOLD for candidate-model inference.** This is a no-weight source review of runner commit `19610e1cb6656e3c514be0833c602635e23e3f92` against prospective protocol `bb0a5bd55e63c396c5408fd9714d1d528c9c9792`, independent source review `72c627f0a23291d17d485abc5fbf7e37ce95965f`, and render review `def72446227fb562141bbcd11596c7e2de494a92`. I did not deserialize or run any of the three candidate checkpoints. This memo is deliberately **not** the `nonspeech_descriptive_source_review_v1` JSON with `status=GO` required by the parent and worker.

## Exact byte binding

| Artifact | SHA-256 |
| --- | --- |
| Frozen render manifest | `a37e6e6027611555cd65ceb1a8d9288b79c72460ae1155469eb1f094d7ddee90` |
| Source-only preflight receipt | `7277fef8e0660e9b02925401df8e55e2b49aac57082e80559c756cb4715ba353` |
| `descriptive_worker_v1.py` | `fe5a9dd81569882017842e01622896269d4ec678b137fbc85de8c399100433ad` |
| `descriptive_parent_v1.py` | `49c6d536d9199b0271212e8fdfeb2dd467db9fe99139efa811f653437893bde6` |
| `descriptive_audit_v1.py` | `77c32864cc8504f7c1d6f5137d90ebe3592f7cdd70e4eb0e62f218d18a21fbeb` |
| `descriptive_pins_v1.json` | `b54f99df30290f71eab94c3f9e5cafec47d6a3062b5b7cf68ff4d227d4110284` |
| `test_descriptive_v1.py` | `124832c11203ecfdf55157380be8b2b4a06f56d89d84c1de6bc7d5e43295b08d` |
| Prospective protocol | `2815d24b7f05aea778aec71c5a76746f952eca92a680d640c3544ab57d832b32` |
| Frozen `spec.json` | `2dd69b687b00d90ed4e2d8bbc266039943498c31532634440f2d3bd0a0ca0951` |
| Frozen `render.py` | `8c20404965c30b11c347e562f1bb8b399e433c58e64d8674f62dab98dcf98cb1` |

The preflight JSON exactly equals a fresh `preflight_receipt()` calculation. All three local weight files were **hashed only**, with their pinned metadata/config/tokenizer and installed source files, under NumPy 2.3.5, Torch 2.15.0, and Transformers 5.20.2. The checkpoint identities are:

| Model | Snapshot revision | Weight bytes | Weight SHA-256 |
| --- | --- | ---: | --- |
| `openai/whisper-tiny` | `169d4a4341b33bc18d8881c4b69c2e104e1cc0af` | 151,061,672 | `7ebd0e69e78190ffe1438491fa05cc1f5c1aa3a4c4db3bc1723adbb551ea2395` |
| `openai/whisper-tiny.en` | `87c7102498dcde7456f24cfd30239ca606ed9063` | 151,060,136 | `db59695928ded6043adaef491a53ef4e12da9611184d77c53baa691a60b958ad` |
| `facebook/wav2vec2-base-960h` | `22aad52d435eb6dbaf354bdad9b0da84ce7d6156` | 377,607,901 | `8aa76ab2243c81747a1f832954586bc566090c83a0ac167df6f31f0fa917d74a` |

## Verified contracts

`trusted_cases()` verifies all 20 WAV and PCM bytes and sorts lexicographically; the plan appends the exact 2-second and 8-second silence rows. The source uses one CPU FP32 model load per fresh attempt, with no automatic retry. The parent enforces a 1,800-second wall cap, sampled 4-GiB process-tree RSS cap, 128-MiB evidence cap, process-group kill, and the pinned Mac AC/open-lid/power-log gate. The existing seven no-weight tests pass. Separate fake `/bin/sleep` runs hit both `rss_cap` and `evidence_cap`, as well as the committed wall-cap test. Sampled RSS cannot prove no sub-250-ms peak; the handoff discloses that limit.

The installed Whisper 5.20.2 source performs multilingual `detect_language()` in a separate forward, and the worker records its 99 language logits and chosen token separately from the transcription `scores` and selected prefix. `score_whisper()` uses the actual score tuple length to separate prefix from generated tokens; it does not infer EOS from a trailing pad token alone. The Wav2Vec2 code saves all CTC frame IDs, a logits hash, and collapsed IDs. The installed Whisper and Wav2Vec2 encoder-layer hooks return tensors at the selected sites. The worker asserts one hook call at each declared stage in each observed pass and writes explicit failure/not-run rows if an exception occurs after its start record. These are source-path findings, not successful candidate-model forward tests.

## HOLD reasons and minimal repair

1. **Active and padded geometry is not saved.** The frozen protocol, “Frozen model observations,” requires “input geometry and padding masks so a 30-second Whisper padded row is not mistaken for active sound.” `infer_one()` in `descriptive_worker_v1.py` calls `processor(...).input_features`, records only the 1×80×3000 mel tensor, and saves the original PCM frame count. It never requests or records the feature extractor's `attention_mask` or an explicit active/padded mel-frame boundary. A reviewer could derive an approximate boundary from the pinned processor and PCM count, but the promised exact processed mask is absent and cannot be validated against the observed mel rows. Request the processor's attention mask, save and validate its 3,000-frame contents/active count alongside sample count and mel shape, and declare the encoder time-axis mapping. For unpadded Wav2Vec2, explicitly save the all-active geometry and CTC frame count. Add fake tests for 2-second versus 8-second masks.

2. **The promised first decoder-step state and raw text logits are absent.** The same protocol requires “first decoder step and subsequent text logits.” `Capture.install()` hooks only Whisper encoder stages; `infer_one()` requests `output_scores=True`, and `score_whisper()` saves processed generation scores and their hashes. In Transformers generation, `scores` are post-logits-processor scores, not necessarily the model's raw decoder logits. Save a declared first-decoder-step activation (or explicit model output hidden state) and raw generated logits, while keeping processed token scores separately labeled. Assert their step/shape alignment and bound their evidence size. A fake test should distinguish raw logits from suppressed processed scores.

3. **The saved-record audit does not validate hook counts or the claimed metric geometry.** `descriptive_audit_v1.py` checks stage-name coverage but never reads `capture_counts`. Its `check_trace()` accepts any 64-character string as a “SHA-256” and any time axis with a matching curve length. A no-weight adversarial call accepted a full Wav2Vec2 trace of fabricated 1×2×3 shapes, all-zero curves, and `"z"*64` hashes. Validate exact per-phase hook-count dictionaries; assert stage-specific ranks, time axes, expected sample/mel/CTC lengths, hash syntax, and saved masks. Check row/difference identities where possible. The audit remains structural because raw tensors are not saved, but it should reject internally impossible records.

4. **The saved language-choice check is incomplete.** `check_whisper()` requires 99 language-score entries and a prefix equal to the stated chosen ID, but does not verify that the chosen ID corresponds to the maximum of those scores. A fake record with arbitrary 99 scores and an inconsistent chosen token passed. Save the language-token-to-ID map or bind it to the pinned generation config and verify the selected token against the stored detection logits (including tie behavior). This does not replace replay of the model forward.

The structural auditor also allows extra record types because it checks only a minimum record count and selected row sets. Require the exact inventory of one start, 22 ordered case rows, 18 silence-difference rows, and one completion record, allowing only the planned interleaving. The existing fake-success test bypasses `infer_one()` and writes incomplete CTC output/traces that the saved auditor would reject; add an end-to-end fake record book that actually passes `audit_attempt()`, plus mutations for each field above.

## Decision

**HOLD** this exact source commit for model inference. Preserve the current prospective protocol and 20 audited WAVs. Repair and independently review a new runner commit, regenerate its source-only preflight receipt, and only then commit a bound `status=GO` JSON. No causal intervention or security claim is reviewed here.
