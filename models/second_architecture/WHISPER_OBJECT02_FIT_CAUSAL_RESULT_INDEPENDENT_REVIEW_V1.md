# Independent result review: Whisper object-word FIT causal replication

18 November 2026. Review target: `runs/whisper_object02_fit_causal_v1/20261118T2301Z`, under the [prospective object-word plan](WHISPER_OBJECT02_FIT_CAUSAL_REPLICATION_PROSPECTIVE_V1.md), its [independent plan review](WHISPER_OBJECT02_FIT_CAUSAL_REPLICATION_INDEPENDENT_PLAN_REVIEW_V1.md), and the [runner/auditor source GO](WHISPER_OBJECT02_FIT_CAUSAL_INDEPENDENT_SOURCE_REREVIEW_V2.md). I did **not** load or run Whisper, modify raw results, inspect corpus v2 or a locked test, or select any intervention from the observed outcomes.

**Decision: GO for saved-data consistency and a FIT-only replication of the local encoder word-window effect.** Across a different lexical category (`plate`/`cup`), two synthetic voices, both directions, and three prespecified encoder sites, all 12 donor word-window patches changed the complete EOS-terminated transcript by the target word alone. This remains an internal ASR intervention, not an attacker-controlled waveform exploit or an agent security finding. Saved files cannot independently prove that the hooks ran in an unmodified model; that evidence rests on the frozen source, preflight, and raw run together.

## Identity and checks

- Frozen no-weight preflight SHA-256: `49016c27dd8c4104e9494a74b761c35d11363c4fe86ddb4893f99cf1f060f0e7`. It pins the four FIT WAV/PCM inputs, prior ASR screen, `openai/whisper-tiny.en` revision `87c7102498dcde7456f24cfd30239ca606ed9063`, model and tokenizer assets, installed source files, CPU/FP32 generation settings, runner source commit, and eight derived random-arm seeds. The plan/source reviews and this receipt were committed before the live attempt.
- Raw `records.jsonl` SHA-256: `b41a66113adc5ebc6adf4bf7807290b8a8c838a2f02cf3f96058b2437665e5bc`. Original `audit.json` SHA-256: `8e0e686e888322a937b0ed98b79c4373ac83d11a2824a09dd7ebd30fe42b9f08`. I reran the independently reviewed saved-data auditor to `/tmp/whisper_object02_independent_result_review_20261118T2301Z.json`; the two audit JSON files were byte-identical, with status `independent_saved_data_audit_pass`.
- The 56 raw rows comprise one preflight, four baselines, two time traces, 48 distinct planned intervention arms, and one completion row, with no failure row. I independently checked each baseline and intervention state reference, every tensor sidecar hash, all 28 trace L2/max-absolute sidecar hashes, and the absence of extra capture files. The 56 capture files total **59,335,168 bytes**. SHA-256 of the concatenation of `filename<TAB>sha256<LF>` for all capture files in lexicographic filename order is `9517e2c0d15674296cc948c559111d00109de4105645b062cd3d5c77576293e2`.
- All four baseline transcripts contain the single intended word, exclude its paired alternative, and have exactly one final EOS token. The completion row reports **7.42 seconds** of assay time and **865,255,424 bytes** peak RSS, within the frozen 30-minute/4-GiB caps. These resource measurements are runner reports, not separate host telemetry.

## Complete-transcript outcomes

I independently normalized every raw complete transcript, replaced only the recipient target word in its own baseline, and required the donor exactly once, the recipient absent, all other normalized words unchanged, and exactly one final EOS. Those recomputed outcomes agree with every saved primary flag.

| Arm | Target-only complete-transcript replacements | Other control observation |
|---|---:|---|
| Donor word-window patch at encoder-block-0 input after convolution and position | **4/4** | Both directions in Piper and Kokoro. |
| Donor word-window patch after encoder block 1 | **4/4** | Both directions in Piper and Kokoro. |
| Donor word-window patch after final encoder normalization | **4/4** | Both directions in Piper and Kokoro. |
| Equal-width pre-word prefix patch across the three sites | **0/12** | All 12 retained the recipient's normalized words and exact token IDs. |
| Recipient self patch across the three sites | **0/12** | All 12 matched the recipient's exact token IDs. |
| Full final-state donor patch | **4/4** | All four matched the donor's exact token IDs. |
| Equal-total-norm random word-window patch at the final state | **0/8** | All eight retained the recipient's normalized words; four had different token IDs or punctuation. |

For every random arm, I independently derived its seed from the frozen SHA-256 engine/direction/base-seed string, reconstructed the float32 Gaussian edit from the saved final states, and matched its recorded edit hash and target norm. **All eight seeds and all eight edit hashes are distinct**, resolving the prior `silver`/`teal` run's random-control independence deviation. The maximum absolute saved realized-versus-target norm difference was `3.814697265625e-05`. Eight controls remain too few and correlated to support a distributional significance claim.

The input log-mel differences first appear at frame **195** (Piper) and **233** (Kokoro), within the frozen word windows. At the post-convolution-plus-position encoder input, they first appear at rows **97** and **116**, again within those windows. At the first captured post-block residual, nonzero differences occur from row 0 onward; bidirectional self-attention can spread the effect outside the acoustic interval. This is first **captured numerical divergence** and a time-restricted causal influence assay. It is not evidence of a unique word circuit or a general rule that all prefix states lack word information.

## Inference boundary

This follow-up was planned after the earlier color-pair result was known, so it tests a **new FIT lexical pair** rather than providing a fresh locked-test estimate. Both pairs use the same two TTS engines, phrasing pattern, and small English ASR checkpoint. There is no natural speech, multilingual, audio-language model, speech output, authorization action, or end-to-end security consequence here. The result strengthens the narrow claim that a spoken word's time-local encoder state can causally determine Whisper's complete transcription across these two FIT pairs. It does not validate a reusable Jacobian Lens readout or transfer to another architecture.
