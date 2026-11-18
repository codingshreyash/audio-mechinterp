# Independent result review: Whisper FIT word-window causal assay

18 November 2026. Review target: `runs/whisper_fit_word_window_causal_v1/20261118T2246Z`, under the committed [V1](WHISPER_FIT_WORD_WINDOW_CAUSAL_PROSPECTIVE_V1.md), [V2](WHISPER_FIT_WORD_WINDOW_CAUSAL_PROSPECTIVE_V2.md), and [V3](WHISPER_FIT_WORD_WINDOW_CAUSAL_PROSPECTIVE_V3.md) plans. I did not load or run Whisper. I read the saved JSONL and tensor files, reran the independently reviewed saved-data auditor to a separate `/tmp` file, and separately recomputed the complete-transcript outcomes from raw rows. I did not inspect corpus V2 or the Voxtral locked test.

**Decision: GO for saved-data consistency and the descriptive FIT-only word-window causal observation; HOLD for the preregistered independent-random specificity control.** The random-control implementation reused each seeded noise tensor in the reverse direction, contrary to V2's independent draw per engine and direction. The raw attempt remains unchanged. No generalization, assistant authorization, or security finding follows from this ASR assay.

## Evidence identity and independent checks

- Frozen no-weight preflight SHA-256: `938decc1dba4def95efbebe9158f573011f5bd05a995639606c46f7a3cd9111f`. It pins the four FIT WAV/PCM identities, `openai/whisper-tiny.en` revision `87c7102498dcde7456f24cfd30239ca606ed9063`, checkpoint and snapshot assets, installed source hashes, CPU/FP32 settings, causal sites, and source commit. The source and auditor reviews were GO before the live attempt.
- Raw `records.jsonl` SHA-256: `990eab76bdbb793b46e6ed3e677e202473cfcf97913f1591f32c901b8c04036f`. Original `audit.json` SHA-256: `cad301712e58adf50a18a7e10a2bc725a045538025cb81e2b8e875f2f18ffc97`. My separate auditor rerun produced byte-identical audit JSON with status `independent_saved_data_audit_pass`.
- Fifty-six raw rows are present: one preflight, four baselines, two time traces, 48 unique predeclared intervention arms, and one completion row; no failure row. The 56 capture files total 59,335,168 bytes. SHA-256 of the concatenation of `filename<TAB>sha256<LF>` for those files in lexicographic filename order is `8576a08659e534e38c67019adb8ebe743eb284e97a5940a5140f8dc7ad6d922c`. The saved-data auditor verified each referenced tensor/hash, transcript decoding, arm identity, hook counts and control parity; it cannot prove model execution from saved files alone.
- All four unmodified baselines contain the single intended word, exclude its alternative, and end with EOS. The completion row reports 7.52 seconds of measured assay time and 864,944,128 bytes peak RSS, within the planned 30-minute/4-GiB caps. These are runner-reported quantities, not independent host telemetry.

## Complete-transcript outcomes

I independently normalized each saved complete transcript, replaced only the recipient target word in its own baseline, and required the donor word exactly once, the recipient word absent, all other normalized words unchanged, and EOS. This matches all saved primary-outcome flags.

| Arm | Target-word-only complete-transcript replacements | Other observation |
|---|---:|---|
| Donor word-window patch at post-convolution-plus-position | 4/4 | Both directions in Piper and Kokoro. |
| Donor word-window patch after encoder block 1 | 4/4 | Both directions in Piper and Kokoro. |
| Donor word-window patch after final encoder normalization | 4/4 | Both directions in Piper and Kokoro. |
| Equal-width pre-word prefix patch across the three sites | 0/12 | All 12 retained the recipient's normalized words; 10/12 had identical token IDs. |
| Self patch across the three sites | 0/12 | All 12 had the recipient baseline token IDs. |
| Full final-state donor patch | 4/4 | All four had the donor baseline token IDs, confirming the final-state implementation control. |
| Equal-total-norm Gaussian word-window patch at final state | 0/8 | Seven retained the recipient's normalized words; one Kokoro silver case lost the target and rest of the question while still ending with EOS. |

The input log-mel differences first appear at frame 195 (Piper) and 233 (Kokoro), within the frozen word windows. The post-convolution-plus-position differences first appear at encoder row 97 and 116, again within those windows. At the first captured encoder block output, differences appear from row 0 onward, consistent with bidirectional attention spreading the signal. This is a **first captured numerical divergence** and an intervention-location result, not proof of a unique word circuit or that early prefix rows independently carry a usable word representation. The observed prefix patches did not replace the word.

## Protocol deviation and limits

V2 requested an independent standard-normal tensor for each engine, direction, and seed. The runner reinitialized `torch.Generator` with the same fixed seed for the reverse direction. The saved random-edit hashes are identical for silver→teal and teal→silver at a given engine/seed. Thus the eight random arms used **four distinct noise tensors**. Their norms were matched to the donor-minus-recipient word-window Frobenius norms and all eight lacked a donor-word replacement, but the planned independent-random specificity control was not executed as written. The one short Kokoro output also shows a nonspecific edit can disrupt a complete transcription. Preserve this as a protocol deviation; a corrected, separately planned FIT replication should derive a distinct seed for each engine, direction, and arm, then freeze that mapping before running.

This one already-inspected `silver`/`teal` FIT contrast uses two synthetic voices and one tiny English ASR checkpoint. The word-window patch is an internal intervention, not an attacker-controlled waveform. The study has no fresh-word, natural-voice, multilingual, audio-language, speech-output, action-path, or downstream security validation. A positive complete-transcript patch demonstrates causal sufficiency for this controlled ASR task only. The earlier Whisper Jacobian readout failure remains a separate result; this experiment does not repair or overturn its held-out evaluation.
