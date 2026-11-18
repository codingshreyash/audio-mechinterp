# Independent ASR screen of frozen corpus_v1

**18 November 2026.** The [protocol, script, and schema](asr_audit.PROTOCOL.md) were committed as `24f18c4` before this CPU-only ASR run. I checked the output against the frozen JSON schema and counted all 384 manifest clips, with no filtering, replacement, or candidate Qwen/Voxtral inference. The full [JSON result](asr_audit.json) has SHA-256 `41da6ee4dd67a2f7b06ece24354fdb0ca6911f7017ec514e170f55f61d38a5e7`; its recorded script SHA-256 is `283245a8343f7ff992b286d83795d5af02fe07aa3e459d9ac73dc9f9fb2c23b9` and manifest SHA-256 is `adf8cd02d24a43ef576bcb993dc54bd2527440a62798e12fc788def3c063971f`.

The local `openai/whisper-tiny.en` checkpoint revision `87c7102498dcde7456f24cfd30239ca606ed9063` was used with CPU, four threads, greedy one-beam decoding, and no network download. Its resolved safetensors SHA-256 was `db59695928ded6043adaef491a53ef4e12da9611184d77c53baa691a60b958ad`. Every WAV and int16 PCM hash matched the frozen render manifest. The ASR model saw only the audio; reference scripts were used after decoding.

| Split | Clips | Normalized full transcript exact | Mismatches | Intended word exactly once | WER |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fit pool | 192 | 164 | 28 | 164 | 2.25% |
| Development | 96 | 86 | 10 | 86 | 2.26% |
| Locked test | 96 | 90 | 6 | 90 | 1.09% |
| **All** | **384** | **340** | **44** | **340** | **1.96%** |

The paired alternative word appeared in **0/384** transcripts. All 44 normalized transcript mismatches also lacked an exact occurrence of the intended word; the complete raw transcripts, references, word-error counts, and split/engine metadata are in the JSON. The errors affected 13 fit word pairs, 4 development pairs, and 3 locked-test pairs. Breakdown by engine: fit Piper 85/96 exact and Kokoro 79/96; development Piper 47/48 and Kokoro 39/48; locked test Piper 45/48 and Kokoro 45/48.

The six locked-test mismatches are retained unchanged:

| Engine / pair | Intended word | ASR transcript excerpt |
| --- | --- | --- |
| Piper / `color_11`, both templates | `tan` | `10` |
| Piper / `object_03`, attention template | `bowl` | `Ball` |
| Kokoro / `object_03`, answer template | `bowl` | Omitted after “Here is my answer word.” |
| Kokoro / `nature_07`, both templates | `marsh` | `Mars` |

Development mismatches include Kokoro `rabbit`, `fox`, `chair`, `mirror`, and `clock`; some transcripts omit the word and the following question, while others render plausible phonetic neighbors such as `fox`→`Vox`, `chair`→`Share`, and `clock`→`Cluck`. Fit has similar phonetic and omission cases, all preserved in the result. The corpus remains a **prospective fixed set**. These ASR outcomes are a quality screen and candidates for blinded human listening, not a basis for editing or selecting evaluation clips. One small ASR model can mishear intelligible speech, and exact-match counting treats orthographic homophones such as `tan`→`10` as mismatches. Conversely, ASR agreement does not establish human perceptual equivalence.
