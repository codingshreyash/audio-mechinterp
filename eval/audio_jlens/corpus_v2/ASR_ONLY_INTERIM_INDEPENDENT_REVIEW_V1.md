# Independent review of frozen v2 ASR-only screen

**18 November 2026. ASR screen complete; human QA pending.** I reviewed the raw Whisper and Wav2Vec2 runs anchored before comparison in `adb389d` and `fb83bb1`, and the post-run interim script/report committed in `31692ca`. The [interim JSON](asr_runs/ASR_ONLY_INTERIM_V1.json) SHA-256 is `15ad56ea32f9dcdcd518dcef15f8370a9e129e18c53031627bd5fac3a3947ebc`. No ASR model was rerun, and no candidate Qwen or Voxtral inference was part of this review.

I invoked the previously frozen CPU auditor's source, corpus, freeze and raw-run verification functions. They rebuilt **480 unique case identities**, rechecked all 480 WAV and decoded PCM hashes against the frozen manifest, and accepted exactly **480 ordered, reference-free raw transcript rows from each recognizer**. They verified the two exact model revisions, weight/processor/tokenizer file pins, CPU/four-thread decode settings, completion receipts, raw-file SHA-256 values, and the freeze receipt. I recomputed normalized full-script comparison, whole-token target occurrence, WER, model disagreement, and ASR-only second-listener flags for all 480 cases; every aggregate equals the committed interim report.

| Measure | Whisper tiny.en | Wav2Vec2 base 960h |
| --- | ---: | ---: |
| Raw outcomes | 480 | 480 |
| Normalized full-script exact | 269 | 246 |
| Intended word exactly once anywhere | 338 | 370 |
| Word errors / reference words | 645 / 4,992 | 505 / 4,992 |
| WER | 12.92% | 10.12% |
| Empty or truncated outputs | 0 | 0 |

The recognizers disagree on **259/480** normalized transcripts. Under the prespecified [middle-slot amendment](ASR_MIDDLE_SLOT_FLAG_AMENDMENT_V1.md), **285/480** clips require a second listener from ASR evidence alone because at least one normalized full script differs from its reference. Of these 285, Whisper alone mismatches on 51, Wav2Vec2 alone on 74, and both mismatch on 160. A target token appearing elsewhere in a transcript does not clear a full-script mismatch. The first human listener can add further flags; these 285 are neither a final human-review workload nor a count of unintelligible clips.

The Wav2Vec2 checkpoint lacks `masked_spec_embed`, so Transformers initializes that parameter and may warn on load. I confirmed the tensor is absent from the pinned safetensors file. A processor-only check on one second of zeros returned only `input_values`, with no mask indices and no model load. In the pinned `modeling_wav2vec2.py` implementation, the parameter is read only when explicit `mask_time_indices` are supplied or when training-time SpecAugment is active. This runner calls `.eval()` and passes no mask indices. Thus the missing parameter is **inert for these inference outputs**; the warning does not invalidate this ASR screen. This conclusion is specific to the pinned checkpoint, source, and no-mask evaluation path.

The raw complete-receipt SHA-256 values are `4b6b28a64644023b286873d8b98e829f5bea4c5a6a8f677944f07569bac27f4a` (Whisper) and `5d22b63b6fd7020a9dfd5c2654799db2e38b87882e56061fd19efcf1a1c89f27` (Wav2Vec2). The interim report is a bounded **ASR-only** summary. It contains **zero human primary responses**, no adjudication, and no independently sealed 480/480 quality receipt. ASR agreement cannot establish human intelligibility. The candidate-model gate remains closed until the full blinded listening and independent quality audit in the frozen protocol are complete.
