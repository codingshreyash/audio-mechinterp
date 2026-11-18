# Frozen semantic-bridge item manifest, version 1

Built 2026-11-18 from [`anthropics/jacobian-lens`](https://github.com/anthropics/jacobian-lens) commit `581d398613e5602a5af361e1c34d3a92ea82ba8e`. This is an **item and split freeze**, not an audio rendering or model result.

## Artifacts

- `manifest.jsonl`: 60 selected items, with source identity/hash, split, typed raw-completion prompt, prospective spoken script, expected intermediate labels, and source next-text answer where supplied.
- `smoke.jsonl`: 12 **fit-only** items, six per category, for input/output plumbing checks. It must not be reported as held-out performance.
- `metadata.json`: source hashes, deterministic selection and split rules, IDs, counts, smoke IDs, manifest hashes, and template-group assignments.
- `build_manifest.py`: standard-library builder. An existing artifact with changed bytes causes an error; changes require a new manifest version and explanation.

The source-file hashes are:

| Source | Items | SHA-256 |
|---|---:|---|
| `lens-eval-association.json` | 102 | `d1a98cd4911b594282e74168091c77d849dae18ffe2acb5761074853f327d71c` |
| `lens-eval-multihop.json` | 93 | `50b7e4c9255291c0ca2a8e94615be9f44531fa57bb1a844e4f9616056d987416` |

Manifest SHA-256: `035a12e0388f0fb1c29dc2762c8c070d8052f64202c05c5198b6230d25d2ccf3`.

## Rules fixed before model evaluation

1. For each source category, sort eligible items by SHA-256 of the UTF-8 source `name`, then take the first 30. The association source contains nine items marked `fr-`, `es-`, `de-`, or `pt-`; exclude them for this initial English pilot. The multihop source has no such marked items. This is a documented pre-fit amendment to the earlier broad selection rule in `../INDEPENDENT_REVIEW.md`.
2. Assign exactly 18 fit, 6 development, and 6 test items in each category. Multihop items with the same `name` prefix before the first hyphen stay together, reducing obvious template reuse across splits. Association items are individually grouped. A deterministic hash-priority backtracking assignment fills the exact capacities.
3. The paired **typed** input is the exact source `prompt` in raw-completion form. The prospective **audio** script is the same characters with trailing whitespace removed. Trailing spaces have no sound; punctuation and all other text are preserved. The target answer is **not** appended to either input. `expected_bridge_labels` are source `intermediates`. Multihop's source `target` is stored as `expected_next_text`; association has no source next-answer label.
4. The readout location is the final prompt token for the typed branch. An eventual audio model needs a separately specified acoustic time rule before scoring. The manifest deliberately contains no encoder-frame or token-alignment claim.
5. The smoke subset is the first six fit items per category under the same name hash. It contains no development or test item.

Validation found 60 unique IDs and 60 unique prompts, zero shared template groups across splits, zero fit/dev/test ID overlaps, and zero marked non-English association items in the selection.

## Rebuild and check

```sh
cd /Users/shrranjan/audio-mechinterp/eval/semantic_bridge
python3 build_manifest.py
shasum -a 256 manifest.jsonl smoke.jsonl
```

The builder checks source hashes and checks that existing outputs are byte-identical. It uses no model, MPS, external service, or audio synthesis.

## Gaps before an audio experiment

- Pin the exact audio model, tokenizer, text/audio input adapters, sample rate, and all TTS engines/voices/versions **before rendering**. A model chat template may be required; if so, define it as a versioned adapter applied equally to both branches where possible. Keep the raw source prompt in this manifest.
- Render audio, record waveform SHA-256 and synthesis settings, and freeze any human speech recordings before scoring. Current `rendered_audio_sha256: null` is intentional.
- Check independently whether listeners hear the incomplete multihop prompts as intended. A TTS system may speak trailing quotes or incomplete clauses differently from text tokenization. Exclusions or rewrites require a new manifest version before held-out evaluation.
- Freeze candidate token surfaces, scored layers, acoustic regions, readout estimator, and exact metric from [`../PROTOCOL.md`](../PROTOCOL.md) before reading development or test results. This manifest fixes stimuli and labels, not those model-dependent details.
- Item disjointness and coarse name-prefix groups do not guarantee all semantic patterns are unrelated. The large lexical overlap between natural-language prompts remains; report this when interpreting transfer or generalization.
