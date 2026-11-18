# Qwen audio transport v2: fresh corpus candidate proposal

**UNFROZEN, CPU-only preparation.** This folder contains proposed words and templates plus a validator. It has produced **no WAVs, ASR output, Qwen answer, activation, score, or model fit**. The [v2 research design](../PROSPECTIVE_AUDIO_LENS_V2_DESIGN.md) is also a proposal. Review and freeze an executable corpus protocol before rendering or using any v2 model output.

## Candidate universe and firewall

[`CANDIDATES_UNFROZEN.json`](CANDIDATES_UNFROZEN.json) contains 15 primary pairs and three ordered reserve pairs in each of animal, color, nature, and object: **60 primary pairs, 120 primary words**, plus **12 reserves, 24 reserve words**. Pair and reserve words are globally unique. The validator reads only the word-selection fields of the prior [48-pair v1 spec](../corpus_v1/SPEC_V1.json), the earlier [held-out plan](../../heldout_flow/plan_v1.json), and the earlier test-stimuli manifest's `selection.pairs`. That includes the old v1 locked-test **word metadata**; it does not open old locked WAVs or model outputs. The prior set has 106 distinct words including the old red/blue and other pilots. Candidate words must be disjoint from that set and from deterministic simple-inflection roots. The suffix rule screens `s/es/ies/ed/ing` variants; it is not a complete English morphological analyzer and needs a final human lexical review before freeze.

For pronunciation, the validator requires the pinned [CMU Pronouncing Dictionary](https://github.com/cmusphinx/cmudict) at commit `74790861f652b15e4ac49015a90074ad62a27690`, `cmudict.dict` SHA-256 `81917843c7f44ce2b094ac63873c2c7a4cf802040792c455ba3ca406891c3d22`. It removes stress digits and rejects any identical pronunciation among all 144 proposed words or between a proposed and a prior word. This is a US English dictionary screen, not proof of how either TTS voice or a person will pronounce a word. The validator also loads only the pinned local Qwen2.5-Omni tokenizer, checks its byte hashes and EOS ID, and rejects overlap between the two words' first-token ID sets for four fixed surfaces: lower/title case, with/without an initial space. It never loads Qwen weights. Multitoken words are allowed; broad first-token prefixes may still fail the later complete-answer coverage/purity gate.

Run the CPU check from the repository root, with a locally pinned dictionary checkout:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  models/qwen_omni/.venv/bin/python \
  eval/audio_jlens/corpus_v2/validate_candidates_unfrozen.py \
  --cmudict /path/to/cmudict.dict \
  --report-out eval/audio_jlens/corpus_v2/CANDIDATE_VALIDATION_UNFROZEN.json
```

The `--report-out` path must not already exist. The current [CPU report](CANDIDATE_VALIDATION_UNFROZEN.json) passed with **144/144** candidate/reserve words distinct and dictionary covered, zero detected homophone or prior-word collisions, and zero within-pair tokenizer-prefix collisions. This report checks lexical feasibility only; no category's model-answer or TTS quality has been measured. If a category cannot sustain 15 viable pairs under the prespecified rules, stop instead of relaxing category balance.

## Deterministic split and rendering grammar

Within each category, pair IDs are fixed by listed order as `v2_{category}_{01..15}`. Sort the 15 IDs by lowercase hex SHA-256 of UTF-8 `seed_text/category/pair_id`; take the first nine as FIT, next three as development, final three as locked test. Within each category's nine FIT IDs, take the first four as basis pairs and the last five as map pairs. This yields **36 FIT** (16 basis + 20 map), **12 development**, **12 locked test** whole pairs. The [CPU report](CANDIDATE_VALIDATION_UNFROZEN.json) lists every assignment. No model or ASR result contributes to this split. The 12 listed reserves are **screened but inactive** in this proposal: the validator implements no substitution rule. If any primary pair fails rendering or the pre-inference WAV audit, **stop before Qwen inference** and publish the failure; a revised candidate/spec version needs a separately frozen replacement rule and fresh CPU audit. Do not silently replace a pair or keep a 59-pair subset.

There are two unique templates per split in the candidate JSON. For each of 60 pairs, two words, two pinned independent voice families (Piper Amy and Kokoro Bella), and the two templates belonging to that pair's split, render exactly **60 × 2 × 2 × 2 = 480 WAVs** grouped into **240 matched pair/voice/template records**. Reuse the existing pinned [TTS plan](../../heldout_flow/plan_v1.json) and [CPU fragment/assembly method](../corpus_v1/render.py): render prefix, each word, and suffix fragments separately; process them to mono 16 kHz PCM16; right-pad the two word fragments to the same slot width; reuse identical prefix, 1,920-sample gaps, and suffix PCM for both words. The two final WAVs in a matched record must have the same total frames, bit-identical int16 samples before and after `[word_slot_start_sample, word_slot_end_sample_exclusive)`, and differing PCM inside the slot.

The future render manifest must contain one row per matched record with `split`, `fit_role` when FIT, `pair_id`, `category`, `engine`, pinned voice ID, `template_id`, `words`, exact slot start/end sample, fragment hashes, both output relative paths, WAV SHA-256, int16 PCM SHA-256, frame counts, sample rate, subtype, `outside_slot_pcm_identical`, and `inside_slot_pcm_differs`. It must bind the candidate/spec/renderer/TTS-pin hashes. An independent CPU audit must reopen **all 480 WAVs**, recompute file and PCM hashes, confirm exact paired outside-slot equality, and record nonempty/finite amplitude and durations before any Qwen v2 answer. The candidate phrases and reserve mechanism require a separate frozen renderer/spec review first.

## Limits before interpretation

The v2 primary corpus varies words, split-exclusive phrasing, and two synthetic voices. It does not test new speakers, accents, noise, microphone/channel effects, speaking rate, or human speech. The later acoustic phase in the v2 design is separate. CMUdict and tokenizer checks do not establish intelligibility: an independent ASR transcript should be saved as input-quality evidence, and any human listening must be independent of model outcomes. The category labels are human semantic judgments; in particular, several color words also name materials or foods. Keep these labels and every raw output rather than dropping words after seeing Qwen responses.
