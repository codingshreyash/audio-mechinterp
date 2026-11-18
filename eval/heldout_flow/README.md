# Held-out audio information-flow set, version 1

This is a **small, locked transfer check** for the fit-only Qwen2.5-Omni-3B color-word information-flow result. The [plan](plan_v1.json) was written before rendering and before any candidate-model answer on these new clips. The [renderer](render_v1.py) loads only pinned CPU TTS engines and writes the [stimulus manifest](stimuli_v1/manifest.json); it never loads Qwen or ASR. The stimuli are ready for an evaluator to consume without changing the selection rules.

## Frozen design

| Axis | Values |
| --- | --- |
| Word contrasts | `cat`/`dog`; `Monday`/`Friday` |
| Spoken phrasings | “The word I chose is … Tell me the word I chose.”; “Please remember this word: … What was that word?” |
| Independent TTS families | Piper VITS `en_US-amy-medium`; Kokoro `af_bella` |
| Count | 2 contrasts × 2 phrasings × 2 families = 8 matched pairs, 16 clips |

Both voices and every word/phrasing differ from the earlier Piper Lessac and Kokoro Sarah `red`/`blue` fit-only pilot. Piper Amy and Kokoro Bella are **new voices**, although the two underlying TTS families were used before. The word lists and wording were selected without candidate-model or ASR results. The renderer records the plan hash, model asset hashes, TTS package versions, fragment WAV hashes, processed WAV hashes, clip hashes, word-slot sample boundaries, and exact scripts. It uses seed `20261118` as an identifier; the selection code makes no random draw, while engine-level determinism is checked by comparing output hashes if rerendered. Piper and Kokoro models, package versions, preprocessing, 120 ms gaps, PCM_16 output, and right padding are fixed in the plan.

Within each engine/pair/phrasing, the prefix and suffix come from the **same** recorded PCM arrays. Only the padded word slot changes. A separate post-render check opened all 16 final WAVs as int16 PCM and confirmed equal lengths and sample-for-sample equality outside each slot for all eight pairs. Every inside-slot comparison differs. The manifest records the exact slot sample indices; these are splice intervals, **not** independent phoneme boundaries.

## Evaluation registered before model answers

The primary question is whether the donor answer flips after a post-block layer-30 patch at nominal word audio-placeholder positions more often than after an otherwise identical post-block layer-31 patch. Evaluate both donor directions for every pair, giving 16 directional cases at each of layers 29, 30, 31, 32, and 35. Report the complete greedy answer and end-of-turn for every clean and patched case. The primary score requires the whole stripped answer to equal the donor word, case-insensitively, allowing only one final punctuation mark, **and** normal end-of-turn. Report all planned cases and a separate clean-eligible subset where both unedited clips answer their own word correctly. Do not silently discard failures or change the answer parser after seeing outputs. [Exact plan](plan_v1.json).

Controls fixed in the plan: self-patch, equal-width wrong-time patch with edit norms, equal-norm random direction at the word rows for seeds 0/1/2, terminal-position donor patch at layers 31 and 35, and cached-versus-uncached prefill activation comparison. The first-token margin is secondary and only meaningful with verified answer tokenization. Hook invocation, edit norms, malformed output, and no-EOS rates are recorded. A positive donor flip is a causal sufficiency observation for that patch, not proof that normal inference uses one unique circuit.

## Reproduction and artifact identity

```bash
/Users/shrranjan/audio-mechinterp/eval/tts/.venv/bin/python -m py_compile /Users/shrranjan/audio-mechinterp/eval/heldout_flow/render_v1.py
/Users/shrranjan/audio-mechinterp/eval/tts/.venv/bin/python /Users/shrranjan/audio-mechinterp/eval/heldout_flow/render_v1.py
```

The second command intentionally refuses to overwrite a nonempty output directory. It has already been run once with the pinned local assets and wrote `stimuli_v1/`. SHA-256 values at freeze time:

| Artifact | SHA-256 |
| --- | --- |
| `plan_v1.json` | `13cea31e2e4843d70a6ab26b060812f43691147435c1e10ea9ebaaa0b24efc78` |
| `render_v1.py` | `bbbd7931dd93121d32b32a9c7898d0a14a3e884f7a421a01d4ecdfd3bd3d063c` |
| `stimuli_v1/manifest.json` | `ae7cfd2e08a09e44847bd7a2e0d5775b8017a6f5e072f3f5d3dc007b9f759a01` |

To reproduce the render separately, pass `--output-dir` pointing to an **empty** new directory. Compare its manifest's source hashes and each final clip's SHA-256 or decoded PCM hashes. Platform-specific ONNX behavior may change byte hashes, so treat such a mismatch as a reproducibility finding and compare decoded PCM before accepting results. The frozen `stimuli_v1/` files are the evaluation input.

## Limits

This is 8 pairs from only two lexical contrasts, not a publication-sized generalization set. Both sources are synthetic and their separately rendered fragments create unnatural prosody or joins. Word slots include TTS silence and right padding; mel features can differ outside nominal slot because of windowing, and post-attention states can mix context. The 30→31 contrast came from the fit-only color pilot and therefore tests transfer of an exploratory boundary. There are no human voices, language variation, complete speech output, or security decisions here. Those require separate, independently frozen evaluations. **No Qwen answers or patch results for these held-out clips had been inspected when this artifact was prepared.**
