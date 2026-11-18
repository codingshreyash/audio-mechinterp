# Frozen Qwen2.5-Omni bridge scoring, version 1

Created 2026-11-18 from the 60-item source [manifest](manifest.jsonl), **before any candidate bridge readout was scored**. A one-prompt fit had already started in parallel; this is pre-readout, not a claim that the scoring rules preceded every fit computation. No model weights, activations, audio, or MPS were used to freeze token candidates. The [scoring module](scoring.py) accepts logits supplied later by the model runner.

## Identity and frozen files

- Model and tokenizer: [`Qwen/Qwen2.5-Omni-3B`](https://huggingface.co/Qwen/Qwen2.5-Omni-3B), revision `f75b40e3da2003cdd6e1829b1f420ca70797c34e`; pinned `Qwen2Tokenizer` under Transformers `5.24.0.dev0` from the local adapter environment. `tokenizer.json` SHA-256 `8441917e39ae0244e06d704b95b3124795cec478e297f9afac39ba670d7e9d99`; chat template SHA-256 and other identity fields are in [scoring_metadata.json](scoring_metadata.json). Text LM head has 151,936 rows; tokenizer has 151,643 base tokens and 151,665 including added tokens.
- Source 60-item manifest SHA-256 `035a12e0388f0fb1c29dc2762c8c070d8052f64202c05c5198b6230d25d2ccf3`.
- [scorable_manifest.jsonl](scorable_manifest.jsonl): **57 items**, SHA-256 `926e53ed66a9f5162e1981a786cd2e51231a273a5ce4b1713c3dea7fd10ac1a8`. It contains every scored label's exact accepted surface strings and token IDs, plus source item/split/prompt hashes. It does not contain activations or results.
- [exclusions.json](exclusions.json): three item exclusions. The original 60-item source manifest remains unchanged.
- [layer_policy_v2.json](layer_policy_v2.json) and [framing_gate_v2.json](framing_gate_v2.json): the current **pre-dev/test** continuation policy. The v1 files remain for audit. V2 was informed by the first two fit-only full-output pilots and is **not preregistered before those pilots**.

Recreate or verify the frozen token files using only the cached tokenizer:

```sh
cd /Users/shrranjan/audio-mechinterp
models/qwen_omni/.venv/bin/python eval/semantic_bridge/freeze_scoring.py
cd eval/semantic_bridge
python3 -m unittest -v test_scoring.py
```

`freeze_scoring.py` compares source and tokenizer hashes, then refuses to overwrite changed frozen artifacts. The tests check tie/rank behavior, min-over-layer scoring, and paired audio/typed aggregation.

## Exact label and rank rule

For each source bridge label, try its original, lowercase, and capitalized spellings, each bare and with one leading space. Preserve only surfaces that encode to **exactly one** tokenizer ID below the 151,936-row head width, are not special/added IDs, and decode to at least one alphanumeric character. No semantic synonyms are invented. Multiple source labels within one item remain separate labels, as in the released Anthropic data. The candidate set for a label is the union of accepted IDs; the best candidate determines its rank.

Primary full-vocabulary rank at one layer is `1 + count(logit[i] > max(logit[candidate_ids]))` over **all 151,936 head rows**. Tied scores share a rank. For each label, take the minimum rank over the **prespecified** post-block Thinker text layers **0 and 17** at the terminal chat position, immediately before the first assistant reply token. Audio and typed inputs must use the same pinned checkpoint, exact system instruction, chat template, layer set, and rank rule. Only user content differs. This is a text-residual readout of an audio-conditioned model; it does not locate an audio-encoder frame or measure the Talker speech path.

The secondary `semantic_rank` uses the same scores but limits competition to 141,600 tokenizer IDs whose decoded string contains a letter/digit and is neither special nor added. Its exact little-endian ID-list hash is in `scoring_metadata.json`. The primary metric remains full-vocabulary rank, including padded head rows, to stay close to Anthropic's released rank convention. Report both; do not switch after seeing results.

For `k ∈ {1, 5, 10, 50, 100}`, each item's pass@k is the fraction of its scorable source labels with min-over-layer rank ≤ k. Category pass@k is the arithmetic mean of item fractions, separately for association and multihop and for fit/development/test. The item, not the token, is the unit. Retain per-label/per-layer ranks, not only the best layer. No model-correctness or answer-quality filtering is applied to this readout metric after the fact; output validity is reported separately through the framing gate.

For each method and source engine (Piper, Kokoro, later human speech), pair an audio item with its exact typed item ID and identical labels/layers. Report mean **audio pass@k minus typed pass@k**, separately by category, with a fixed-seed 10,000-replicate percentile interval bootstrapped over items. `scoring.py` implements `score_item`, `aggregate_scores`, and `paired_audio_typed`. Methods must be analyzed separately.

## Exclusions fixed before readout

| Item | Split | Reason |
|---|---|---|
| `multihop:super-populous-capital` | fit | Source says China→Beijing for the currently most populous country. [UN DESA](https://www.un.org/development/desa/pd/content/india-overtakes-china-world%E2%80%99s-most-populous-country) reports India overtook China in 2023. The source bridge/answer are factually outdated. |
| `multihop:nhop-clock-element` | dev | Its sole bridge label `12` encodes as two Qwen tokens, so the single-token rank is undefined. |
| `multihop:nhop-chess-element` | dev | Its sole bridge label `16` encodes as two Qwen tokens. |

Scorable counts are association **18 fit / 6 dev / 6 test** and multihop **17 fit / 4 dev / 6 test**, total 57 items and 61 scorable labels. Do not silently backfill excluded items or change denominators. A future multi-token label metric would be a separately versioned analysis.

## Readout baselines and validity gate

The primary candidate is an audio-conditioned Jacobian readout at the frozen terminal text position. Compare it with a **raw logit lens** from the same layers and final norm/head, using identical candidate IDs and ranks. Report the actual final Thinker next-token distribution separately as a predictive reference, never as a latent-bridge lens or a causal explanation. Add a within-split shuffled-audio null with a fixed permutation and matched length/engine when an audio readout is ready; keep all shuffles inside fit, dev, or test independently. A probe or audio-encoder readout uses a separately frozen task-appropriate metric; it must not be mislabeled as the same vocabulary-rank lens.

The released association prompts are **vignettes without an explicit question**; multihop prompts are often **unfinished clauses**. When Omni receives them in a chat user turn, the chat template and system instruction alter the original raw-completion task. Therefore source bridge labels remain hypotheses for internal readout, but they are **not yet validated Omni answer labels**. The fit-only [task-framing pilot](task_framing_smoke_2item_snapshot.json) generated typed, Piper, and Kokoro continuations for two association items using `Continue the user's text naturally. Do not explain your continuation.` All six outputs were related prose; none explicitly named its bridge word, and Piper/Kokoro largely echoed one vignette. The typed `association:interview` run hit the 80-token cap without EOS, so it is censored rather than a complete answer. These two items cannot pass the 12-item gate.

The current v2 policy keeps that continuation instruction because it better matches the source benchmark. Before a held-out bridge claim, run all 12 **fit-only** [smoke items](smoke.jsonl) in typed, Piper, and Kokoro routes, retain full generated outputs and EOS status, and apply the frozen [v2 judging rules](framing_gate_v2.json). A token-capped output cannot establish a complete-answer judgment; increase the cap only under a versioned v3 policy before further evaluation. If association or multihop fails its fit-only gate, report dev/test source-label readouts only as exploratory diagnostics and define a new explicit task prompt before a fresh locked test. The first two fit-only outputs informed v2 and must not be represented as a pre-registered success.
