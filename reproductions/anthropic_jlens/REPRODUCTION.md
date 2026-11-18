# Anthropic Jacobian Lens: pinned local reproduction (2026-11-17)

## Scope and provenance

- Reference implementation: [`anthropics/jacobian-lens`](https://github.com/anthropics/jacobian-lens), commit `581d398613e5602a5af361e1c34d3a92ea82ba8e`, checked out at `jacobian-lens/`.
- Model: [`Qwen/Qwen3-1.7B`](https://huggingface.co/Qwen/Qwen3-1.7B), Hugging Face revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`. Local config SHA-256 `1ddb5b89ebc90dcb417a45c213d818577e65976454d29385c8f6140771d95197`.
- Pre-fitted lens: [`neuronpedia/jacobian-lens`](https://huggingface.co/neuronpedia/jacobian-lens), revision `d2527c3cdbaa60dd4b65cbca1604921f044e989a`, file `qwen3-1.7b/jlens/Salesforce-wikitext/Qwen3-1.7B_jacobian_lens.pt`. Lens SHA-256 `6fcc79011bd921ffd87612255e2e99950a124fa519470ee44ebaf161c39be9d6`; width 2048, fitted layers 0–26, 466 fitting prompts. Name, checkpoint family, and width match; the lens file itself does not contain an exact source-model revision hash.
- Released association eval SHA-256 `d1a98cd4911b594282e74168091c77d849dae18ffe2acb5761074853f327d71c` (102 items). Released probe-swap SHA-256 `a0edd27ca23f7b4d0fbe90448c2ddcc7457a3d812121bf024ed12a032ff86796` (90 items).
- Hardware: local arm64 Mac with MPS, model loaded in float16. Python 3.12 venv and exact installed package list in `environment.freeze.txt`. Seed 0, greedy generation, no GPU instance.

## Commands and results

```sh
cd /Users/shrranjan/audio-mechinterp/reproductions/anthropic_jlens
git -C jacobian-lens rev-parse HEAD
.venv/bin/python -m pytest -q jacobian-lens/tests
.venv/bin/python download_assets.py
.venv/bin/python inspect_readout.py
.venv/bin/python evaluate_association.py
.venv/bin/python run_swap.py
```

Upstream tests: **32 passed**. `download_assets.py` pins the two Hugging Face revisions above. `inspect_readout.py` ran the released 90 probe-swap prompts through the exact-model J-lens and stored all top-token readouts in `readout_results.json`. It is diagnostic only, because top tokens do not establish a faithful causal readout.

The full released **association** evaluation was scored at the final prompt token, using the minimum full-vocabulary rank across all 27 fitted layers. Candidate forms were all single-token original/lower/title-case surfaces with and without leading space, matching the scoring convention in the open [J++ implementation](https://github.com/safety-research/jpp_lens). Three of 102 concepts had no single-token form under this tokenizer and were logged as dropped. No layer was selected after observing results; no nonsemantic-token filter was used.

| Metric | Pre-fitted Jacobian Lens | Raw logit lens |
|---|---:|---:|
| pass@1 | 0/99 | 0/99 |
| pass@5 | 0/99 | 0/99 |
| pass@10 | 2/99 (2.02%) | 0/99 |
| pass@50 | 4/99 (4.04%) | 1/99 (1.01%) |
| pass@100 | 9/99 (9.09%) | 8/99 (8.08%) |

The pretrained lens did not produce a strong association readout on this small model. The published Qwen3.6-27B result is a different model and method, so these numbers are **not** a failed replication of that reported number. The released repository provides prompts and metric, but no published Qwen3-1.7B value for direct comparison. The complete per-item ranks and exclusions are in `association_eval.json`.

## Causal swap, preliminary subset

`run_swap.py` applies the [J++ causal runner](https://github.com/safety-research/jpp_lens/blob/main/src/lens_evals/causal_evals/runner.py) two-direction coordinate clamp: each concept direction is `J_l.T @ W_U[token]`; source and target vectors are normalized; clean residual coordinates on their two-vector pseudoinverse are exchanged at scales 0.5, 1, 2, and 4. We edited original prompt tokens except the attention sink at a fixed contiguous band, layers 12–16. Greedy continuations use the same edit on original tokens at every full forward. We selected four released prompts whose unmodified answer ranked first **before** running interventions. The saved continuations are capped at eight generated tokens; their first answer sentence terminates within that limit for the two candidate flips. This is a small gated qualitative subset, not the published 90-item causal score or a held-out validation.

| Released prompt | Clean next answer | Main concept swap | Eight-token continuation at that scale | Controls |
|---|---|---|---|---|
| Colosseum country / currency; Italy→Mexico | `euro` | `peso` top-1 at scale 1 and 2 | `peso. In the country where the...` at scale 1; `Mexican peso...` at scale 4 | self-swap stays `euro`; random `pencil` swap and layer-0 edit do not give `peso` at scale 1; direct answer swap gives `peso` at scale 4 |
| Darwin first name; Darwin→Einstein | `Charles` | `Albert` top-1 at scale 4 | `Albert. The second name...` | self and random controls do not give `Albert`, but the direct answer-swap positive control also fails at this scale |
| Elephant part; elephant→snake | `trunk` | no `tongue` top-1 at any scale | `snout...` or `tail...` | random null gives `tongue` at scale 1: clear false positive |
| Newton first name; Newton→Shakespeare | `Isaac` | no `William` top-1 at any scale | `Isaac...` or `Newton...` | controls similarly unstable |

This is **one candidate qualitative branch flip**, one weaker candidate, and two failures on the chosen subset. The Colosseum result supports a content-sensitive causal effect of this internal edit in this text model: the clean output, edited answer sentence, self, random, early-layer, and positive control are recorded. It does **not** prove the model naturally relies on a unique Italy direction, and it is not an audio-model or security finding. `swap_results.json` contains every scale, trial, rank, eight-token continuation, and skipped-layer diagnostic. Controls show that this intervention can also cause unrelated failures, so broader claims require a full, separately held-out causal evaluation and more precise controls.

## Implementation caveats and next gate

- The public Anthropic repo has released prompts and lens code but no directly runnable causal-swap evaluator. `run_swap.py` adapts the later open J++ clamp and is an independent implementation; it is not presented as an exact Anthropic benchmark reproduction.
- The small pre-fitted lens is weak on the association task. Before any audio extension, repeat a published readout on a model/lens pair with a reported reference value, then run the entire released swap set with a frozen band and full control counts.
- The code hooks post-block language-model residuals. An audio encoder state is a different space; neither this lens nor its causal claim transfers to audio by construction.
- No model fit, H100 instance, external service write, or real tool action was used.
