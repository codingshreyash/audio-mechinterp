# EOS loss audit for *Never Stop Speaking*

**Source-only methodological note, 19 November 2026.** This checks the printed equations in [Cheng et al., *Never Stop Speaking*, arXiv:2608.10405v1](https://arxiv.org/html/2608.10405v1), especially Eqs. 5–7. It is not a replication and does not question the reported attack outcomes by itself.

## Categorical EOS probability is shift invariant

The paper sets `p_t = sigmoid(z_EOS,t)` and calls it the EOS probability. For a decoder that selects among vocabulary tokens, the categorical EOS probability is `softmax(z)_EOS = exp(z_EOS) / sum_j exp(z_j)`. Adding the same constant to every logit leaves the categorical distribution, greedy choice, and sampled-token distribution unchanged. It changes both the paper's `sigmoid(z_EOS)` and its Eq. 5 gate `z_EOS > 0`.

Minimal example with just `[EOS, other]` logits:

| Logits | Actual EOS probability | Printed sigmoid proxy | Eq. 5 gate | Greedy token |
| --- | ---: | ---: | --- | --- |
| `[-1, 0]` | 0.26894 | 0.26894 | Off | other |
| `[1, 2]` (add 2 to both) | 0.26894 | 0.73106 | On | other |

Thus the printed proxy and gate can change substantially when actual decoding is identical. This is a mathematical check on the equations, not evidence that an audio waveform can induce a pure common-logit shift.

Eq. 6 also omits the probability mass of **reaching the generation cap without EOS**. If `p_t = 0` at every step, the printed expected length is zero, while a generation capped at `N` emits `N` steps. Under an EOS-hazard interpretation, a capped expected length needs the surviving-at-cap term `N * product_{i=1}^N (1-p_i)` added to the printed sum. A definition that treats cap termination differently would need to be stated.

## Reproducibility boundary

The [arXiv source package](https://arxiv.org/src/2608.10405) contains LaTeX, references, styles, and figures, but no implementation or data files. The paper's LaTeX contains only commented template code/data links. Targeted GitHub repository searches by title, arXiv ID, and first author found no matching repository on this date. **The implementation remains unverified**: it may differ from the printed equations, and a release could appear later.

The paper reports actual generated length and cap-hit rates on three audio models, including a Qwen2-Audio appendix. The equation issues do not invalidate those observed outcomes. They mean we should independently validate any surrogate loss against **decoder-relative EOS probability, actual EOS emission, and full capped output length** before using it to explain a model's failure. A Qwen2.5-Omni replication alone would extend the model list; a distinct mechanistic result would require a causal audio-state intervention that changes complete termination behavior on held-out speech under matched decoder and cap settings.
