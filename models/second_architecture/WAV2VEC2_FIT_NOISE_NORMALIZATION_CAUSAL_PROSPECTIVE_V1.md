# Wav2Vec2 FIT acoustic-noise normalization: prospective causal follow-up

19 November 2026. **Source-only design; no new model inference.** This
implements the question selected in the independent [next-study decision](ACOUSTIC_NORMALIZATION_CAUSAL_NEXT_STUDY_DECISION_V1.md). It uses already
inspected synthetic FIT audio. It is not a held-out evaluation, an
attacker-suppliable exploit, or a security finding. A reviewed worker,
immutable no-weight preflight, separate bounded launcher, and independent
saved-result audit must precede any checkpoint load.

## Frozen inputs and known outcomes

Use all **eight** clean/noisy pairs: Piper and Kokoro × spoken `silver` and
`teal` × the existing `word_noise` and `outside_noise` arms. The source
recordings, exact 16-kHz PCM, fixed 20-dB noise field, windows, and all
transformed PCM hashes are pinned by `acoustic_fit_noise_pilot_v2.py`
(SHA-256 `72a03647cea026b401a9f9db40ffe3dc29ab90b4ceb1e7a23bee31acc85860cb`).
The independently audited [20-arm stock result](ACOUSTIC_FIT_NOISE_WAV2VEC2_V2_RESULT_INDEPENDENT_REVIEW_V1.md) has worker
JSONL SHA-256 `c1badaf305dfaebe9beb9567b73f4e4bfdfab84b45ca8194f4c13c943f60038c`.
Require the exact stock clean/noisy complete CTC frame-ID paths and
transcripts from that JSONL to replay before interpreting counterfactuals.
The checkpoint is `facebook/wav2vec2-base-960h` revision
`22aad52d435eb6dbaf354bdad9b0da84ce7d6156`, weight SHA-256
`8aa76ab2243c81747a1f832954586bc566090c83a0ac167df6f31f0fa917d74a`,
under Torch 2.15.0 and Transformers 5.20.2 with Wav2Vec2 modeling-source
SHA-256 `0c5027789d81a622f72f062eab1ee16566bb8dfb200cb6da0a9ffef06543b287`.

Only two of these eight prior contrasts changed a complete transcript:
Kokoro/teal `word_noise` changed the model's clean error `TIL` to the
intended `TEAL`; Kokoro/silver `outside_noise` changed preceding `IS` to
`ISE` while `SILVER` stayed correct. The six unchanged contrasts remain in
every denominator and result table. These are observed FIT outcomes, so a
mechanism on either case is post hoc; no rate or generalization estimate
follows. The outside-noise arm directly edits outside-word samples and
cannot demonstrate remote spread. The word-noise arm keeps outside-word
raw PCM byte-identical and is the localization contrast.

## Stage I: full-utterance input-statistic 2×2 on all eight pairs

Let `X∈{clean,noisy}` select exact PCM. From `x=PCM16.astype(float32)/32768`
calculate full-utterance `mu_X=x.mean()` and population
`var_X=x.var()` with pinned NumPy 2.3.5 operations. Let
`S∈{clean,noisy}` select **which waveform supplied the statistics**, and
feed `(x_X-mu_S)/sqrt(var_S+1e-7)` as Wav2Vec2 `input_values`.
Do not truncate silence, alter waveform length, or use the older
pair-outside statistics here. Save both statistics, their dtype/bit hashes,
input tensor hash, and exact current-waveform/selected-stat provenance.

Freeze arm order per pair as `(X,S) = (clean,clean), (clean,noisy),
(noisy,clean), (noisy,noisy)`. For the two diagonal arms, require the
manual float32 values to equal the pinned stock processor output **byte
for byte**, then require the complete CTC frame-ID
path, transcript, and logits to match a no-hook stock forward exactly.
Run a read-only input hook and require bitwise logits and CTC path parity
with no hook. Any failed diagonal or read-only parity makes that pair's
Stage I counterfactuals uninterpretable; keep all attempted rows and do
not choose another implementation after seeing endpoints. The two
off-diagonal arms swap only normalization statistics while holding source
PCM fixed. Report complete logits, top-2 margins, CTC paths, transcripts,
and time-resolved `input_values`, first raw convolution, first GroupNorm,
feature-projection, and post-block-0 differences for all arms.

Primary Stage I contrasts for each pair are:

* **Statistic sufficiency, clean waveform:** compare `(clean,noisy)` with
  `(clean,clean)` at identical PCM. A complete decoded change supports
  sufficiency of the noisy full-utterance statistics in this fixed model
  context.
* **Conditional statistic dependence, noisy waveform:** compare
  `(noisy,noisy)` with `(noisy,clean)` at identical noisy PCM. Restoration
  of the clean decision when clean statistics replace noisy statistics
  supports conditional dependence; persistence of the noisy decision
  shows direct noisy samples or their interactions can suffice here.

Neither contrast alone proves unique mediation. A statistic swap is a
white-box preprocessing intervention, not a waveform processed by the
stock feature extractor. Use the complete CTC sequence, not only the
spoken target word, as the decision endpoint.

## Stage II: first GroupNorm-statistic 2×2 on four word-noise pairs

Use **only the four word-noise pairs**, regardless of prior transcript
change. For each pair, calculate one common input mean/variance from the
clean samples **outside** the frozen word slot, using the exact shared
input normalization formula and receipt in
`wav2vec2_fit_baseline_stage_capture_v1.py` and its [independent review](WAV2VEC2_FIT_BASELINE_V2_RESULT_INDEPENDENT_REVIEW_V1.md). Process both
clean and word-noisy PCM with those identical statistics. Require exact
outside-word input equality and exact equality of first raw-convolution
outputs at frames whose kernel-10/stride-5 receptive interval does not
overlap the word slot. Piper slot `[30527,49847)`, Kokoro slot
`[33920,57120)`; frame `j` overlaps `[s,e)` iff `5j<e` and `5j+9≥s`.

Capture the **clean and noisy first raw convolution tensors under this
same shared-input path**, each `[1,512,T]`, before constructing any
counterfactual. For each channel take mean and population variance across
its entire time axis; call those exact saved statistic sources `G_clean`
and `G_noisy`. Then cross raw-convolution source
`R∈{clean,noisy}` with statistic source `G∈{clean,noisy}`. Arm order is
`(R,G)=(clean,clean),(clean,noisy),(noisy,clean),(noisy,noisy)`.
At the first GroupNorm output only, calculate
`(raw_R-mu_G)*rsqrt(var_G+1e-5)*gamma+beta` with pinned FP32 affine
weights. Feed that tensor to the unchanged downstream model; all other
layers and weights stay stock. Save exact raw/stat/affine/output hashes,
shape, dtype, edit norm, hook count, full logits and greedy CTC path.
Do **not** use stock-normalized or Stage I tensors as `G` donors. This
provenance rule is essential: both `G` statistics come from the same
pair-specific shared-input path as their `R` tensors.

Require no-hook versus read-only GroupNorm hook **bitwise** logits/path
parity. On each natural-statistic diagonal, compare the manual GroupNorm
output with the stock output at maximum absolute error `≤2e-5`, logits
at maximum absolute error `≤1e-2`, and require **exact complete CTC
frame-ID path and transcript parity**. The prior saved formula audit found
up to `1.15e-5` floating-point discrepancy, so a bitwise logit rule for
the algebraic replacement would be unsound. Record the smallest top-1
versus top-2 logit margin on any frame whose argmax differs between
compared arms. If that margin is `≤2×` the relevant max-abs logit
reproduction error, mark the decoded change numerically **inconclusive**.
Any failed natural diagonal invalidates Stage II interpretation for that
pair; do not loosen thresholds or choose a different site afterward.

Primary Stage II contrasts mirror Stage I at a different boundary:
`(clean,noisy)` versus `(clean,clean)` tests noisy GroupNorm-statistic
sufficiency with the raw convolution fixed; `(noisy,noisy)` versus
`(noisy,clean)` tests conditional dependence with the noisy raw
convolution fixed. Require the crossed clean-statistic arm to restore
exact outside-word first-GroupNorm equality where raw inputs were
identical. If later representations or the CTC sequence still differ,
report the measured residual downstream effect, without calling it an
attention circuit: later convolutions, attention and interactions remain
possible. The shared-input path is a controlled counterfactual, not the
stock deployed preprocessing path.

## Boundaries and run admission

The frozen scored matrix has **32 Stage I + 16 Stage II** complete CTC
forwards. Stage I also has eight no-hook and eight read-only-hook stock
forwards. Stage II has eight no-hook and eight read-only raw-convolution
capture forwards, one clean and one noisy per pair. Its capture hook must
return the input unchanged and match no-hook logits bitwise. **The total
ceiling is 80 model forwards**: 48 scored and 32 controls/captures. No
forward can be retried under the same attempt ID, and an early failure
must leave all remaining cells censored in the ledger. Save all eight
pairs and all failures/censors, including
missing captures, unexpected shapes, changed source/weight bytes,
partial outputs, and invalid numerical-parity cells. There is no
feature-projection donor-patch matrix in this follow-up: the prior
selected [one-case state transplant](ACOUSTIC_FIT_NOISE_CAUSAL_RESULT_INDEPENDENT_REVIEW_V1.md) already
addresses that different question, and repeating it here would not
isolate either normalization statistic.

The future worker must use one pinned CPU checkpoint load, one-sample
batches, `eval()` with no dropout, four Torch threads, a 1,200-second hard
parent deadline, 4-GiB RSS ceiling, 2-GiB new-evidence ceiling, AC
power/open lid, fresh pre/post sleep/thermal logs, and no retry under
the same attempt ID. A separately reviewed saved-data auditor must
rebuild both normalization formulas from pinned PCM and recorded
statistics, verify every tensor/CTC path and output inventory, and
enforce the resource/call ledger. The original 20-arm pilot and selected
state transplant remain immutable.

A meaningful result is a complete CTC change under a frozen statistic
swap with valid stock/read-only/natural-statistic controls and an
independent saved-result audit. A null remains informative: global
activation spread alone may leave the complete decision unchanged.
This FIT-only English ASR assay cannot show fresh-speaker/word transfer,
human intelligibility, an assistant action, attacker reachability, or a
security vulnerability. The sealed audio-language studies address
those separate questions.
