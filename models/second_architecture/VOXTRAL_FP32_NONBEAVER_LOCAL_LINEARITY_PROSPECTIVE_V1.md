# Prospective FP32 audio-state local-linearity pilot on a second FIT pair

18 November 2026. **Source-only design; no model run or executable freeze.**
This pilot tests whether a local audio-to-language Jacobian predicts its own
small, realized source-state edits on a different old-FIT lexical pair. It is
not a fitted reusable lens. The selected `color_05` FIT pair and its earlier
causal outcomes have already been inspected, so this is a numerical method
check, not held-out efficacy or an audio security result. Fresh corpus v2,
its development split, and its locked test remain unopened for this work.

## Why this version is needed

The [old-FIT numerical follow-up](VOXTRAL_V1_FIT_BACKWARD_NUMERICAL_FOLLOWUP_PROTOCOL_V1.md)
used a gradient-enabled clean graph **and gradient-enabled paired zero/edit
forwards** for its measured local changes. Its FP32 four-coordinate median
relative errors, about 0.072 at encoder 31 and 0.165 at the projector, are
therefore already graph-aligned; the BF16 errors near 1 cannot be attributed
to a graph/no-grad mismatch in that follow-up. The separate Stage-A handoff
used a gradient-enabled clean VJP and default no-grad paired traces, so its
stagewise predictions have that additional comparison limit. Preserve all
their original gates and outcomes.

The independently reviewed [four-condition factorial result](VOXTRAL_OLD_FIT_GRAD_LEAF_FACTORIAL_RESULT_INDEPENDENT_REVIEW_V1.md)
found a tiny, repeatable FP32 difference after the first captured LM block
when both grad mode and a differentiable source leaf were present. Source,
projector, LM input, and LM0 pre-input matched bitwise. Fixed-input direct-LM
replay reproduced the difference, and the first-token argmax did not change.
That one-case result motivates reporting graph-aligned and deployment-mode
edit deltas separately. It does not identify a kernel, an audio-specific
effect, or a reason for the earlier BF16 error.

## Fixed population and preflight

Use only `color_05`, words `silver` and `teal`, from the frozen
[FIT v2 selection](VOXTRAL_FIT_V2_SELECTION.json), SHA-256
`f897d360dbcec9c09f1813f9c070294753fbabc25c6213ba6bdac34b1d79b56b`.
The template is `fit_chosen` only. The four recipient contexts are Piper
`silver`, Piper `teal`, Kokoro `silver`, and Kokoro `teal`, in that order; the
opposite word in the same voice/template is the donor. The four exact WAV
SHA-256 values are:

| Engine | `silver` | `teal` |
|---|---|---|
| Piper | `fe554972f68b470fb0f4a1c844b874e70758279c281bba47e963792bef2634a9` | `112ed36abf5fb2be1ce77dcfffa9d1bb6deba10b4ecb2d14fc5960b8106626e9` |
| Kokoro | `bcad702fa8b55c453286cb0e83ee3336a4bb6ade5e5ac9d4a42dc82c1a97c26e` | `5038dbe167ae2ba6c9317f74c17bd9bb8ef7d07bbdee9da49f6dc0731e119365` |

Bind the [frozen FIT masks](VOXTRAL_FIT_V2_CAUSAL_MASKS.json), SHA-256
`a634fc0ab8454b7a82802c56a5365dc90e79f446e4d280cee6321adfead4108e`.
Select its `color_05|fit_chosen` Piper and Kokoro records and exactly their
`whole_word_rows`: 60 encoder/15 projector rows for Piper and 73 encoder/18
projector rows for Kokoro. Rehash the four WAVs, require mono 16-kHz signed
16-bit PCM, verify matched-pair equality outside the recorded word slot and
difference inside it, and recompute the mask rows from the frozen builder.
Reject any path outside the selected four clips. Hash the rendered WAV bytes,
processor batch tensors, prompt IDs, audio-placeholder positions, source
selection, mask builder, installed Transformers code, checkpoint files, and
protocol before loading weights. A reviewed CPU preflight receipt must bind
the exact four batch shapes and hashes. If they differ from the nominal
`[1,396]` text/attention, `[1,128,3000]` audio feature, 375 audio-placeholder,
`[1,1500,1280]` encoder-31, or `[375,3072]` projector geometry, stop and
version a new protocol **before** live inference. The nominal row clock is
not an exclusive acoustic receptive field.

Use the same local
`mistralai/Voxtral-Mini-3B-2507@3060fe34b35ba5d44202ce9ff3c097642914f8f3`
checkpoint, pinned one-word audio prompt, Transformers source and SDPA
implementation as the reviewed FIT assay. Load the BF16 checkpoint as a
whole FP32 model, assert every named parameter is FP32 on the intended
device, freeze weights, use `eval()`, `use_cache=False`, no autocast, and
record PyTorch/MPS versions. Upcasting the checkpoint does not recover
unquantized pretrained weights. One early source/hook shape check and one
natural greedy `max_new_tokens=32` answer per recipient and donor are fixed
diagnostics; wrong or malformed answers stay in the planned population.

## Local derivative and two execution modes

The two source boundaries are the forward output of audio encoder block 31
before its final norm and the forward output of the audio-to-LM projector.
The target is the prompt-final LM block-29 post-block/pre-final-norm residual,
width 3,072. Capture full LM29 target states and first-answer logits, and
passively capture prompt-final LM0 and LM14 states. For each recipient/site,
let `R` be the fixed whole-word rows and let
`δ = mean_R(H_donor) − mean_R(H_recipient)`, computed from the natural source
states of the matched donor and recipient. A donor never supplies a target,
gradient, fitted parameter, or test-label-dependent selection. Record and
hash both full natural source states before editing. If either direction has
zero/nonfinite `δ`, retain the failed cell and stop that context without
substituting a different pair.

For each recipient, construct one gradient-enabled clean graph with a
differentiable encoder-31 source leaf and its ordinary connected projector
path. Do **not** detach or replace the projector on that clean graph. Capture
both source gradients from the same LM29 target graph. The primary target
probe is a frozen four-vector orthonormal Rademacher basis in width 3,072,
generated by CPU float64 QR from a seed derived only from the selection SHA
above and literal label `color05_fp32_four_probe_v1`, with positive QR
diagonal signs. Save its exact tensor bytes and SHA in the pre-run freeze.
This basis probes numerical transport; it is neither trained on this pair nor
a semantic readout. For each basis coordinate, take one VJP to both source
sites, for exactly four VJPs per recipient and sixteen across the four
contexts. A disconnected, nonfinite or wrong-dtype gradient fails that
context without a substitute.

For site `s`, add the same vector `αδ_s` to each row of `R_s` at the source
hook, separately from the other site. The **G1** path uses
`torch.enable_grad()` and a FP32 source leaf with `requires_grad=True` on
both paired zero and edited forwards. The **N0** deployment path uses
`torch.no_grad()` and a leaf with `requires_grad=False` on both paired
forwards. Assert and save the actual grad-mode and leaf flags inside the
source hook and LM0 pre-hook, plus whether the LM input has a gradient path.
G1 is the derivative's matching forward definition; N0 is a separate
deployment comparison. Never subtract a G1 zero from an N0 edit.

## Frozen edit schedule and resource estimate

Use the signed nonzero grid `A = (+.00625, −.00625, +.025, −.025,
+.1, −.1)` in cycle 1, reversed in cycle 2, at each site. For every alpha
in each cycle, run an exact zero edit immediately before its edited forward.
Do all G1 pairs first for a site, then all N0 pairs; reverse the site order
for the second voice. Within a context this is 2 sites × 2 modes × 2 cycles
× 6 alpha values × 2 paired forwards = **96 patched forwards**, plus one
donor capture, one clean graph with four VJPs, and two natural generations.
Across four contexts the fixed workload is 384 patched forwards, 16 VJPs,
four donor captures, four clean graphs, and eight natural generations. This
is an intentionally broader, independent-input pilot than the old one-case
72-forward follow-up. Do not shrink it after viewing an early result.

Run one fresh process per context in the fixed order above, with a separate
exclusive reservation and start receipt for each. Freeze all four process
entries before the first starts; a failed entry remains in the all-planned
denominator and does not authorize a retry. Cap each process at 300 seconds,
MPS driver allocation below 30 GiB, and combined local output below 1 GiB;
cap stdout/stderr separately at 8 MiB and the independent CPU audit at 120
seconds. The earlier FP32 pilot completed 72 patched forwards, four VJPs,
two generations, and its ancillary captures in about 81 seconds while
peaking near 28.8 billion MPS driver bytes. That is a cost reference, not a
guarantee for this four-context, 96-forward-per-process workload. A source
review must make a forward-count and serialized-byte estimate using actual
tensor shapes before freezing the launcher. If even one fixed process cannot
fit the 30 GiB cap, stop and prospectively port the same population and
schedule to an available CUDA GPU under a separately reviewed environment
receipt. No fallback dtype or omitted mode/site is allowed.

Save full clean/donor source states, and at least one full zero-state
source/projector/LM-input capture per site/mode/context. On every paired
forward save its prepatch and postpatch selected rows, effective FP32 row
edits, prompt-final LM0/14/29 vectors, full LM29 target, first logits,
hook counts, mode flags, and hashes of full source/projector/LM inputs. A
hash without its full tensor is a receipt claim, not independently
recomputable tensor evidence; the auditor must make that limit explicit.
Use a local content-addressed tensor sidecar with a SHA/byte manifest and
commit the small raw receipts. Preserve the original failed and completed
one-shot attempts unchanged.

## Metrics, decisions, and limits

For target probe coordinate `k`, row gradient `g_{k,j}`, realized within-edit
source change `e_j = H_postpatch,j − H_prepatch,j` from the edited forward,
and same-mode paired target `z`, report:

`P_row(k) = Σ_{j∈R} g_{k,j}·e_j`,
`P_uniform(k) = (Σ_{j∈R} g_{k,j})·mean_R(e_j)`, and
`D_G(k) = u_kᵀ(z_G1,edit − z_G1,zero)`.

`P_uniform` is the primary uniform-bin Jacobian estimate for the future
lens; `P_row` is the local rowwise oracle control. Report ideal `αδ`
contraction separately. Also report `D_N` from the matched N0 pair and the
mode interaction `D_G−D_N`, without treating a G1 clean-to-N0 edit
difference as an edit effect. Save the corresponding full 3,072-wide LM29
changes, LM0/14 movements and full-logit movement, and report any top-token
change. Do not infer complete-answer changes from first logits.

Require the selected prepatch source rows in every paired zero and edited
forward to match that mode's clean source rows bitwise. If they differ,
retain the traces but mark the local-derivative cell invalid; do not treat
baseline drift as part of the patch edit or silently adjust the derivative.
Compare the G1 and N0 clean prepatch source rows as a separately reported
condition check.

For each context/site/mode, take the maximum L2 spread among its twelve
paired zero target projections as observed repeat drift `N`; define that
mode's resolution floor as `max(3N, 1e−6)`. With only two cycles, this is a
control threshold, not a noise distribution. Keep all 48 planned site-alpha
cells per mode (four contexts × two sites × six alpha values), including
malformed, unresolved and failed cases, in the denominator; retain all
paired zero baselines separately.
For cells above the G1 floor, report `||P−D_G||₂ / max(||D_G||₂,N,1e−6)`
for both predictions, cosine only when both norms exceed the floor, and
positive/negative odd/even components at each absolute alpha. Record
intended/realized edit norm, rowwise nonuniformity and mean fidelity; require
each finite nonzero edit to have mean-fidelity and nonuniformity ratios at
most 0.10 or mark the cell invalid without replacing it. Compare G1 and N0
full-target edit deltas only after each mode's own paired-zero control.

The **advance-to-fresh-corpus feasibility gate** is deliberately numerical:
at both sites, at least 18 of 24 planned site-alpha cells must exceed the
G1 repeat floor, the median primary uniform-prediction relative error over
the resolved cells must be at most 0.25, and no more than four resolved
cells per site may exceed 0.50. All four contexts must produce valid
preflight, finite gradients, complete captures and a passing independent
saved-data audit; otherwise the gate fails. These thresholds are design
choices, not achieved results. A source-only readout and development/locked
test remain separate later gates. If the numeric gate fails, publish the
full FIT pilot and revise a new protocol before any reusable-map inference.

An independent CPU auditor must recompute the fixed basis, all tensor
hashes available from saved bytes, pairwise arithmetic, repeat floors,
source/edit fidelity, counts and thresholds from the sidecars. A separate
reviewer must verify the source and result. This experiment can show only
whether this FP32 local derivative predicts fixed constructed state edits
on one already-inspected FIT lexical pair, and whether G1 and N0 edit
effects agree there. It cannot establish a reusable audio lens, precise
acoustic time localization, generated speech or actions, an attacker
controlled waveform, or a security vulnerability. It cannot open fresh v2
inference until that corpus's independent 480/480 intelligibility gate and
a new executable lens protocol both pass.
