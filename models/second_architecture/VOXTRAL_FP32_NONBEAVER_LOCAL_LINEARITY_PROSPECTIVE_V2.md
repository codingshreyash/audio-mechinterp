# Prospective FP32 audio-state local-linearity pilot on a second FIT pair — V2

18 November 2026. **Amended source-only design; no model run or executable
freeze.** The [V1 protocol](VOXTRAL_FP32_NONBEAVER_LOCAL_LINEARITY_PROSPECTIVE_V1.md)
and its [independent HOLD review](VOXTRAL_FP32_NONBEAVER_LOCAL_LINEARITY_CONCEPTUAL_REVIEW_V1.md)
remain immutable. This V2 defines matched clean/zero topology, two-cycle
scoring, N0 references, and an exact target-probe recipe before any model
work. It requires a separate conceptual GO, implementation review, and
external freeze.

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
`δ = mean_R(H_donor,N0) − mean_R(H_recipient,N0)`. The donor is a separate
no-grad natural capture; the recipient is the **prepatch source** from that
site's explicit N0 clean-reference forward, which runs the ordinary upstream
audio path before its α=0 replacement. Thus both terms use the same FP32
mode and matched voice/template. A donor never supplies a target,
gradient, fitted parameter, or test-label-dependent selection. Record and
hash both full natural source states before editing. If either direction has
zero/nonfinite `δ`, retain the failed cell and stop that context without
substituting a different pair.

For each recipient **and each site separately**, construct a G1 clean-VJP
graph with the *same hook topology* as that site's later G1 zero/edit calls.
At encoder 31, return a detached FP32 leaf requiring grad from the block-31
hook and allow the real projector to remain connected downstream. At the
projector site, let the unmodified encoder and projector run, then replace
the projector hook output with a detached FP32 leaf requiring grad. Do not
reuse an encoder-31-leaf graph as the projector VJP graph. An α=0 G1 patch
must use the same detach/replace operation, leaf flag, grad context, hook
registration order, batch, and logits settings as its site's clean-VJP
graph. This consumes **two** clean-VJP graphs per recipient, not one shared
graph for both sites.

Generate the fixed four-probe basis `U` with this exact CPU recipe before
any model load. Let `S` be the 32 bytes decoded from the selection SHA-256
above, and `L` be the ASCII bytes
`color05_fp32_four_probe_v2`. Let `d = SHA256(S || L)`, then
`seed = int.from_bytes(d[:8], "big") mod 2**63`. In the reviewed, pinned
PyTorch CPU environment, create `torch.Generator(device="cpu")` and call
`manual_seed(seed)`. Draw
`signs = torch.randint(0, 2, (3072, 4), generator=g,
dtype=torch.int8)`, convert to float64, and set `M = 2*signs - 1` without
other random draws. Compute `Q,R = torch.linalg.qr(M, mode="reduced")` in
float64, reject any zero/nonfinite diagonal of `R`, then multiply each
column `Q[:,j]` by `sign(R[j,j])`, and make `Q` contiguous. Require
`Q.T@Q` to equal identity within `atol=1e-12, rtol=0`. Define
`Q_np = Q.cpu().numpy().astype("<f8", copy=False)` and the basis digest as
`SHA256(b"float64[3072,4]\0" || Q_np.tobytes(order="C"))`. Save `Q`, this exact
digest, PyTorch version and CPU environment in the pre-run freeze. A
CPU-only calculation with local PyTorch `2.15.0` and NumPy `2.3.5` gives
seed `3693039655029897049` and basis digest
`10c9132454a01e034886a3e14bf9b9670e75bb3befd58300b9b45f8d9ba7751f`;
the independent source review must reproduce these exact values before
freezing them. A runtime recomputation must match the frozen tensor
bitwise or stop. This
basis probes numerical transport; it is not trained on this pair and is not
a semantic readout. On each of the two site-specific G1 clean graphs, take
four VJPs of `U[:,k]·z_LM29` to that site's source leaf. This fixes eight
VJPs per recipient and **32 total**. A disconnected, nonfinite or
wrong-dtype gradient fails that context without a substitute.

For each site and recipient, capture an explicit **N0 clean reference** with
the identical site hook and α=0 replacement used later, but under
`torch.no_grad()` with a nondifferentiable FP32 leaf. Save the N0 clean
prepatch and returned full source, full projector output, full LM input,
prompt-final LM0/14/29 states and first logits. Save the analogous G1
clean-VJP tensors *before* backward. Compare G1 and N0 prepatch source
rows and full states descriptively, but never require cross-mode equality
for the within-mode derivative estimand. The N0 clean reference defines
`that mode's clean source rows` for all subsequent N0 paired calls.

The G1 clean-VJP graph must be tied to its actual patched zero baseline.
For every G1 zero call at that site, compare its returned full site-source
tensor, full projector output, full LM input, full prompt-final LM29 state,
and full first-answer logits against the saved site-specific G1 clean-VJP
graph. **All five must be bitwise equal**; no tolerance or observed-floor
substitution is allowed. Also require bitwise equality of the prepatch
full source and selected rows. Apply the same bitwise clean-versus-zero
rule to every N0 zero call against its site-specific N0 clean reference.
If any required tensor differs, save both sides and invalidate that
context/site numerical gate. Do not use its VJP against a different
baseline, choose a later zero, or soften the parity rule after seeing data.

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
donor capture, two site-specific G1 clean graphs with four VJPs each,
two site-specific N0 clean-reference forwards, and two natural generations.
Across four contexts the fixed workload is 384 patched forwards, 32 VJPs,
four donor captures, eight G1 clean graphs, eight N0 clean references,
and eight natural generations. This
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
guarantee for this four-context, 96-patch-plus-five-reference-forward
process. A source
review must make a forward-count and serialized-byte estimate using actual
tensor shapes before freezing the launcher. If even one fixed process cannot
fit the 30 GiB cap, stop and prospectively port the same population and
schedule to an available CUDA GPU under a separately reviewed environment
receipt. No fallback dtype or omitted mode/site is allowed.

Save full G1 clean-VJP, N0 clean-reference, and donor
source/projector/LM-input captures. Save full tensors for **every** paired
zero call so the CPU auditor can independently verify the bitwise parity
rules above. On every paired
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

For each context/site/mode, the twelve zero calls are ordered by cycle
`r∈{0,1}` then the six alpha positions. Let `q_i=Uᵀz_zero,i` in float64.
Define observed zero drift **exactly** as
`N_mode = max_{0≤i<j<12} ||q_i−q_j||₂` over all 66 unordered zero pairs.
The mode's fixed resolution floor is `F_mode=max(3N_mode,1e−6)`. A missing,
nonfinite or malformed zero makes that context/site/mode invalid; do not
compute a floor from the remaining zeros. Two cycles provide an observed
control threshold, not a sampling-based noise distribution. The bitwise
clean-versus-zero rule above remains a separate mandatory check and is
never weakened by `F_mode`.

For each `(context,site,alpha,mode,cycle)` pair, require exactly one zero
and one edit in the fixed order, valid hook/shape/dtype/runtime flags,
finite full target/logits/selected rows, and exact zero effective edit.
For edited forwards, require the selected prepatch rows to match the zero
and clean references bitwise and assert at runtime that the *full* prepatch
source hash matches its clean reference; the saved selected rows permit
independent row comparison, while the edited full hash remains a runtime
claim because that full tensor is not archived. For its nonzero edit,
require finite nonzero realized
norm and mean-fidelity and row-nonuniformity ratios each ≤0.10. A failed
check invalidates **that cycle** and retains its raw tensors. A planned
`(context,site,alpha)` cell is valid only if both G1 cycles and both N0
cycles are valid and all clean-versus-zero parity checks at that site pass.
If either cycle fails, the cell is invalid and remains in its planned
denominator; there is no single-cycle fallback or replacement.

For each valid G1 cycle `r`, compute its own projected edit delta
`D_G,r`, uniform prediction `P_uniform,r`, rowwise prediction `P_row,r`,
and relative errors
`E_type,r=||P_type,r−D_G,r||₂ /
max(||D_G,r||₂,N_G1,1e−6)` in float64. The cell is **resolved** only when
both `||D_G,0||₂` and `||D_G,1||₂` exceed `F_G1`. Its primary cell error is
`max(E_uniform,0,E_uniform,1)`; the rowwise cell error uses the analogous
maximum. Report both individual cycle values and the maximum; never
average away a bad cycle. Report cosine for a cycle only when its predicted
and measured norms each exceed `F_G1`; otherwise record null. For N0,
compute `D_N,r` against its own immediately paired zero. Report
`D_G,r−D_N,r` and its norm for each matched cycle/alpha, even when N0 is
below its own floor, while labeling that N0 change unresolved. For each
absolute alpha and each cycle separately, report odd
`(D(+h,r)−D(−h,r))/2` and even `(D(+h,r)+D(−h,r))/2` components, likewise
for the predictions. Do not choose a favorable cycle or alpha.

There are exactly 48 planned `(context,site,alpha)` cells (24 per site),
each with four paired mode/cycle measurements. Keep invalid and unresolved
cells in those counts. A site passes the numerical gate only if **all 24
cells are valid**, at least 18/24 are resolved, the median of the resolved
cells' maximum-cycle primary errors is ≤0.25, and no more than four of the
24 cells have maximum-cycle primary error >0.50. If fewer than 18 resolve,
the median criterion is not evaluated and the gate fails. Compare G1 and
N0 full LM29 edit deltas only from each mode's own paired zero; report
cross-mode clean differences descriptively, never as an edit effect.

The **advance-to-fresh-corpus feasibility gate** requires both sites to
pass the preceding exact 24-cell rule. All four contexts must also pass
preflight, finite site-specific VJPs, complete captures and an independent
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
