# Old-FIT FP32 grad-context × leaf-requirement factorial diagnostic

18 November 2026. This is a distinct prospective numerical study. The
graph/no-grad parity diagnostic V1 reservation `20261118T151548690094Z`
remains consumed and failed before any full forward, because its runner
required a plain `dict` whereas the reviewed CPU preflight supplies a
`transformers.BatchFeature` mapping. V1 is not retried, reopened, or used as
evidence for a numerical conclusion. Its raw receipts, failure manifest, and
independent review must be verified before this study can start.

## Question and fixed input

On the same one old-FIT `beaver` audio request and identical archived BF16
clean source values cast to FP32, can grad context and source-leaf
`requires_grad` be separated as factors in the resulting forward states?
The four conditions are `N0` (`torch.no_grad`, leaf flag false), `N1`
(`torch.no_grad`, leaf flag true), `G0` (`torch.enable_grad`, leaf flag false),
and `G1` (`torch.enable_grad`, leaf flag true). Model parameters are frozen;
the model remains in eval mode. `G1` recreates V1's graph-building condition,
and `N0` recreates its no-grad condition, but this study asks a different
factorial question using two additional conditions. `N1` does not build a
backward graph under `no_grad`; `G0` can enable grad mode without a leaf that
requires grad. This contrast cannot, by itself, identify a specific kernel or
prove a model-level audio-security mechanism.

The exact source sites are post-encoder layer 31 (`encoder31`, shape
`[1,1500,1280]`) and post-audio projector (`projector`, shape `[375,3072]`).
The archived full BF16 clean source from reviewed Stage-A BF16 cycle zero is
fixed before model load: encoder31 SHA-256
`e0abb561ad91255d1945c81c1dead1a75266f09d0e7f3695a123040fef47b2b6`;
projector SHA-256
`a96b5e19ffefe48e537b30ddecb9de11c8c40becf4b4dd3f2813ab5583fce100`.
The old-FIT batch must be normalized from a `Mapping` of tensor values to a
plain CPU tensor dict **before model construction**. Require exactly
`input_ids` int64 `[1,396]`, `attention_mask` int64 `[1,396]`, and
`input_features` float32 `[1,128,3000]`, with tensor-byte SHA-256 values
`915a4530875920b9c988724ca9e7ed8e63550bdfec3302ddb66e84e67b4b314a`,
`9ecc0f254cc29a01ab79e006021d96ef320b7438202692dcee10c770fbd65282`,
and `4f23ae2441eca7587b7acf19d89f717292a4f1efb3067631c7abb7e27236ee61`,
respectively. The mask is all ones, audio token ID is 24, and the input has
exactly 375 audio placeholders. The exact pinned local Voxtral checkpoint,
SDPA, model parameter-layout rule, and offline environment remain as in the
independently reviewed old-FIT handoff. The fresh v2 corpus is not opened.

## Fixed schedule and captures

For `encoder31`, run full-model forwards in this order:
`N0,G1,G0,N1,N1,G0,G1,N0`. For `projector`, use the site-shifted order:
`G1,N0,N1,G0,G0,N1,N0,G1`. Each factor combination appears exactly twice
at each site, with conditions spread across early and late calls. The full
count is 16. Capture each returned full posthook source, full projector
output, full LM input, full LM0 pre-block input, terminal post-block vectors
at LM0, LM7, LM14, LM22, LM29, full first-answer logits, source hash,
site/condition/repeat/ordinal, and audio-placeholder positions. Record the
actual grad context and source-leaf `requires_grad` in the source hook.
Record grad context and `.requires_grad` on the full LM input and LM0
pre-block tensors separately. Require
correct shapes, finite tensors, exact source override, correct hook counts,
and exact projector-to-LM audio-position equality. Do not assert equality
between separate forwards before saving all captures.

For each site, take the **first** full-model `N0` LM input. Replay that
single bitwise-fixed LM input directly under the same four conditions,
using that site's exact eight-call full-model order. The direct count is 16. The leaf flag
is applied to the fixed direct input; attention mask, optional position IDs,
LM0 pre-block input, five terminal vectors and first logits are captured.
Record the actual direct input leaf flag inside its grad context, and the
separate LM0 pre-block grad flags. A detached or transformed LM0 input must
be visible in the record rather than mislabeled as a leaf-factor failure.
No audio front end runs in the direct calls. No VJP, generation, edited
source, or attack is in scope. Total planned model forwards: 32.

## Independent saved-data analysis

An independently reviewed CPU auditor must rehash retained tensors and
verify the schedule, batch, source, capture shapes, counts and provenance.
Verify that the requested factor flags match the flags observed on the
source/direct input, and report downstream grad connectivity separately.
For each site and ordered boundary, report bitwise equality, changed
coordinate count, float64 L2/max-absolute/symmetric-relative L2 for all
two-by-two cross-condition repeat pairings. Report each within-condition
repeat spread. A boundary's observed resolution floor is the larger of
`3 × max(within-condition L2 spread)` and `1e-6`; two repeats do not estimate
a statistical noise distribution. Keep the first bitwise differing saved
boundary separate from the first difference above the floor.

Report the 2×2 contrasts at fixed leaf flag (`N0` vs `G0`, `N1` vs `G1`),
fixed grad context (`N0` vs `N1`, `G0` vs `G1`), and the interaction, with
within-condition stability stated explicitly. For full-model LM input,
report audio and non-audio positions separately. For direct-LM calls,
require bitwise-identical fixed input and LM0 pre-block input before making
any downstream statement. If repeats vary, do not claim a deterministic
factor effect. If contrasts agree, do not infer why the earlier V1 failed;
that failure occurred before the diagnostic forward. Saved tensors verify
arithmetic and provenance, not the actual execution of a model forward.

## One-shot evidence and caps

The V1 failed attempt, including its immutable one-shot reservation and raw
error, must have an independently reviewed manifest. The new runner, gate,
auditor, and tests need source reviews before a later external freeze binds
their exact hashes, the V1 failed manifest and review, old-FIT upstream
sidecars, checkpoint identity, schedule, and caps. An independent freeze
review follows. Only the frozen gate may create a new exclusive reservation
and UTC start receipt. The worker claims it before model construction. The
hard per-process cap is 300 seconds, MPS driver memory is below 30 GiB, and
the combined output cap is below 1 GiB; stdout and stderr each stay below
8 MiB. The CPU audit has a separate 120-second deadline. Earlier model
process time, including the V1 failed attempt, is retained in cumulative
accounting. This study has one attempt, no retry or silent workload change.
Large tensors use a local content-addressed sidecar; small receipts and a
hash/size manifest are committed after the attempt.
