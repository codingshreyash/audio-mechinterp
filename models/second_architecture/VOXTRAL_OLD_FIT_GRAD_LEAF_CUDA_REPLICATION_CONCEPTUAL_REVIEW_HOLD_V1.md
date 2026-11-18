# Independent conceptual review: CUDA grad-leaf replication V1

18 November 2026. Reviewed design commit `d3ab80a` and protocol SHA-256
`188925bc4022aa9841502d310899b5f9b28bf8ec0ba32ab7f7acb3fc45579050`.
**Decision: HOLD for implementing an executable CUDA replication from this
protocol.** This review started no Brev instance and loaded no model weights.

## What is fixed and supported

The two planned inputs are the archived old-FIT beaver request and the already
inspected Piper `nature_09` waterfall request. I rehashed the protocol, prior
MPS factorial protocol and result review, FIT selection and masks, waterfall
WAV, and the local 509,990,581-byte MPS sidecar; their SHA-256 values match
those stated in the design and prior manifest. The waterfall WAV is in the
frozen FIT selection. The reported CPU processor tensor hashes remain a
prospective gate for the CUDA implementation and freeze; I did not run the
processor in this review.

The plan preserves the MPS `N0/N1/G0/G1` factors, site-specific eight-call
orders, fixed-input direct replay, two repeats, source and LM boundaries, and
full first-logit vectors. It correctly separates bitwise differences from
differences above an observed repeat floor. It names two cases and four
site/case cells, treats the MPS result as a descriptive reference, and does
not turn a numerical match into an audio-specific or security claim. CUDA
source, checkpoint and environment identity, deterministic settings, a
42-GiB L40S cap, byte-budget pilot, one-shot claims, separate reviews and
external freeze are specified before any weight load. The observed Brev
quotes in the design are not a cost guarantee.

## Blocking decision-rule gap

The `same_pattern` rule requires direct language-model replay to
“reproduce” the corresponding full-model post-LM0 and first-logit contrast,
but never defines whether reproduction means bitwise identity of each
condition's tensors, equality of the G1-minus-N0 difference, or agreement
within a numeric tolerance. Those tests can disagree for the same saved
outputs. The prior MPS result used **bitwise equality of every corresponding
full-model and direct-replay post-LM0 state and first-logit tensor**. The
CUDA design must state an exact direct/full comparison rule and its outcome
when the rule fails, before implementation or CUDA outputs. A conservative
amendment could require bitwise direct/full equality at LM0 pre-input,
post-LM0 and first logits for each case/site/condition/repeat, and label a
valid mismatch `different_pattern`. If a numeric tolerance is chosen
instead, fix the formula, scope and value prospectively and justify it from
already-open data. Keep runtime/hook/provenance failures `invalid` and do
not change the floor after seeing CUDA output.

After an amended protocol is committed, independently rereview its exact
bytes. This hold does not prevent CPU-only planning or resource inventory;
it prevents a subjective `same_pattern` label or an executable freeze under
the current design. The Mac MPS four-context campaign remains separate.
