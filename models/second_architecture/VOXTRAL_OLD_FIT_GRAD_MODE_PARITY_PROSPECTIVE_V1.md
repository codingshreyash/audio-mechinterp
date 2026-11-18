# Old-FIT FP32 graph/inference forward parity diagnostic

18 November 2026. This is a new, one-process numerical diagnosis, not a retry
or completion of the failed Stage-B FP32 replay. The Stage-B FP32 reservation
`20261118T143734566440Z` remains consumed, with parent model-process time
`41.18471045792103` seconds and status `launch_failed`. The failed worker
reported `Direct LM replay mismatch at lm0` while comparing the first
graph-building source replay to a no-grad repeat. Its comparison
function checks LM0 before the projector and LM input. Therefore the error
identifies the first **check** that failed, not the first layer where the
forward passes diverged. Neither trace was saved; no magnitude or first
divergence can be inferred from that receipt.

## Fixed question and inputs

On the same one old-FIT `beaver` audio request, do identical FP32 source
overrides and LM inputs produce repeatable forwards within mode? If so, does
the composite graph-building condition differ from inference mode, and where
is the first **captured** difference? This study does not measure a learned
concept, a security property, a new word or a new speech sample.

The exact archived BF16 clean source is cycle zero, positive alpha `0.00625`
for each site in the reviewed Stage-A BF16 sidecar. The two sites, in order,
are post-encoder layer 31 (`encoder31`, shape `[1,1500,1280]`) and post-audio
projector (`projector`, shape `[375,3072]`). The saved source is cast to FP32
without changing values. Its full tensor SHA-256 and its reviewed Stage-A
manifest, posthoc audit and result hashes must be checked before model load.
The original BF16 Stage-A launch must retain `launch_failed`; the Stage-A
FP32 continuation and BF16 Stage-B direct replay must retain their independently
reviewed results. The failed Stage-B FP32 attempt, its one-shot reservation,
raw bytes and independent failed-attempt review are mandatory prerequisites.

Use the same pinned local Voxtral checkpoint, task batch, parameter-layout
rule, `sdpa` attention implementation, `eval()` mode and offline environment
as Stage B. Model weights have `requires_grad=False`. `N` means the exact
no-grad path used by Stage B: a nondifferentiable source leaf under
`torch.no_grad()`. `G` means its exact graph-building path: a differentiable
source leaf under `torch.enable_grad()`. These conditions change two factors
together; a result cannot separate global grad mode from leaf requirements.
No VJP or optimizer step is performed.

## Fixed forward schedule

Run the entire schedule in one fresh FP32 model process. For `encoder31`, run
four complete model forwards using the same archived clean source in this
order: `N,G,G,N`. For `projector`, run `G,N,N,G`. The site orders counterbalance
the mode order. Capture and save each full returned posthook source, full
post-projector state, full language-model input, full LM0 pre-block input,
terminal post-block states at LM0, LM7,
LM14, LM22 and LM29, and full first-answer logits. Capture source input hash,
audio-placeholder positions, mode, site, call ordinal, exact batch `input_ids`, `attention_mask` and
`position_ids` when present, model `eval()`/SDPA configuration, and the
PyTorch/MPS version. Require exactly one invocation per source and downstream
hook, correct shapes, finite tensors, and the requested graph/no-grad mode,
correct source
override and projector-to-LM placeholder equality, but do not assert
cross-forward parity during the run.

Then, for each site in the same order, choose the **first** saved N and first
saved G full LM inputs from that site's four full-model calls. For each fixed
origin input (N then G), perform four direct language-model forwards in order
`N,G,G,N`. This yields 16 direct-LM forwards: two sites × two fixed input
origins × two execution modes × two repeats. Save the exact input, full LM0
pre-block input, the same five terminal layer vectors and first logits for
each call. The full-model and direct-LM counts are exactly eight and sixteen.
The direct calls use the same attention mask, position IDs, final-token head
and FP32 model weights; they omit audio preprocessing. No complete-answer
generation, VJP or edited BF16 source is in scope.

## Analysis and interpretation

An independent CPU auditor rehashes every retained tensor and recomputes all
pairwise comparisons at each ordered boundary. Report bitwise equality,
changed-coordinate count, maximum absolute difference, L2 difference and
relative L2 in float64 for all four N/G pairings. Define symmetric relative
L2 as `||a-b||₂ / max(||a||₂, ||b||₂)` when either norm is positive and null
when both are zero. For the within-mode repeats of an identical input,
compute their L2 spread; set the boundary resolution floor to
`max(3 × maximum within-mode spread, 1e-6)` using float64 arithmetic. With
two repeats per mode, this is an observed floor and not a statistical noise
estimate. Report cross-mode differences
whether resolved or not, without discarding any pair. Check all full-model
source, projector and LM inputs (including the non-audio positions) before
post-LM0 states. Report the first bitwise differing saved boundary separately
from the first boundary exceeding the resolution floor. A direct-LM fixed-input
comparison may be called a downstream mode contrast only if the input and
LM0 pre-block input are bitwise identical and the within-mode repeats are
stable at that boundary. A full-model contrast may be localized to no earlier
than the first **saved** differing boundary. A difference within N/N or G/G
is repeat instability and prevents a deterministic mode conclusion. If every
new comparison agrees exactly, the original failure remains a preserved but
unreproduced event. Do not infer a particular MPS kernel, a model-level audio
mechanism or a security issue from this one case. A later orthogonal study
would need to vary grad context and leaf `requires_grad` separately and, if
feasible, compare an alternative backend on the same exact input.

## One-shot execution and evidence

The new runner, gate, CPU auditor and their tests require separate source
reviews. A later externally committed, first-add-immutable freeze binds all
source, independent reviews, exact failed Stage-B evidence, old-FIT sidecar
hashes, checkpoint identity, schedule and caps. An independent freeze review
must follow. Only then may the gate create an exclusive reservation and UTC
start receipt before the child. The worker claims the reservation exclusively
before any model construction. A parent watchdog kills the process group at
the hard 180-second model-process deadline. The observed MPS driver cap is
30 GiB, serialized worker plus launcher output is below 1 GiB, stdout and
stderr are each below 8 MiB, and the CPU saved-data audit has a separate
120-second deadline. Model time for the earlier failed Stage-B FP32 attempt is
retained in the new receipt's prior-work accounting. There is no retry,
workload reduction or return to the failed Stage-B phase after reservation.
The worker is capped below 1 GiB to reserve space for launcher captures and
the final receipt. If a subprocess writes past a polled stream or combined
size limit, the launcher records its original byte count and SHA-256, keeps
the first 512 stream bytes, and writes a failure receipt; the discarded stream
tail cannot later be reconstructed from the local files.
The full raw tensor bundle uses a local content-addressed `.pt` sidecar;
small receipts and a SHA/size manifest are committed. A source-only scaffold
must fail closed until the CPU auditor and all external reviews exist. Saved
tensors can verify arithmetic and provenance, not independently attest that
forwards executed or why an MPS kernel behaved as it did.
