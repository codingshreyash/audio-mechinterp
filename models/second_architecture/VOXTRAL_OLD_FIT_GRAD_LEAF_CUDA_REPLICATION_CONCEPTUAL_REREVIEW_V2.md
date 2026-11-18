# Independent conceptual rereview: CUDA grad-leaf replication V2

18 November 2026. Reviewed amendment commit `7552a08`, SHA-256
`c86537552b55f6542c88d2be9a7c71d5e1cecd589c6a11cc89eec514c49419ca`,
against unchanged V1 SHA-256
`188925bc4022aa9841502d310899b5f9b28bf8ec0ba32ab7f7acb3fc45579050`
and the preserved conceptual HOLD SHA-256
`c957310d936b548497fc96a17bb63be56d09edcb277cea83b801603c8cc6f4a3`.
**Decision: GO for implementing CUDA replication source and auditor only; no
CUDA model work until independent source reviews and external freeze review.**
No cloud instance was started or model weight loaded in this review.

The V2 amendment resolves the HOLD's undefined direct/full replay criterion.
It joins calls by exact case, site, condition and within-condition repeat,
then compares contiguous CPU raw bytes, including signed zero, at full versus
direct LM0 pre-input, post-LM0 state and full first-answer logits. The
arithmetic is fixed: two cases × two sites × four conditions × two repeats
gives 32 call pairs; three boundaries give 96 comparisons. Each case/site
cell contains eight pairs and 24 required comparisons. Shape and dtype must
match before byte comparison. A missing, duplicated or invalid call makes the
cell `invalid`; valid captures with any required byte mismatch make it
`different_pattern`; `same_pattern` requires all 24 direct/full checks plus
every unchanged V1 condition. The separate within-condition repeat floor
cannot relax direct/full equality.

I verified the V1 protocol and HOLD bytes still match the hashes above. The
two fixed already-open FIT inputs, CUDA environment and resource controls,
four-cell denominator, and separation from the reserved Mac campaign remain
as reviewed in the V1 HOLD. This GO approves an unambiguous implementation
target, not the feasibility of CUDA execution, a selected kernel explanation,
cross-backend equivalence, an audio-specific mechanism, or a security result.
The runner, auditor, launch gate, tests, CPU byte-budget preflight, environment
receipt and later executable freeze still need their own independent reviews
before any checkpoint load.
