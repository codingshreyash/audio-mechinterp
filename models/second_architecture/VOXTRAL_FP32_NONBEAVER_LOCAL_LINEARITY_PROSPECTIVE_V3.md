# Prospective FP32 non-beaver local-linearity pilot — V3 output-budget amendment

18 November 2026. **Source-only amendment; no model run or executable freeze.**
The independently conceptually reviewed [V2 protocol](VOXTRAL_FP32_NONBEAVER_LOCAL_LINEARITY_PROSPECTIVE_V2.md) remains immutable. V3 changes only its impossible 1 GiB serialized-output cap. All four fixed contexts, source sites, G1 and N0 definitions, site-matched clean/zero topology, six signed alphas, two cycles, exact 384 patched forwards, 32 VJPs, eight G1 clean graphs, eight N0 clean references, four donor captures, eight natural generations, 300-second process cap, 30 GiB MPS-driver cap, acceptance arithmetic, failure handling, and claim limits remain as in V2. This amendment requires a new independent conceptual review, implementation/source review, byte estimate check, and external freeze before any weight load.

## Why the V2 cap cannot preserve its required evidence

Each context has 48 paired zero forwards: 24 at encoder 31 and 24 at the projector. In FP32, a full encoder-31 source is `1×1500×1280×4 = 7,680,000` bytes, a full projector state is `375×3072×4 = 4,608,000`, a full language-model input is `396×3072×4 = 4,866,048`, and a 131,072-wide first-logit vector is `524,288`. Conservatively counting full prepatch and returned source, full projector, full LM input, and logits for every zero gives

`24 × (2×7,680,000 + 4,608,000 + 4,866,048 + 524,288) + 24 × (3×4,608,000 + 4,866,048 + 524,288) = 1,069,744,128 bytes`.

This leaves only 3,997,696 bytes under 1 GiB (`1,073,741,824` bytes), before G1/N0 clean references, donor states, 48 edited forwards, gradients, batch tensors, metadata, serialization, or launcher receipts. It is therefore not a feasible hard cap for the V2 evidence requirement. We will not omit full zero captures, alter the schedule, or use a hash claim in place of a retained tensor to make the old cap pass.

A conservative additional **tensor-payload** estimate per context, using the V2 schema, is:

| Component | Piper bytes | Kokoro bytes |
|---|---:|---:|
| 48 zero full captures above | 1,069,744,128 | 1,069,744,128 |
| Four site-specific G1/N0 clean references in this context | 89,145,344 | 89,145,344 |
| One donor full encoder/projector/LM input/logit capture | 22,286,336 | 22,286,336 |
| 48 edited selected-row pre/post/effective tensors and full logits | 60,555,264 | 68,001,792 |
| Three LM terminal vectors on 96 patched forwards | 3,538,944 | 3,538,944 |
| Four VJPs at each of two sites for this context | 1,966,080 | 2,379,776 |
| Recipient and donor CPU batch tensors | 3,084,672 | 3,084,672 |
| **Estimated tensor payload** | **1,250,320,768** | **1,258,180,992** |

The table deliberately double-counts the projector source and passive projector capture on projector-site zero calls. It does not count Python/torch serialization overhead, clean/donor LM terminal vectors, progress records, full-tensor hash strings, selected zero rows, or launcher stdout/stderr. The estimate is a partial, intentionally conservative tensor budget, not a measured output size. An independent source review must recompute this table from the implemented record schema and include all retained fields before freeze. If that implementation estimate reaches the new cap, stop and version a new resource plan before launch.

## Revised hard file and disk limits

The **combined per-context** local output cap is now **2 GiB = 2,147,483,648 bytes**, including the worker output directory, content-addressed tensor sidecar, CPU audit and result, launcher capture directory and final receipt. The worker's own output limit must be set lower than 2 GiB by a fixed reserved allowance for capped stdout/stderr and final launcher/audit receipts; the exact allowance is frozen in source and externally reviewed. The parent launcher must enforce the combined bound and abort the attempt on overflow. Stdout and stderr remain individually capped at 8 MiB. The CPU saved-data audit retains its separate 120-second deadline.

Across all four fixed contexts, the **combined local campaign output cap is 8 GiB = 8,589,934,592 bytes**, including all four worker directories, sidecars, launch captures, audits, and receipts. The prospective tensor-payload sum is `2×1,250,320,768 + 2×1,258,180,992 = 5,017,003,520` bytes before serialization and launch overhead. Before campaign reservation, check that the selected filesystem reports at least **12 GiB available** and that no campaign directory or reservation already exists. Do not delete or overwrite an earlier attempt to make room. A failed context remains failed in its planned denominator; no retry, reallocation, compression-only fallback, or change of evidence fields is authorized. If an actual context cannot fit the frozen 2 GiB cap, stop the campaign and prospectively revise the method or environment; do not silently enlarge the cap after seeing outputs.

The four contexts must be reserved together before context zero. Each model process still has the fixed 300-second and 30-GiB MPS limits. The extra disk allowance does not authorize more model time, memory, forwards, VJPs, generation, or data access. Large sidecars remain local and content-addressed; small raw receipts and exact hash/byte manifests are committed. The CPU auditor must verify the saved bytes that are available and explicitly mark runtime-only hash claims that cannot be recomputed from a retained full tensor.
