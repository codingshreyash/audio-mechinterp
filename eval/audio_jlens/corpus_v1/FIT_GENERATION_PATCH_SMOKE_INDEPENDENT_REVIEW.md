# Independent review: FIT-only Stage A generation-patch smoke

**Verdict: no plumbing blocker for Stage A.** This smoke checks one predetermined FIT clip and the generation hook. It is not a held-out transport result, a task-validity pass, or a security finding. No development or locked-test model input/output was used in this review.

## Provenance and executed path

- Result: `fit_generation_patch_smoke_v1.json`, SHA-256 `ecac2511d73be0312a9f8e6161538d1dfcbcc6f05315c533a2d52d165c7065da`; memory sidecar SHA-256 `4cb3820a9e3da7619fa221d574d28cee6578d276c64fe5be01a50fef9cc7a3c8`. The result's memory hash matches the sidecar.
- Every source SHA stored in result metadata matches the current frozen source/artifact: smoke runner `853167a8…`, Stage A runner `6f314b7a…`, FIT report `bd5b281d…`, stage-1 tensor `8276aca0…`, lens `fef84a20…`, and V2 scoring policy `e7ea18cd…`. The FIT report independently binds the same stage-1 and lens hashes.
- Deterministic first map-fit context: `animal_06/fit_chosen/kokoro/deer`; donor is its same-pair, same-template, same-voice `moose` clip. The script calls the same `generation_result` helper frozen for Stage A and passes an expected full source shape to its hook.

## Checks

| Check | Encoder 31 | Connector |
|---|---:|---:|
| Source shape matches recipient **and donor** FIT stage-1 records | `[281,1280]` | `[140,2048]` |
| Six selected row counts match recipient **and donor** records | `17,16,17,16,17,16` | `8,9,8,8,8,9` |
| Source hook calls at α=0 and α=.025 | `1,1` | `1,1` |
| Generated token IDs: clean, α=0, α=.025 | `[30665,151645]` in all three | `[30665,151645]` in all three |
| Intended α=.025 edit Frobenius norm from saved donor/recipient bin means | `2.162` | `2.424` |

The clean generation used no patch hook. Each site's α=0 generation exactly matches clean token IDs; α=.025 used a nonzero FIT source delta and the expected hook once. Both edited generations happened to keep the same answer. The smoke output does not save post-hook activation values or a score change, so it establishes the intended edit path through code and nonzero saved delta, **not** a measured behavioral effect at α=.025.

The pinned tokenizer independently reproduces all five saved raw decodes, special-token decodes, and strict classifications. The complete sequence is `Dear<|im_end|>`: EOS is terminal, with no hidden special token or early EOS, but `Dear` is the wrong answer to the scripted `deer` request. Therefore `semantic_word_plus_eos=false` and the V2 prefix class is `unclassified` in every outcome. This is the known FIT item error, retained because the clip was chosen by frozen case order. It must stay in any all-planned denominator; the smoke must not be cited as an answer-accuracy success.

The six memory samples cover model load, clean generation, and both α conditions at each site. Peak driver allocation was **18.62 GiB**, below the **30 GiB** cap; all `over_limit` flags are false. The sidecar is a sampled upper bound at those checkpoints, not a continuous memory trace.

**Scope:** The integration checks exact-case hook operation and token-structure accounting. Development still needs its own 96-direction planned denominator, valid source/row checks, score and answer comparisons, and the predeclared gate. Stage B controls and complete-sequence scoring remain separate requirements for a positive claim.
