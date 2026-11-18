# Independent result audit: Voxtral v2 FIT first-record causal benchmark

**18 November 2026.** Read-only audit of [the completed first-record run](runs/voxtral_fit_v2_causal_first_record/20261118T091057373532Z/result.json) with [an independent CPU auditor](audit_voxtral_fit_v2_causal_first_record.py). No Voxtral model inference, MPS run, or development/locked-test input or answer inspection. The run uses only `animal_08/piper/fit_chosen`, words **beaver/monkey**, selected before the v2 natural pilot.

## Provenance and completeness

The auditor rehashed the frozen runner, design, mask manifest, v2 selection, corpus index, natural-pilot result/gate, selected WAVs, checkpoint configuration and both weight shards, tokenizer/processor files, and installed Transformers sources. All match run metadata. It independently reconstructed the two CPU processor inputs and verified native, blank, and six-condition feature hashes. It decoded every generated suffix with the pinned tokenizer, checked the complete-answer classifier and candidate-score subtraction, verified hook-event counts/shapes/BF16 dtype and edited-row masks, checked BF16 control norms and random seeds, and checked natural/blank passive and self-patch token-ID parity. The append-only progress hash matches the terminal result.

| Planned first-record unit | Independently observed |
| --- | ---: |
| Source-state captures | **8/8** |
| Complete generations | **100/100** |
| Teacher-forced candidate-score forwards | **200/200** |
| Unique progress slots | **108/108** |
| Invalid conditions | **0** |

The progress SHA-256 is `02dd6d7ab85d2595c6db6e375c23c40531332a355b166276eae745c5dd0423ca`. The CPU auditor passed at source SHA-256 `234a552cb0f221fc1802ddb1fce9e411841f216f1811a9cd80ac187766e92d5a`; `py_compile` also passed. Model logits cannot be independently recomputed from saved scalar scores without another model run; the audit verifies the pinned scoring code and saved contrast arithmetic, not a second numerical forward.

## FIT-only observed outcomes

Both blank baselines yielded the quoted string `"chose"` with EOS, classified as `malformed_surface`, so **2/2** blank bases are restoration-eligible. For each hook site, the whole-word clean-state patch changed the blank run to the exact recipient word with EOS in **2/2** directions. The reciprocal blank-state patch changed each clean answer to the valid single word `None` with EOS in **2/2** directions. Under the frozen rule, `None` is `other`, so this meets the assay's lexical-loss definition; it is **not** evidence that the model chose the opposite target word. Opposite-word clean-donor patches gave the opposite target word in **2/2** directions at each site.

| Site | Whole-word restoration | Whole-word lexical loss | Opposite-donor control |
| --- | ---: | ---: | ---: |
| Encoder layer 31, post-block | 2/2 planned, 2/2 eligible | 2/2 planned (`None`) | 2/2 opposite |
| Post-projector connector | 2/2 planned, 2/2 eligible | 2/2 planned (`None`) | 2/2 opposite |

As a descriptive check on this first record, pre/post/padding and random controls produced no complete restoration; pre/post/padding produced no complete lexical loss. The projector random loss control changed **1/2** answers to `other`, so the random control is not inert. Whole-word signed canonical score movement exceeded each same-arm control in both directions at both sites. This is a two-direction observation, not the preregistered 16-direction time-preference gate. The four word-quarter outcomes are retained separately without selecting a best bin.

The measured first-record variable time is **83.42 s**, fixed preflight plus load is **7.80 s**, and the predeclared eight-record projection is **675.17 s** against a **2700 s** cap. Peak MPS driver allocation is **10,667,950,080 bytes** against **32,212,254,720 bytes**. The projection gate passes with a complete first-record workload; it is a feasibility estimate, not evidence the remaining seven records will have identical cost or effects.

## Verdict and scope

**No audit blocker found in the saved first-record result.** This is causal evidence for the specified **constructed feature blanking and activation patches** on one synthetic Piper pair and two word directions, with exact full-answer changes at both sites. It does not meet or test the design's 12/16 FIT population thresholds, establish a natural waveform intervention, localize a unique circuit, or demonstrate a security vulnerability. The next defensible step is the separately frozen remaining FIT run, preserving all 16 planned directional cases and controls before any development or locked-test use.
