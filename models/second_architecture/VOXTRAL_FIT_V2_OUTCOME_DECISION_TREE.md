# Voxtral v2 FIT causal-assay decision tree

**Written before the full eight-record result was opened.** Apply the frozen [causal design](VOXTRAL_FIT_V2_CAUSAL_DESIGN.md), exact 16-direction denominator, and separate [read-only auditor](audit_voxtral_fit_v2_causal_full.py). A first-record positive result does not decide the full population. Any failed gate remains in the record; no pair, voice, condition or site is dropped after seeing outputs.

```text
Full eight-record run complete, hash-bound, and independently audited?
  no  → report incomplete/invalid; no 16-direction effect label.
  yes → all required passive-hook and self-patch suffix-ID parity at each site?
          no  → no causal interpretation at the affected site.
          yes → at least 12/16 blank baselines restoration-eligible?
                  no  → report blank-base failure; no restoration gate.
                  yes → whole-word restoration ≥12/16 and ≥75% of eligible,
                         and whole-word lexical loss ≥12/16, separately per site?
                          no  → report measured counts; no population effect claim
                                 for the failed arm/site.
                          yes → for each arm/site, whole-word answer successes
                                 strictly exceed every pre/post/padding/random
                                 control AND score movement wins ≥12/16 with
                                 ≥14/16 valid paired scores per control and
                                 zero invalid main/control conditions?
                                  no  → conditional loss/restoration may stand,
                                         but no time-preference label.
                                  yes → FIT-only time-preference label at that
                                         arm/site; freeze a separate fresh-data
                                         transfer protocol before development.
```

Interpret all outcomes at the declared hook: post-encoder block 31 or post-projector. The last encoder block is bidirectional, so a nominal word row may already include other time regions. Conversely, a wrong-time control effect may reflect legitimate distributed content, not an experimental defect. The word-blank base is a constructed feature intervention, and a successful internal state patch is not an attacker-reachable waveform or a security vulnerability.

If the full FIT gate fails, diagnose only with FIT inputs: compare blank-base eligibility, exact generated suffixes, equal-norm realized edits, per-pair/voice counts, and the separately saved feature-splice conditions. Any revised mask, blanking policy, layer, prompt, or answer rule must be a newly versioned study on fresh data; it cannot repair this assay's result. Keep Voxtral development and locked test unopened under this v2 protocol.
