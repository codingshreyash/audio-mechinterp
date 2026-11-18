# Fit-pool capability result: independent saved-data audit

**Verdict:** The saved result is internally consistent. The 180 versus 87 gap is a genuine limitation of the frozen *single-token alias* coverage, not a counting or decode error. This review did not rerun Qwen.

I recomputed SHA-256 bindings for the committed runner, frozen spec, rendered manifest, progress JSONL, and memory JSONL. All match the result. Reconstructing the 192 cases from frozen fit-pool inputs reproduces the recorded `cases_sha256`; every saved case field and ID agrees with its planned case. The 192 progress rows exactly equal the 192 report rows. Independent classification and per-category, voice, template, and map-fit count arithmetic reproduced the saved counts. The pinned tokenizer independently decoded every saved token sequence to the recorded raw and special-token strings.

| Outcome across all 192 planned clips | Count |
| --- | ---: |
| Ended on EOS | 192 |
| Exact expected word plus EOS | 180 |
| Semantic expected word plus EOS | 180 |
| First generated token in expected one-token alias set | 87 |
| Correct answer with expected first-token alias | 87 |
| Correct answer with another tokenization | 93 |
| Incorrect answer | 12 |

Of the 93 correct answers outside the one-token alias set, 85 used two word tokens and eight used three word tokens, followed by EOS. For example, the correct generated `Lion<|im_end|>` has token IDs `[43, 290, 151645]`, while the frozen one-token `lion` variants have IDs `[32099, 39032, 78151]`. Thus the first-token alias set covers **87/180 = 48.3% of correct complete answers** in this fit sample. It can serve as a restricted first-step log-odds readout, but it does not measure the full probability of saying the word. Keep full-answer accuracy as a separate endpoint; consider a prespecified sequence-level candidate score as a secondary readout before new evaluations.

The 12 incorrect answers remain in the denominator. They include speech/lexical confusions such as `deer`→`Dear`, `grove`→`growth`, and `pillow`→`palo`, plus one longer explanatory answer. The memory log contains one post-load sample and 192 post-case samples; peak MPS driver allocation was **18.635 GiB**, with zero recorded samples above the 30 GiB limit. This is fit-only capability evidence on synthetic audio, not a held-out Jacobian-lens result.
