# Independent rereview: Whisper FIT word-window causal assay V2

18 November 2026. Review target: committed `WHISPER_FIT_WORD_WINDOW_CAUSAL_PROSPECTIVE_V2.md` at `2705225`, read together with V1 and its preserved independent HOLD review. **Decision: GO for implementation, subject to a separate runner source review before model inference.** No model was loaded or run in this rereview.

V2 resolves each V1 HOLD point:

1. Installed Transformers 5.12.1 `WhisperEncoder.forward` computes `conv1`, `conv2`, permutation, positional addition, dropout, then passes the resulting tensor as the first positional argument to `encoder.layers[0]`. The specified layer-0 forward pre-hook is the correct site after these operations. The pre-hook must preserve the second positional `None` attention argument and any keyword arguments when replacing only the first tensor.
2. The same source runs `encoder.layers[1]` before the remaining blocks and applies `encoder.layer_norm` after block 3. V2 now uses the output of `encoder.layer_norm` as its final intervention site, which is the normalized tensor passed to decoder cross-attention. It correctly labels post-block 3 as an observation boundary *before* that normalization. The final-site all-row donor swap is therefore a meaningful exact-token positive control.
3. V2 defines the Gaussian control against the Frobenius norm of the donor-minus-recipient difference over the full selected window `W`, with local fixed seeds, zero/nonfinite stop, and stored norms/edit hash. It does not imply per-row norm matching. The count remains 4 baseline plus 48 intervention decodes.

The V1 frozen inputs, windows, directions, failure logging, and modest claim scope remain in force. An off-slot prefix patch remains an intervention-location control rather than a guaranteed zero-information control, because the encoder is bidirectional. This review approves the *prospective specification* only. The concrete hook implementation, generation parity, hashing and result auditor still need independent source checks before inference.
