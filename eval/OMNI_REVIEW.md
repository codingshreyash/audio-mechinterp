# Independent review: Qwen2.5-Omni-3B Thinker adapter

Reviewed 2026-11-17. Scope: [`models/qwen_omni/adapter.py`](../models/qwen_omni/adapter.py), its [loaded smoke](../models/qwen_omni/loaded_smoke_verified.json), the installed pinned Transformers source, and one independent loaded-model rerun. This is an adapter/plumbing review; no lens was fitted.

## Verdict

The narrow loaded-model claim holds: the pinned Thinker processes the WAV through its audio tower, the observation hooks leave the full forward logits unchanged, and a final-position token logit has a nonzero gradient through input audio features, the first audio block, the audio projection, and a text block. This verifies a usable gradient path for a future audio-conditioned fit. It does **not** verify an interpretable readout, causal intervention, generated answer, or speech-output behavior.

| Check | Evidence |
|---|---|
| Checkpoint/source identity | `pins.json` specifies `Qwen/Qwen2.5-Omni-3B` revision `f75b40e3da2003cdd6e1829b1f420ca70797c34e`. The local HF snapshot directory has that commit, `config.snapshot.json` hashes to the recorded `20790f...`, and installed Transformers `direct_url.json` names commit `e7e568dc7673ac99a87660c7c50e84830268285c`. `load_pinned_thinker` passes the checkpoint revision to both processor and Thinker and defaults to local cache. |
| Processor route | `prepare_audio_batch` passes a local `{"type":"audio","audio":path}` item to `Qwen2_5OmniProcessor.apply_chat_template` with `tokenize=True`. Independent processor inspection found 158 **total** input positions, of which 137 are audio placeholders; the feature mask contains 550 valid mel frames. The source WAV is 16 kHz mono, 5.5 s, SHA-256 `ba226916abdd36bef212a0b478d49eb63028fb541d84a7934f50e4871fe6d21d`. The padded `input_features` tensor is `[1,128,30000]`; do not mistake 30,000 padded frames for 5.5 s of content. |
| Where audio enters text | Pinned Transformers `Qwen2_5OmniModel.forward` calls `get_audio_features` and `masked_scatter`s projected audio into text embeddings at audio placeholders ([source](https://github.com/huggingface/transformers/blob/e7e568dc7673ac99a87660c7c50e84830268285c/src/transformers/models/qwen2_5_omni/modeling_qwen2_5_omni.py#L1891-L1898)). The audio encoder runs 32 blocks, pools and projects 1280-wide activations to 2048-wide text features ([source](https://github.com/huggingface/transformers/blob/e7e568dc7673ac99a87660c7c50e84830268285c/src/transformers/models/qwen2_5_omni/modeling_qwen2_5_omni.py#L862-L899)). |
| Hook integrity | `ActivationCapture` returns `None` from each forward hook; its detached path copies tensors but does not replace output. The loaded smoke and independent rerun both found maximum absolute difference `0.0` across complete logits with versus without hooks. The independent rerun found all hooks removed after context exit. |
| Gradient and input dependence | Independent rerun selected first reply token `39814` (`"Sure"`). Gradient norms from its logit: mel `input_features` **35.49**, audio block 0 **10.10**, audio projection **0.839**, text block 17 **0.888**. Zeroing the mel tensor while keeping tokens/mask fixed moved that token's logit by **4.125** and some logit by **24.0**. This out-of-distribution control checks audio dependence only; it does not demonstrate acoustic or semantic localization. |

Reproduce the independent audit with:

```bash
cd /Users/shrranjan/audio-mechinterp
HF_HOME=$PWD/models/qwen_omni/.hf_cache \
  models/qwen_omni/.venv/bin/python eval/omni_adapter_audit.py \
  > eval/omni_adapter_audit.log 2>&1
```

The machine-readable output is [`omni_adapter_audit.json`](omni_adapter_audit.json). The audit loads cached BF16 weights on local MPS; it does not download a model or use cloud compute.

## Caveats for the next milestone

1. The script calls `model.forward` on the full prompt; it does not call `generate` or decode a complete answer. Its logit is the first reply token, `"Sure"`. A readout benchmark must freeze a target token/position definition and record real generated text before scoring positions.
2. This is the **Thinker** text-logit path only. The Qwen Omni Talker, audio decoder, and native speech output are outside the check. Results cannot be described as speech-to-speech behavior yet.
3. Nonzero gradients to module outputs and mel features establish differentiability, not a fitted Jacobian lens. The 0.15 s single-logit backward in the smoke is not a full 2048-coordinate Jacobian cost estimate. The next one-prompt benchmark must measure that separately.
4. The processor expands the spoken input into 137 audio placeholders; a 275×1280 audio-layer capture and 137×2048 projection capture are compressed/chunked representations. Their row indices are not word timings. A claim about where a spoken concept appears needs independent alignment and negative controls.
5. The local processor emitted generic configuration warnings for several audio/vision/TTS special-token IDs relative to an 8448-token sub-vocabulary. The Thinker forward and audio-token count succeeded, so these warnings did not block this smoke. Keep the pinned versions fixed and check generation separately.
6. The sole WAV is synthetic Samantha speech. One prompt, one device, BF16, and an out-of-distribution zeroed-feature control do not establish stability across speakers, languages, natural recordings, or precision modes.
