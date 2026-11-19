# Pre-inference amendment: exact descriptive processor and generation arguments

19 November 2026. **Source-only. No causal candidate forward has run.** During executable implementation, direct inspection of the valid V2 [descriptive worker](descriptive_worker_v1.py) found an error in the original causal design prose: it said no attention mask was supplied and omitted two generation output flags. This amendment supersedes only that argument list. The [V2 donor amendment](WHISPER_NONSPEECH_CAUSAL_DESIGN_AMENDMENT_V2.md) still controls the 328-call schedule and no-hook/read-only distinction.

Every causal call must use the same processor and generation arguments as the valid descriptive V2 result:

```python
processed = processor(audio, sampling_rate=16000, return_tensors="pt",
                      return_attention_mask=True)
result = model.generate(
    input_features=processed.input_features,
    task="transcribe", language=None,
    attention_mask=processed.attention_mask,
    return_timestamps=False, force_unique_generate_call=True,
    condition_on_prev_tokens=False, do_sample=False, num_beams=1,
    max_new_tokens=128, output_scores=True,
    output_logits=True, output_hidden_states=True,
    return_dict_in_generate=True)
```

The processor attention mask is an explicit part of the source pipeline and must be saved/checked as 200/3,000 or 800/3,000 active mel frames for these 2- and 8-second sounds. The installed Whisper encoder block itself receives `None` for its attention mask, but the generation wrapper may use the processor mask for segment accounting. Do not infer that passing the mask is behaviorally irrelevant. Actual no-hook complete prefix/generated IDs through EOS must match the valid V2 descriptive record for every one of the 20 unique WAVs, **byte for byte as token-ID sequences**, before any causal interpretation. Stage B repeat no-hook and all read-only/self arms also need exact token parity against the same run's true no-hook case. A discrepancy remains in the ledger and invalidates that case's causal reading; it is not tuned away or silently called a new baseline.

No prompt, waveform, model pin, target site, schedule arm, random seed, decoder logit processor, output endpoint, or resource cap changes here. This correction must pass a new independent source/design review before any executable GO or model load.
