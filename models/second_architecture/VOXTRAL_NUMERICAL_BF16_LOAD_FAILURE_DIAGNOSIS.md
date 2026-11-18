# Old-FIT numerical BF16 load failure diagnosis — 18 November 2026

The first externally frozen BF16 launch stopped at the model-load dtype assertion before generation, any activation capture, or any intervention. The immutable attempt is `runs/voxtral_v1_fit_backward_numerical_launch_v1/bf16/20261118T120738563890Z/`; its `runner/error.json` says `invalid_old_fit_arm`, `ValueError: Full-pipeline parameter dtype/device changed`. The launch gate's CPU auditor accepted only the failed-attempt receipt. This is **no numerical result** and cannot count toward the BF16/FP32 comparison.

I independently loaded the same pinned Voxtral checkpoint offline on local MPS with `dtype=torch.bfloat16`, `low_cpu_mem_usage=True`, `device_map="mps"`, and `attn_implementation="sdpa"`, under the same 30-GiB MPS fraction guard. Of 762 named parameters, 761 were BF16 on MPS and exactly one was FP32 on MPS:

| Parameter | dtype / device | shape |
|---|---|---|
| `model.audio_tower.embed_positions.weight` | `torch.float32` / `mps:0` | `(1500, 1280)` |

This is intentional in the installed pinned Transformers source, not an unexpected partial device move. `VoxtralForConditionalGeneration` declares `_keep_in_fp32_modules_strict = ["embed_positions"]` in `transformers/models/voxtral/modeling_voxtral.py`. `transformers/modeling_utils.py` applies that strict plan for either FP16 or BF16 loads. The audio tower adds that FP32 positional table to convolution embeddings and then casts the result to the convolution dtype. A blanket all-BF16 assertion in the frozen numerical runner therefore rejects the library's normal BF16 model. The source itself is bound by the prior `modeling_voxtral` SHA-256 in the attempt metadata.

**Recommended prospective correction:** keep the model load unchanged and change only the assertion: on the BF16 arm, require exactly `model.audio_tower.embed_positions.weight` to be FP32 on MPS and every other parameter to be BF16 on MPS. On the FP32 arm, require all parameters FP32 on MPS. Record the observed parameter dtype/device counts in the next attempt's metadata. Reject any additional exception. Do not cast the positional table to BF16; doing so would change the published model pathway and break comparability with the earlier BF16 pilot. Any corrected runner requires a new source review and external freeze receipt, while the failed launch remains preserved under its original freeze.

The inventory process exited before a second model run. No checkpoint or experiment result was modified by this diagnosis.
