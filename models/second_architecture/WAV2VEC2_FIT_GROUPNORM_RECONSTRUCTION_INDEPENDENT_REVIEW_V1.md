# Independent review: first Wav2Vec2 GroupNorm reconstruction

19 November 2026. **Decision: GO for a posthoc, saved-tensor algebraic reconstruction only.** I reviewed committed source/result `07ddb6b` (`wav2vec2_groupnorm_reconstruction_v1.py` SHA-256 `65bf38aa89bc84fc2546e92def376e55234628e125e1449474a41229b6f4f0b3`; JSON SHA-256 `0de2b2e8846073922e234c9495b11834984dacf0725d51dd76b93e09e736454c`) against the pinned checkpoint, baseline receipt and capture inventory. I ran no model forward or intervention.

The pinned first convolution has kernel 10, stride 5, no padding, followed by `GroupNorm(num_groups=512, num_channels=512, affine=True)` before activation. The baseline capture hooks are placed immediately after `conv0.conv` and `conv0.layer_norm`; thus the stored arrays are the appropriate raw and normalized tensors. The script's output-frame overlap rule, `5j < e` and `5j+9 >= s` for the edited PCM interval `[s,e)`, matches that convolution geometry. Its GroupNorm formula uses each channel's full time axis, population variance, epsilon `1e-5`, and the checkpoint's learned affine weights and biases.

I independently rehashed all **16** common-normalization first-convolution raw/normalized sidecars and matched their sizes and SHA-256 values to the reviewed baseline inventory. A fresh run of the script was byte-identical to the committed JSON. Separately, I constructed outside-window frame indices by slice bounds and called `torch.nn.functional.group_norm` directly on each saved raw tensor. Its output matched every saved post-GroupNorm tensor **exactly** (maximum absolute error 0 in all eight recordings), and the outside-window maxima and frame counts matched the JSON:

| Pair / voice | Outside raw difference | Outside saved GroupNorm difference | Outside frames |
| --- | ---: | ---: | ---: |
| `silver`/`teal`, Kokoro | 0 | 2.7117157 | 13,726 |
| `silver`/`teal`, Piper | 0 | 2.1520462 | 13,508 |
| `plate`/`cup`, Kokoro | 0 | 2.5044632 | 13,726 |
| `plate`/`cup`, Piper | 0 | 4.7978191 | 13,508 |

The script's direct formula differs from the saved outputs by at most `1.1444091796875e-5`, within its frozen `2e-5` check. Its shared-outside-stat calculation gives exact zero outside-window difference because the raw inputs there are identical and the same statistics are applied; this is an **algebraic counterfactual**, not a run of the modified model. The reviewed conclusion is that the actual first GroupNorm numerically accounts for the observed off-word spread under the common-normalized FIT inputs. It does not establish how that spread affects CTC decoding, a unique internal circuit, held-out transfer, or a security vulnerability.
