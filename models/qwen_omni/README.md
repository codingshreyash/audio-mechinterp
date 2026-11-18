# Qwen2.5-Omni-3B Thinker adapter scaffold

This directory records a source-audited adapter for the first audio-aware Jacobian-lens experiment. Its lightweight interface tests do not load weights; a separate, pinned loaded-model smoke test has now passed on local MPS. Checkpoint weights remain in an ignored local cache and are not bundled here.

## Exact pins and source audit

- Model: [`Qwen/Qwen2.5-Omni-3B`](https://huggingface.co/Qwen/Qwen2.5-Omni-3B) at revision `f75b40e3da2003cdd6e1829b1f420ca70797c34e`.
- Full upstream config: [config.snapshot.json](config.snapshot.json), SHA-256 `20790f362c37a1718a3e764f597ae33dfc20177399762018dffaabf8321d4dd1`.
- Transformers source: commit `e7e568dc7673ac99a87660c7c50e84830268285c`, not an unpinned `main` install. The exact install command for a later environment is `python -m pip install 'git+https://github.com/huggingface/transformers.git@e7e568dc7673ac99a87660c7c50e84830268285c'`.
- [Source lines 816–845](https://github.com/huggingface/transformers/blob/e7e568dc7673ac99a87660c7c50e84830268285c/src/transformers/models/qwen2_5_omni/modeling_qwen2_5_omni.py#L816-L845) define 32 audio encoder blocks, pooling, normalization, and the projection into text width.
- [Source lines 1549–1569](https://github.com/huggingface/transformers/blob/e7e568dc7673ac99a87660c7c50e84830268285c/src/transformers/models/qwen2_5_omni/modeling_qwen2_5_omni.py#L1549-L1569) define the text decoder's embedding, 36 blocks, and final norm.
- [Source lines 1676–1681](https://github.com/huggingface/transformers/blob/e7e568dc7673ac99a87660c7c50e84830268285c/src/transformers/models/qwen2_5_omni/modeling_qwen2_5_omni.py#L1676-L1681) place both modules under `thinker.model`.
- [Source lines 1891–1898](https://github.com/huggingface/transformers/blob/e7e568dc7673ac99a87660c7c50e84830268285c/src/transformers/models/qwen2_5_omni/modeling_qwen2_5_omni.py#L1891-L1898) insert projected audio features into text-token placeholders with `masked_scatter`.
- [Source lines 2108–2122](https://github.com/huggingface/transformers/blob/e7e568dc7673ac99a87660c7c50e84830268285c/src/transformers/models/qwen2_5_omni/modeling_qwen2_5_omni.py#L2108-L2122) expose text logits and hidden states.
- [Processor lines 156–210](https://github.com/huggingface/transformers/blob/e7e568dc7673ac99a87660c7c50e84830268285c/src/transformers/models/qwen2_5_omni/processing_qwen2_5_omni.py#L156-L210) process audio and expand audio-token placeholders. The same source's example uses `apply_chat_template(..., tokenize=True, return_dict=True, return_tensors="pt")` for multimodal input.

The pinned config has audio width 1,280, projected width 2,048, text width 2,048, 36 text layers, and vocabulary size 151,936. The audio layer output and `audio_tower.proj` can be hooked with standard PyTorch forward hooks. Text layer outputs are post-block residuals. `adapter.py` validates the expected layer counts and widths before registering hooks.

## Scaffold use

`adapter.py` provides:

- `load_pinned_thinker(allow_download=False)`: optional later load of Thinker and processor; by default it reads only an existing local cache.
- `prepare_audio_batch(processor, audio_path, system_prompt=...)`: preserves the local audio path and SHA-256, and delegates actual preprocessing to the pinned processor.
- `resolve_hook_points(model, audio_layers=..., text_layers=...)`: names audio encoder layers, the audio-to-language projection, and text decoder residual layers.
- `ActivationCapture(points, detach=True)`: records hook outputs without changing the forward result. Set `detach=False` only when a subsequent gradient calculation needs the graph.

Run the no-weights checks with `/opt/homebrew/bin/python3.12 -m unittest -v test_adapter.py` from this directory. They use fake modules and a fake processor, verify exact config pins and module paths, and check that hooks are removed after use. These are **interface checks**, not evidence that a loaded model runs or that a lens is accurate.

## Next gate

Follow [BENCHMARK_PLAN.md](BENCHMARK_PLAN.md) before fitting a corpus lens. The evaluation and evidence rules are in [eval/PROTOCOL.md](../../eval/PROTOCOL.md). The upstream [Anthropic J-lens adapter](https://github.com/anthropics/jacobian-lens/blob/main/jlens/hf.py) expects a text decoder; its `from_hf` layout can be given `Layout(path="model.language_model")` for a loaded Thinker, but that must be verified on the pinned checkpoint. Its stock `fit()` accepts text prompts only. An audio-conditioned fit needs a separate multimodal input and position-masking implementation.

## Loaded integration gate: 2026-11-17

The [loaded smoke script](smoke_loaded.py) ran locally on an Apple Silicon Mac with 48 GiB unified memory and MPS; **no Brev instance was created**. The isolated environment is frozen in [requirements.freeze.txt](requirements.freeze.txt). The processor required Pillow and torchvision even for this audio-only batch because its Omni processor also constructs image and video components. The [full measurement record](loaded_smoke_verified.json) used the pinned 3B Thinker checkpoint (4,703,464,448 loaded parameters, all 1,442 parameter tensors on `mps:0`) in BF16.

Input was [sample_apple_5p5s.wav](sample_apple_5p5s.wav), a 5.5-second trim of the previously generated Apple Samantha calendar clip. Source SHA-256: `6a2db368f25bfc2bc39025059e75843e00d5328159a8b00c3b6bb06e9821344a`; trimmed SHA-256: `ba226916abdd36bef212a0b478d49eb63028fb541d84a7934f50e4871fe6d21d`. The pinned processor produced 158 text/audio placeholder positions and 137 projected audio positions. Hooks captured encoder layer outputs of `275×1280`, a projection output of `137×2048`, and text residuals of `1×158×2048`.

The complete logits were **exactly equal** with and without detached hooks (`max_abs_logit_delta: 0.0`). A selected final-position token logit had nonzero gradient norms to audio encoder layer 0 (`10.1003`), the audio projection (`0.8390`), and text layer 17 (`0.8876`). With explicit MPS synchronization, the cached-weight run measured 0.261 seconds for an ordinary forward, 0.128 seconds for the gradient-tracked forward, and 0.153 seconds for the single-logit backward probe; MPS driver allocation after backward was about 12.0 GB. These timings are **one-run local measurements**, not an estimate of the full J-lens cost. The first weight download took about 290 seconds and occupies roughly 11 GB in the ignored local cache.

The next gate after the loaded smoke test was a [bounded 16-coordinate Jacobian cost benchmark](JACOBIAN_BENCHMARK.md), now complete. The loaded smoke test shows access to the needed graph and modules; it does not establish that a J-lens produces useful or causal audio readouts.

The bounded [16-coordinate Jacobian pilot](JACOBIAN_BENCHMARK.md) has since completed locally on MPS. A full-width fit and held-out readout/causal validation remain open.

The separate [terminal-position fitter](TERMINAL_FIT_PLAN.md) is implemented with retained-graph coordinate batches, atomic per-batch checkpoints, and CPU analytic/resume tests. Its full-width MPS run has **not** started; its readout remains unvalidated.
