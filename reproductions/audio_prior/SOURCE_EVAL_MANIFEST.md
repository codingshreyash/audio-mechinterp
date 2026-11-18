# Audio interpretability prior: source and evaluation manifest

Status: local reproduction and source audit, 2026-11-17. All commands below run from this directory unless a `cd` is shown. No cloud GPU was allocated. Evidence labels follow [`../../eval/PROTOCOL.md`](../../eval/PROTOCOL.md).

## Pinned sources and claim status

| Source | Pinned repository commit | Published result selected | Local evidence |
|---|---|---|---|
| [Audio Jacobian Lens](https://github.com/kennethli319/audio-jacobian-lens), [Whisper Tiny pilot report](https://github.com/kennethli319/audio-jacobian-lens/blob/main/docs/experiments/WHISPER_TINY_PILOT.md) | `1784e62e4caaf801bfc8dc1208860703b5f3d074` | Early decoder Jacobian lens improves emitted-token rank over direct unembedding. Aligned encoder-to-decoder lens fails lexical recovery: 0% top-10 on each layer; aligned and remote affine scores nearly identical. | **Independent measured rerun; exact and directional partial replication.** See metrics below. This upstream pilot explicitly calls itself preliminary, not a validated scientific result. |
| [Audiolens](https://github.com/eric-tramel/audiolens) | `0e5438ebfdae136c26c2dfb943593360eb9a7641` | README reports pre-registered `no_band`: 8/9 gates passed; poetry delta AUC `-0.02016`. | **Source inspected and unit tests only. Published result not reproduced.** The canonical fit/evaluation requires gated `google/gemma-4-E2B-it`, authenticated Modal H100, and its original canonical artifact/report. The repository does not include those raw outputs. |
| [AudioSAE demo](https://github.com/audiosae/audiosae_demo), [paper](https://arxiv.org/abs/2602.05027) | `5c6708f487244a581397e612e4fce73e70f127e0` | Paper's aggregate feature/steering claims. | **Pretrained inference smoke only. Published aggregate not reproduced.** Demo README says training, feature steering, and probing utilities used in the paper are intentionally omitted. |

## Reproduced Whisper Tiny pilot

- Model: `openai/whisper-tiny.en` revision `87c7102498dcde7456f24cfd30239ca606ed9063`; artifact model fingerprint `5a77767bfe9c962d`.
- Corpus: 16 generated macOS `say` clips using voices Samantha and Moira; first 10 fit, final 6 held out. Local corpus fingerprint `2cc7fd8c5aa6fb73b0bd`; report fingerprint `46640e2cbca442f599ef`. Different audio bytes or environment prevent a bit-for-bit replication claim. The held-out run contains 50 generated token positions.
- Exact count matches: WER `5/47` words, CER `11/287` characters, decoder direct L0 median emitted-token rank `7678`, decoder J-lens L0 top-10 `14%`, J-lens L1 top-10 `18%`, and J-lens L2 top-10 `54%`.
- Directional match: decoder J-lens L0 median rank `1050.5` versus upstream `1037`; both improve markedly on direct unembedding (`7678`).
- Negative match: aligned encoder delta has **0% top-10** emitted-token recovery at all four layers and at most **2% top-100**. Aligned versus remote encoder affine MRR differs by under `0.001` at every layer. This supports the *methodological caution* in the report: a readable decoder trajectory does not validate a reusable encoder readout or establish causal audio localization.
- This is a readout evaluation, **not a causal intervention or security finding**. The fitted/evaluated alignment uses Whisper cross-attention, so the report itself notes circularity in any localization interpretation.

### Reproduction commands

```bash
cd audio-jacobian-lens
../.venv/bin/python scripts/make_macos_tts_corpus.py --output-dir artifacts/tts_corpus --count 16
PYTORCH_ENABLE_MPS_FALLBACK=1 ../.venv/bin/audio-jlens-fit artifacts/tts_corpus/manifest.jsonl \
  --output artifacts/pilot/whisper_tiny_en_10_tts_aligned.pt \
  --model openai/whisper-tiny.en --revision 87c7102498dcde7456f24cfd30239ca606ed9063 \
  --limit 10 --encoder-estimator aligned --no-resume
PYTORCH_ENABLE_MPS_FALLBACK=1 ../.venv/bin/audio-jlens-eval artifacts/tts_corpus/manifest.jsonl \
  --lens artifacts/pilot/whisper_tiny_en_10_tts_aligned.pt \
  --model openai/whisper-tiny.en --revision 87c7102498dcde7456f24cfd30239ca606ed9063 \
  --start 10 --limit 6 --output artifacts/pilot/eval_aligned_affine.json
cd ..
.venv/bin/python compare_whisper_pilot.py
```

The generated audio and lens came from the locally cloned, pinned `audio-jacobian-lens` source through the first two commands above; the third command generated the evaluation JSON locally. **No measured counts were copied from the upstream report.** The comparator hard-codes only the report's reference values and reads every observed value from the locally generated JSON. It checks ten explicit count/pattern assertions and passed all ten; see [`whisper_replication_check.json`](whisper_replication_check.json) and the fresh stdout capture `whisper_replication_check.log`. The assertions were written after the local evaluation and are an audit, not a pre-registered statistical test. Full measured metrics are in [`whisper_tiny_eval_aligned_affine.json`](whisper_tiny_eval_aligned_affine.json), SHA-256 `c1e785d5e32d16c5540fedd405d4a1f8e570a45cc135957c873c8be0efb4442b`. The fitted lens at `audio-jacobian-lens/artifacts/pilot/whisper_tiny_en_10_tts_aligned.pt` has SHA-256 `d4fe16219f1e4f36b74193cc72ed0ada94b8b768c70211d822f13c32cc6ea9da`. Fit/evaluation logs: `audio_jlens_fit_pinned.log`, `audio_jlens_eval.log`.

## Source validation and AudioSAE smoke

```bash
cd audiolens && ../.venv/bin/pytest -q
cd ../audio-jacobian-lens && ../.venv/bin/pytest -q --ignore=tests/test_showcase_examples.py
cd .. && .venv/bin/python audiosae_smoke.py
```

- Audiolens: **244 passed, 1 skipped** (`audiolens_pytest.log`). This validates repo tests, not the reported `no_band` result.
- Audio Jacobian Lens: **464 passed, 3 skipped** with `test_showcase_examples.py` excluded (`audio_jlens_pytest_without_private_artifact.log`). The unfiltered run had one failure because the test expects the absent, Git-ignored `artifacts/mlx/lfm2_5_audio_1clip_projected_rank512.pt`; the remaining **464 passed, 3 skipped** (`audio_jlens_pytest.log`).
- AudioSAE: `audiosae_smoke.py` loaded `facebook/hubert-base-ls960` revision `dba3bb02fda4248b6e082697eee756de8fe8aa8a` and `Egorgij21/Audio-SAE-hubert-base` revision `52e90df0a81bc23005e418a01cab475ca40141e7`. Layer-3 SAE checkpoint SHA-256 `d745298d155f7fab7f6998bb58aaa003cfa6eac500c05b4438a2764724b0be26`. On one 16 kHz resampled TTS clip it returned activation shape `[1,130,768]`, feature shape `[1,130,6144]`, L0 `48.66`, and fraction of variance explained `0.853`. These are **single-clip smoke outputs, not paper benchmarks**; see `audiosae_smoke.json`.

## Environment and limitations

Apple arm64, 48 GiB unified RAM, MPS; isolated `.venv` created with `/opt/homebrew/bin/python3.12` (3.12.13). Relevant versions: torch `2.15.0`, transformers `5.12.1`, huggingface-hub `1.33.0`, numpy `2.5.5`, scipy `1.18.1`, soundfile `0.14.0`, librosa `1.0.1`, pytest `9.2.0`. First fit under transformers `5.23.0` failed on a generation cache API mismatch; a retry after pinning `5.12.1` required `--no-resume` because the earlier partial artifact was incompatible. The successful fit is `audio_jlens_fit_pinned.log`; failed attempt is preserved in `audio_jlens_fit.log`.

The pilot's synthetic speech, tiny ASR model, and 50 token positions provide a good plumbing and negative-control gate. They do not establish that the same pattern holds for natural speech, multilingual requests, or speech-to-speech assistants. Further work should keep the prior pilot frozen and use independent labels, held-out speakers/audio, non-speech controls, and full-output interventions before making causal or security claims.
