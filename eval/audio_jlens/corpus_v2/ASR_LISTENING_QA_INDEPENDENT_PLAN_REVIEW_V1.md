# Independent static review: Qwen v2 intelligibility plan

**Verdict: the metadata-only queue is internally consistent; no intelligibility gate has run.** I reviewed the committed prospective protocol and queue generator without opening any WAV, loading an ASR model, asking a listener, or inspecting candidate-model output. This review does not authorize Qwen inference.

## Sources and checks

The reviewed protocol is SHA-256 `30285a6c897e742eff59f593b3d12a1bd92d62585cd72171837c3e18fabe662e`; generator `a1b966afcab16785e0541b849ad4618b031c5786f89e0ddb58ff1d489e8ea3e8`; internal case map `d9a08a61b76e8ab56ab3a9b2d72a4aedc5871555487c7a33b712097fe3206ed8`; blind queue `19319ac394a24f8d742285d4bb3ecfd78020b1d131fa57def5b3362fd0b73873`. They bind to the rendered manifest `15395c7d132eb3f1c1a6b2b7d555146faa83be4ee2b77fba4f481cf70f26761d` and technical binding `858f8f68fce65ee60e328b1536b34608ced6b3e84d725f229460c6376e14af93`.

I reran the two CPU metadata tests and `--validate-only`; both passed. An independent read of the saved JSON confirmed 480 unique IDs, paths and WAV hashes; split counts 288 FIT / 96 development / 96 locked test; 60 distinct pairs with eight clips each; exact equality of queue IDs and private case IDs; and no adjacent clips from the same pair. The public queue contains only 20-character hexadecimal IDs, order, and provenance hashes. It contains no reference word, alternative, filename, pair, voice, or split. The generator validates the 240-record/480-WAV technical binding and keeps all 12 reserve pairs inactive.

The plan correctly saves raw ASR transcripts before reference comparison, fixes post-decoding normalization, preserves homophone and other lexical errors rather than mapping them to the expected word, and makes ASR agreement only a flagging aid. Its 480/480 human quality rule retains every failure and does not permit posthoc pair replacement. The proposed Whisper encoder–decoder and Wav2Vec2 CTC models are distinct ASR architectures according to their [Whisper model card](https://huggingface.co/openai/whisper-tiny.en) and [Wav2Vec2 model card](https://huggingface.co/facebook/wav2vec2-base-960h); the second checkpoint is still only proposed, not pinned or run.

## Execution boundary

The package is a **prospective queue**, not a completed QA system. Before any Qwen v2 answer or activation inference, the project still needs exact-source ASR runners and a pinned second checkpoint, a listening interface that serves audio by opaque ID without exposing filenames or the internal case map, immutable blinded responses from actual human listeners, and a separate auditor that recomputes the 480-case gate. Those sources and their hashes must be frozen before their outcomes. No automated agent may label its own metadata inspection as human listening.

The case map and manifest are committed in the repository. The opaque IDs are derived from known WAV hashes, so they are **pseudonyms, not a secrecy barrier** for someone with access to those files. Blinding is credible only if listeners receive an isolated audio interface and are not given the repository or private map. In the current workspace there are no human responses or authorized external reviewer contacts; the static work can proceed autonomously, but the human part cannot be fabricated. This is an execution dependency, not an observed corpus failure.
