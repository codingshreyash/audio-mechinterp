# Controlled spoken-word intervention pilot, version 1

This fit-only assay asks whether a known change in speech can be localized in Qwen2.5-Omni-3B's internal audio path and causally change its **complete** answer. It is a mechanism check before making a security claim.

## Fixed first pair

- Spoken question: “I said the color red. Which color did I say?” versus the identical question with “blue.”
- The raw prefix, word and suffix are rendered separately with the pinned Piper VITS and Kokoro TTS models in [`../tts/pins.json`](../tts/pins.json). Within each engine, prefix and suffix PCM are reused byte for byte. The color-word slot is padded to the same sample count in both versions; the final 16 kHz WAVs must have equal length and identical PCM outside that slot. The exact changed interval and every source/output SHA-256 are saved in a manifest.
- The Thinker receives the audio as one user message and the fixed system text `Answer the spoken question with one color word.` Greedy generation stops at the tokenizer's end-of-turn ID. Record full decoded answers and whether generation actually stopped.
- A positive behavioral control requires both engines to answer “red” to red and “blue” to blue across two identical inference runs. An independent ASR transcription is a quality check, not proof of human perception. If the behavior control fails, this pair is not used to support a causal interpretation.

## Causal test design, after the behavior control

Capture post-block encoder states and the connector states with the pinned extractor. Map the known word-slot sample interval to nominal model rows; encoder self-attention mixes each 2-second chunk, so a row is **not** asserted to be a pure word location. Patch matched red/blue states at a prespecified layer and row window, then score the complete answer and prespecified red-versus-blue first-token margin. Repeat the reciprocal patch.

Controls: unedited, self-patch, wrong-time window of equal width, wrong layer, norm-matched random patch and full-region swap. Check whether the edit changes unrelated output or causes incoherent generation. A changed margin without a changed complete answer is an intermediate observation. A whole-state swap alone does not identify a specific direction. Keep this pilot in the fit split; develop a separate locked set of voices, word pairs and phrasings before a general mechanism claim.

The first pair is intentionally simple. Its purpose is to separate an audio pathway effect from task ambiguity and give later semantic, prosody and security evaluations an independently controlled positive case.
