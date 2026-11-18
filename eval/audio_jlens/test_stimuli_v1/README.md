# Frozen audio-Jacobian test stimuli

The [renderer](render.py) consumed the unchanged [split](../SPLIT_V1.json) and the pinned Piper Amy/Kokoro Bella settings in [heldout_flow/plan_v1.json](../../heldout_flow/plan_v1.json). It wrote the [manifest](manifest.json), 16 clips, and a [split-to-manifest binding](BINDING.json) **before any audio-Jacobian candidate evaluation**. The renderer loads CPU TTS models only; no Qwen, ASR, response scoring, or answer-based selection ran.

- Split SHA-256: `2b48246361a629fbb0fa80c79a3d061b6977c8cb8af71cdc2275b1e4b0b4d45a`
- Manifest SHA-256: `88bb12c02bc000ffee7dee8329ac3aeddd57d5b44898786f8e6c88d63454f845`
- Binding SHA-256: `da9e8fa761dfe70b84b6f5a8f0d2946e19dfb7acc3298a30fb2e9fa063753113`

There are two unseen contrasts (`north/south`, `seven/eight`) × two further phrasings × two TTS families = eight pairs. Each word is rendered separately and padded to the longer word slot within its pair. Prefix and suffix PCM are reused exactly. An independent post-render check reopened every final WAV as 16 kHz int16 PCM, verified every saved file hash, checked identical length, and confirmed sample-for-sample equality outside the word slot and a nonzero difference inside it. The manifest records all fragment and clip hashes, slot samples, versions, prompts and source plan hashes.

To recreate separately, copy `render.py` to a new empty directory under the same parent and run it with `eval/tts/.venv/bin/python`; it refuses to overwrite the existing manifest or generated folders. Byte-identical rerendering is not assumed across ONNX platforms. This small synthetic set tests lexical and wording transfer only. The split voices are new relative to the red/blue fit pair but were used in the development set. Splice timing is a constructed waveform boundary, not an independently aligned phoneme boundary.
