# Independent CPU audit of the prospective corpus render

**2026-11-18; no Qwen or ASR inference.** The selection was committed in `70ad236` before rendering. I reopened the [manifest](rendered/manifest.json), [binding](rendered/BINDING.json), and every final WAV using a separate read-only script after the renderer completed.

| Check | Result |
| --- | --- |
| Selection SHA-256 | `4c4cb6beafdd27509a073872f71dea004be686a6d8fafccbae410b8b11d7f265` |
| Rendered manifest SHA-256 | `adf8cd02d24a43ef576bcb993dc54bd2527440a62798e12fc788def3c063971f` |
| Matched pair / WAV count | 192 / 384 |
| Fit-pool pair records | 48 Piper + 48 Kokoro = 96 (24 word pairs × two templates × two engines) |
| Development pair records | 24 Piper + 24 Kokoro = 48 |
| Locked-test pair records | 24 Piper + 24 Kokoro = 48 |
| Distinct words by split | fit 48, development 24, test 24; all sets disjoint |
| Every WAV hash/PCM hash | Match saved values |
| Every pair's audio | Same 16 kHz PCM16 length, identical samples before and after word slot, unequal samples inside slot |
| Pre/post quarter-bin margin | At least 25,418.5 samples before the word slot and 25,820 samples after it |

The independent script used only `json`, `hashlib`, NumPy, and SoundFile. It did not use the renderer's PCM equality function. The integrity result proves the assembled waveform relationship and provenance, not that a person or model hears each word correctly. Independent intelligibility and ASR checks remain unrun. The corpus is pending separate metadata review before it is declared final.
