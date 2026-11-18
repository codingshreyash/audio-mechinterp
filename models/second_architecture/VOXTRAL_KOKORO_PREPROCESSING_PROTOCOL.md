# Voxtral Kokoro Sarah fit-voice replication v1

**Purpose.** Replicate the [committed Piper preprocessing decomposition](VOXTRAL_PREPROCESSING_DECOMPOSITION_PROTOCOL.md) on the same red/blue lexical pair spoken by a second pinned TTS voice, Kokoro `af_sarah`. This is a **fit-voice replication**, with no development or held-out audio. It tests whether the feature-splice pattern survives a change in voice and rendering; the Kokoro clip is also longer, so any difference cannot be attributed to voice identity alone. Hybrid mel tensors are off-manifold. No security finding is claimed.

## Frozen inputs, code, and hashes

The [controlled-pair manifest](../../eval/tts/outputs/controlled_pair_v1/controlled_pair_manifest.json) SHA-256 is `318807c92971babacad93b69b7d8478e29f397219da0cb9ed6e3134649aea98b`. Kokoro `af_sarah` is pinned in [TTS pins](../../eval/tts/pins.json), SHA-256 `a934a5c1a8391d463a0a305e0932a39630049963838832e30a3e5b18f5f73612`. The 16 kHz red/blue WAVs are each **101,840 samples (6.365 s)** and sample-identical outside word samples `[35920, 61120)`. WAV SHA-256: red `b3f09333295ede3cc909f3e1b6f1d84493f3f0d730d7bde8b3ed068ec267cfdb`; blue `fd7864c3bc14c210fc22f49888ee6274e7a0c64e4283d12303d915db2aeeb962`.

The standalone runner is [voxtral_kokoro_preprocessing_decomposition.py](voxtral_kokoro_preprocessing_decomposition.py), SHA-256 **`acaf7a81b0d1dddd4329f258609dd1945286720c608dd913abd2e4bddddf1856`**. It imports the committed Piper decomposition source, SHA-256 **`026445da1ac2c4c2f4b8f45ab2f8d7a1f4679924774714951b16d5cd29de0788`**, without modifying it. Both use the same locally pinned `mistralai/Voxtral-Mini-3B-2507@3060fe34b35ba5d44202ce9ff3c097642914f8f3`, Transformers v5.20.2 installed source hashes, and [frozen dependencies](requirements.freeze.txt). The runner checks source, snapshot revision metadata, WAV hashes, six feature hashes, and outside-word PCM equality before inference.

## Preprocessing and factorization

The CPU-only preflight independently reconstructs the pinned Whisper 128×3,000 log-mel tensors from PCM, without calling its private feature-extraction function. Both reproduce the processor tensor with maximum absolute error **0.0**. Raw log-mel maxima are red **1.4036763906478882** and blue **1.5550248622894287**. Raw off-word log-mel frames match exactly; using the pair's shared maximum for the floor/normalization makes all off-word and padded frames match exactly. Thus clip-wide normalization creates the original off-word feature difference before Voxtral's encoder.

For Kokoro the conservative direct acoustic word window is mel frames **224–383** (160 frames); nominal feature padding begins at frame **637**. The matching pre-attention projector window is rows **27–48** of 375; nominal projected padding begins at row **80**. The bidirectional encoder can spread information beyond these indices. The six inputs use the same 390 prompt token IDs and differ only in `input_features`:

| Condition | Word frames 224–383 | Other real frames | Frames 637–2999 | Float32 feature SHA-256 |
| --- | --- | --- | --- | --- |
| `word_red_outside_red` | red | red | red | `0f7178b63810e1838845a7a38e60c7c7c9030a5a500114590a7b687eea9c5c47` |
| `word_blue_outside_blue` | blue | blue | blue | `c766d366a5260558bc34ad872f14c781570263c96e34aa268ba4b55c904f10a0` |
| `word_red_outside_blue` | red | blue | blue | `fb528d66bd0346cc011a9a538e5309ddf98856d0ee5b65bf42f4f1c2f2be640a` |
| `word_blue_outside_red` | blue | red | red | `bb8ae02fde06297e0b0b2367b2488a8df76e8b217bda6eb688478ed9fd85b524` |
| `red_with_blue_padding` | red | red | blue | `9328a3f16d84b68a210e5bb96b3650ad99d30f5d27ff86bddc23c2aabba36f83` |
| `blue_with_red_padding` | blue | blue | red | `f6495044318f726b0b8ba2ab22faa5eebbdc2a0be1fdaa25dddb2d378187ab4b` |

The first two conditions are exact native processor outputs; the rest are off-manifold feature interventions. The original audio chat prompt, greedy generation, and candidate tokens are unchanged from Piper.

## Frozen readouts and gate

Run the six conditions in the listed order, batch size 1, BF16 MPS, with the inherited **30 GiB** process cap. For each: record complete generated suffix IDs, decoded answer and EOS; teacher-force complete `Red + EOS` and `Blue + EOS` replies and report `LL(Blue + EOS) − LL(Red + EOS)`; capture post-projector `[375,3072]` vectors and all 375 per-row L2 distances to both native endpoints, summarized over word rows 27–48, other real-audio rows, and nominal padded rows 80–374. As in the Piper script, stop interpretation of hybrids if either native answer fails the exact color-word plus EOS gate. A changed teacher-forced margin without a changed complete answer is an intermediate result.

The script defaults to **CPU-only preflight**. Only `--run` loads model weights and uses MPS. New JSON evidence goes to `runs/voxtral_kokoro_preprocessing_fit_v1/` without overwrite; existing Piper scripts, protocol, and results remain untouched. The run is bounded to six generations (`max_new_tokens=32` each) and 12 teacher-forced passes. No dev/test examples are used.
