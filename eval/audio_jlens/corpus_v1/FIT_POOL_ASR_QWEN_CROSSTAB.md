# Fit-only ASR and Qwen answer comparison

This joins saved results by **WAV SHA-256**, with no new inference. The Qwen result is `d0e185bcc778f2fc681a243b59417858ca91f4bb04fa3f715adaea64d2a3dd23c`; the ASR audit is `41da6ee4dd67a2f7b06ece24354fdb0ca6911f7017ec514e170f55f61d38a5e7`. Both bind rendered manifest `adf8cd02d24a43ef576bcb993dc54bd2527440a62798e12fc788def3c063971f`; the ASR script hash also matches its audit metadata. All 192 fit-pool Qwen clips join uniquely to 192 fit-pool ASR records, with matching word, path, and voice engine.

The ASR criterion is whether its normalized transcript contains the target word **exactly once**. The Qwen criterion is whether its full generated answer is the expected single word and ends on EOS.

| | Qwen correct | Qwen incorrect | Total |
| --- | ---: | ---: | ---: |
| ASR target once | 159 | 5 | 164 |
| ASR target absent | 21 | 7 | 28 |
| Total | 180 | 12 | 192 |

For the 48 primary map-fit cases, the four cells are 43, 3, 1, and 1 in the same order. The five Qwen failures despite ASR target recognition are `deer` twice, `moose`, `grove`, and `pillow`.

| Qwen case | Qwen answer | ASR target once? | ASR target rendering or omission |
| --- | --- | --- | --- |
| `animal_06/fit_chosen/kokoro/deer` | `Dear` | Yes | `deer` |
| `animal_06/fit_chosen/piper/deer` | `Dear` | Yes | `deer` |
| `animal_06/fit_remember/kokoro/deer` | `Dear` | No | `Dear` |
| `animal_06/fit_remember/kokoro/moose` | `moo` | No | `Mose` |
| `animal_06/fit_remember/piper/deer` | `Dear` | No | `Dear` |
| `animal_06/fit_remember/piper/moose` | `moo` | Yes | `Moose` |
| `color_06/fit_chosen/piper/cyan` | `sand` | No | `San` |
| `color_06/fit_remember/piper/cyan` | `San` | No | target omitted |
| `color_07/fit_chosen/piper/beige` | `The word you chose is 'beige'.` | No | `BASH` |
| `nature_10/fit_chosen/piper/grove` | `growth` | Yes | `Grove` |
| `nature_10/fit_remember/piper/grove` | `growth` | No | `Gerve` |
| `object_09/fit_remember/piper/pillow` | `palo` | Yes | `Pillow` |

The longer `beige` answer includes the expected word but fails the frozen one-word output rule. These observations compare an independent Whisper-tiny.en transcript with Qwen2.5-Omni-3B's answer on the same bytes. Agreement or disagreement does **not** locate the cause inside either model or establish what a human listener hears.
