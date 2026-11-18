# Voxtral typed-control prompt v2 (fit-only)

The first Voxtral smoke used the v1 typed prompt `I said the color {red|blue}. Which color did I say? Answer with one color word.` The v1 Blue response was `**Blue**` with EOS. It therefore failed the preregistered exact-one-word gate even though the color was intelligible. The v1 response and gate outcome remain unchanged.

This **fit-only protocol repair** adds one fixed instruction to the same typed sentence:

> I said the color {red|blue}. Which color did I say? Answer with one color word. Reply with exactly Red or Blue. Do not use Markdown or punctuation.

Both colors use that template. The audio prompt and model generation settings are unchanged. Invoke `voxtral_smoke.py --typed --word red --prompt-version v2` (or `blue`). The script records the exact text, its SHA-256, `prompt_version=v2`, script SHA-256, raw generated IDs, complete answer, and EOS. New v2 results go under `runs/voxtral_typed_prompt_v2_fit/`; v1 artifacts are never rewritten.

The v2 gate uses the existing `exact_one_color_word` scorer (case insensitive; it permits a final `.` or `!`) plus EOS. The raw answer also shows whether the no-punctuation instruction was followed. A v2 success is evidence about the repaired **fit-only** control; it does not retroactively convert the v1 Blue result into a pass or validate behavior on new wording, voices, or models. Freeze this prompt before a separate development or held-out test.
