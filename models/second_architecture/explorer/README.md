# Voxtral FIT result explorer

Open [voxtral_fit_v2.html](voxtral_fit_v2.html) in a browser. It is a single offline, read-only page with the 800 saved generations embedded. Choose a lexical pair, synthetic voice, recipient word, internal site, and restoration or loss arm to hear the original pinned WAV and inspect the complete output, strict classification, candidate contrast, edit rows, norms, token IDs, and provenance. Audio playback reads the local corpus path; copy the repository together with the page to keep playback available. The population panel uses the audited frozen gates. This is a data browser, not a hidden-state projection.

Rebuild from the committed run:

```sh
python3 models/second_architecture/explorer/build_voxtral_fit_v2.py
python3 -m unittest discover -s models/second_architecture/explorer -p 'test_*.py' -q
node --check models/second_architecture/explorer/app.js
```

The builder requires the independent audit to have passed, then verifies the result, metadata, model runner and mask hashes, all eight checkpoint and attempt hashes, the exact 864 raw slots, 800 generated conditions, 64 captures, zero invalid conditions, and the 16 selected directions. It refuses changed source data rather than silently refreshing the page. The generated HTML is deterministic from those audited artifacts. The browser does no network requests.

The evidence is limited to FIT speech synthesized by Piper and Kokoro and constructed mel-feature/internal-state edits in one Voxtral model. A successful hidden-state patch is not a natural waveform attack or security finding. The restoration time-preference gate fails despite 16/16 answer restoration; the loss time-preference gate passes on 15/16 valid lexical losses. See the [independent result review](../VOXTRAL_FIT_V2_CAUSAL_FULL_RESULT_INDEPENDENT_REVIEW.md) for the exact limitations.
