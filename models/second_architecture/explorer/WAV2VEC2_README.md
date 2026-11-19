# Wav2Vec2 FIT evidence explorer

Open [wav2vec2_fit_v1.html](wav2vec2_fit_v1.html) in a browser. It is an offline, read-only page. Select one paired synthetic recording to play its original WAVs, compare stock versus shared-stat audio-state difference traces at five pinned model boundaries, and inspect the complete CTC transcript for each of seven internal patch arms at either hook site. The word-window overlay uses the fixed PCM receptive field, not a learned alignment. The trace uses max-pooling into display bins so spikes are retained; exact first-nonzero and outside-window values are shown below it.

The builder verifies the pinned baseline and causal JSONL/audit hashes, the independent causal result review, all 320 baseline and 448 causal sidecar hashes, exact source WAV hashes, and the expected 16 baseline/224 patch matrix. It will refuse missing or changed evidence. The generated HTML embeds only verified compact records and time traces; source WAV playback uses paths relative to the repository. Copy the repository with the page to keep audio playback available. The browser makes no network requests.

Rebuild and check:

```sh
reproductions/audio_prior/.venv/bin/python models/second_architecture/explorer/build_wav2vec2_fit_v1.py
reproductions/audio_prior/.venv/bin/python -m unittest discover -s models/second_architecture/explorer -p 'test_build_wav2vec2_fit_v1.py' -v
node --check models/second_architecture/explorer/wav2vec2_fit_v1.js
```

This viewer shows already-open FIT speech synthesized with Piper and Kokoro in one English CTC ASR model. Mode C's shared outside-word normalization is a constructed counterfactual. The state patches are internal edits, not waveform attacks. The page makes no claim about human speech, speech assistants, tool calls, or a security vulnerability. Its primary intended-word count includes only directions with correct clean transcripts at both endpoints; all misheard and random-control outputs remain visible.
