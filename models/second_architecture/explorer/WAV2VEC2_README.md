# Wav2Vec2 FIT evidence explorer

Open [wav2vec2_fit_v1.html](wav2vec2_fit_v1.html) in a browser. It is an offline, read-only page. Select one paired synthetic recording to play its original WAVs, compare stock versus shared-stat audio-state difference traces at five pinned model boundaries, inspect the complete CTC transcript for each of seven original patch arms, and inspect the separately audited four-state word/outside factorial at either hook site. The word-window overlay uses the fixed PCM receptive field, not a learned alignment. The trace uses max-pooling into display bins so spikes are retained; exact first-nonzero and outside-window values are shown below it.

The builder verifies pinned baseline, word-window and complement JSONL/audit hashes, both independent result reviews, all 320 baseline, 448 word-window and 384 complement sidecar hashes, exact source WAV hashes, and the expected 16 baseline/224 word-window/192 complement patch matrix. It will refuse missing or changed evidence. The generated HTML embeds only verified compact records and time traces; source WAV playback uses paths relative to the repository. Copy the repository with the page to keep audio playback available. The browser makes no network requests.

Rebuild and check:

```sh
reproductions/audio_prior/.venv/bin/python models/second_architecture/explorer/build_wav2vec2_fit_v1.py
reproductions/audio_prior/.venv/bin/python -m unittest discover -s models/second_architecture/explorer -p 'test_build_wav2vec2_fit_v1.py' -v
node --check models/second_architecture/explorer/wav2vec2_fit_v1.js
```

This viewer shows already-open FIT speech synthesized with Piper and Kokoro in one English CTC ASR model. Mode C's shared outside-word normalization is a constructed counterfactual. The state patches are internal edits, not waveform attacks. The factorial is the same words, voices and sites as the original patch assay, not an independent replication. The word and outside edits have different natural magnitudes; outside-only failure shows lack of sufficiency under this edit, not absence of information there. The page makes no claim about human speech, speech assistants, tool calls, or a security vulnerability. Its primary intended-word count includes only directions with correct clean transcripts at both endpoints; all misheard and random-control outputs remain visible.
