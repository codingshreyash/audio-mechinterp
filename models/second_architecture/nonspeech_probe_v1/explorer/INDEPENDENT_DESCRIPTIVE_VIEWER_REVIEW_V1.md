# Independent review: offline non-speech descriptive viewer

19 November 2026. Reviewed source commit `1d36cdf` and its generated
`descriptive_v1.html` without loading model weights or running inference.

**Evidence fidelity: GO. Viewer release: HOLD for one filter/selection defect.**
The defect changes which saved case the detail panel displays after a filter,
but does not change the underlying evidence or the output table.

## Saved-evidence and build checks

- `build_descriptive_v1.py --check` passes: the checked-in 4,327,418-byte HTML
  is a deterministic rendering of the pinned saved files. Its SHA-256 is
  `068a6744326b6405d079feff395a9acdba9a6b784a2f8c202c781450e96dbe7c`.
  The three builder tests pass. Extracting its single inline script and running
  `node --check` passes. `node --check` on the HTML pathname itself is not a
  valid syntax test; Node rejects the `.html` extension.
- Independently parsed the JSON assignment in the generated HTML, apart from
  the builder. Its 20 case IDs exactly equal the pinned render manifest. All
  20 embedded data-URL WAVs are byte-identical to their local source WAVs,
  totaling 3,200,880 WAV bytes. WAV and decoded PCM SHA-256 values, duration,
  RMS, peak, mono PCM16 format, and 16-kHz rate match the manifest. The four
  displayed source-file SHA-256 values match the local manifest and V2 record
  files.
- For each of the three models, independently checked the exact 42-record
  inventory: one start, 22 case rows including two exact silence repeats, 18
  same-duration-silence differences, and one completion. Every displayed
  transcript, lexical flag, termination, script list, geometry, prefix and
  generated token ID, first raw/processed/selected token ID, logit digest,
  EOS/censor flag, and CTC field matches its primary V2 case record. The
  multilingual prefix represents a separately recorded detection ID 50259;
  it is not counted as a generated lexical token. All Whisper rows have the
  checkpoint-specific EOS (50257 multilingual, 50256 English) and no censor.
  CTC frame counts and displayed collapsed IDs match the saved frame paths.
- Recomputed nonempty counts of 20/20 multilingual Whisper, 20/20 English
  Whisper, and 3/20 Wav2Vec2. Recomputed nonsilence differences from each
  model's matched-duration silence as 4/18, 0/18, and 3/18. All differences
  occur in the beep family, exactly as the independent V2 result reviews say.
- The 10 provenance links resolve to local files. The document has no remote
  links, remote scripts/styles, imports, iframes, or fetch/WebSocket calls.
  Browser DOM inspection found all audio elements use embedded `data:audio/wav`
  URLs. The page consistently labels these as descriptive, small-checkpoint
  observations and explicitly avoids a circuit, human-perception, speech-to-
  speech, causal, or security finding. It correctly says that the saved CTC
  logits hash does not permit independent argmax verification.

These checks establish faithful presentation of the audited saved records;
they do not independently recompute missing model tensors or validate the
models' original inference.

## Viewer release defect

In a real browser, opening the file selects `d02_beep_train_high` by default.
Set **Duration** to **8 seconds**, **Sound** to **Beep train**, and enable
**Only rows that differ from silence**. The table correctly contains only
`d08_beep_train_high` and `d08_beep_train_low`, but the detail heading remains
`d02_beep_train_high`; no visible table row is selected. Clicking a visible
`d08` row repairs the state. This can mislead a reviewer who assumes the detail
panel follows the current table filter.

Fix `renderRows()` so that, after filtering, `selected` is moved to the first
visible case if the current case is hidden, then redraw the detail panel. If
no rows are visible, show an explicit empty result and clear the stale detail.
Add a browser-level regression check for both filter transitions and a row
click. Rerun the deterministic builder check and inspect the changed generated
HTML before lifting the release HOLD.
