# Independent UI rereview: descriptive viewer

19 November 2026. **GO for viewer release.** I rereviewed fix `c844ec8` against
the UI HOLD in `INDEPENDENT_DESCRIPTIVE_VIEWER_REVIEW_V1.md`. This was an
offline, saved-evidence check; no model inference or raw-evidence edit occurred.

The generated HTML passes the pinned builder's `--check`, all three builder
unit tests, and `node --check` on its extracted inline script. The added
`test_viewer_browser_v1.sh` passes. The embedded evidence JSON is byte-for-byte
unchanged from the held viewer at `5b78da0`: SHA-256
`d32c8e533e1876c2ae25efdf919a5a033c88d5d28d17fb12c81735b786e8dbf2`.
The prior independent audit of the 20 WAVs, displayed V2 fields, 42-record
inventories, and 4/0/3 matched-silence differences therefore still applies.

In a separate real-browser session, I opened the checked-in file directly and
observed 20 initial rows with the detail and selected row both at
`d02_beep_train_high`. Setting Duration to 8 seconds immediately moved both to
the visible `d08_band_noise_high` case. Setting Sound to Beep train and enabling
the difference-only filter left exactly the two `d08_beep_train` rows, with
`d08_beep_train_high` selected and shown in detail. Clicking the low case moved
both the selection and detail to `d08_beep_train_low`; its two detail audio
elements retained embedded WAV data URLs.

With the difference-only filter still active, selecting Duration 2 seconds and
Sound Band noise produced one explicit no-match table row, a no-match detail
message, no case buttons, and no stale detail audio. Switching Sound to All
recovered two visible `d02_beep_train` rows with exactly one selected case and
matching detail. The original hidden-selection defect is resolved.

This GO concerns the evidence viewer's presentation and navigation. It does
not upgrade the underlying descriptive observations to a causal or security
finding.
