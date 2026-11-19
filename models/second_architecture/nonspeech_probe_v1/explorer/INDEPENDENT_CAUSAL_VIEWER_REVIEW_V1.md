# Independent review: causal attempt 002 viewer

19 November 2026. **HOLD for viewer release pending a visible donor-control census.** The saved-evidence mapping, labels, filters, and audio work. This review did not run model inference or edit the underlying records.

## Verified

- The deterministic builder passes `--check`; its three unit tests pass; the inline JavaScript passes `node --check`; and `test_causal_viewer_browser_v1.sh` passes in Chrome through `agent-browser`.
- I separately parsed the checked-in HTML and the pinned JSONL, compared every displayed output field, token ID array, complete sequence, EOS/censor status, first raw-logit hash, language fields, parity object, call name, and call index for all **328** saved forwards. All matched. The 18 recipients have exactly 10 Stage A and 8 Stage B arms each (324 rows); the two silence donors have no-hook and read-only calls (four more rows).
- All 20 embedded WAV byte strings hash to the frozen manifest. In a separate browser session, the selected eight-second beep recipient and silence donor both decoded as eight-second audio. The HTML contains no remote script, stylesheet, media URL, fetch, XHR, or WebSocket request.
- The four informative beep cases are exactly `d02_beep_train_high`, `d02_beep_train_low`, `d08_beep_train_high`, and `d08_beep_train_low`. Their native/donor transcripts are ` Fuck`→` you` once and ` You`→` you` three times. All five named 128-token censored calls are retained. Among the eight informative random arms per stage, exact donor overlap is 1/8 in Stage A and 0/8 in Stage B. The on-page explanation correctly distinguishes the saved parity comparison flags from the independent audit's zero parity failures, and clearly says this is not a security finding.
- Browser filtering, row selection, case detail, censor display, empty-filter clearing, and selection recovery worked. No stale hidden case remained selected.

## Release gap

The page headline reports **328 saved forward calls** and says it exposes every saved intervention arm. Its visible per-case tables render only the 324 recipient arms. The four donor calls are in embedded JSON and linked raw evidence, but the UI does not show the donors' no-hook or read-only complete token paths, termination, and parity fields. The donor read-only control is therefore not inspectable in the same viewer. This matters because the reported exact donor transfers depend on the saved donor baseline and its read-only control.

Add a small visible donor-control table, preferably with the same transcript, generated IDs, full sequence, EOS/censor, and saved-field details as recipient rows. The alternative is to make the page scope explicit as a 324-call recipient viewer with four donor calls available only through the raw link, and avoid claiming a fully inspectable 328-call record. Rerun the deterministic check and browser test after the change. This HOLD concerns the viewer presentation only; the underlying independent result remains `GO_SAVED_EVIDENCE_WITH_CENSORING`, with no security or localized-mechanism promotion.
