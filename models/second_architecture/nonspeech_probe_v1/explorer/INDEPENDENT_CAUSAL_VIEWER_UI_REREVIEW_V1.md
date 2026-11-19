# Independent UI rereview: causal attempt 002 viewer

19 November 2026. **GO for viewer release.** I rereviewed `dfc69ea` against the donor-control UI HOLD in `INDEPENDENT_CAUSAL_VIEWER_REVIEW_V1.md`. This was a saved-evidence presentation review; no model inference or raw-evidence edit occurred.

The page now has a visible four-row silence-donor table: no-hook and read-only for both the two-second and eight-second donor. Each row shows the saved call identity, transcript, generated token IDs, complete sequence IDs, termination, EOS/censor status, and comparison flags. The recipient detail tables still show 18 cases × (10 Stage A + 8 Stage B arms) = 324 recipient calls. Together these account for all 328 saved forwards. The donor read-only complete sequences equal their same-donor no-hook sequences, both finish with EOS, and their saved comparison flags are true.

The embedded causal JSON is byte-for-byte unchanged from the initially reviewed viewer at `35a8225`: SHA-256 `017aac116e8169cbabd9b1c09df44c9ca708f93fd68421f9993df388f56595f5`. The prior independent 328-record field comparison and 20-WAV hash/decode review therefore still applies. The new template adds only local DOM rendering; it introduces no remote resource or network call.

The deterministic builder passes `--check`, all three builder unit tests pass, and the extracted inline script passes `node --check`. The real Chrome test `test_causal_viewer_browser_v1.sh` passes the donor-row census and fields, 18-case and stage tables, paired embedded audio, evidence hashes, filters, selected detail, length-censored arm, empty-filter clearing, and selection recovery. The donor controls remain visible under recipient filters.

This GO covers the offline viewer's evidence presentation. The underlying run remains `GO_SAVED_EVIDENCE_WITH_CENSORING` and does not establish a localized audio mechanism or security finding.
