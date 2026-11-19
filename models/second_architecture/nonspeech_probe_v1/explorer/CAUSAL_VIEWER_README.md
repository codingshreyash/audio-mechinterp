# Audited causal attempt 002 viewer

Open `causal_v1.html` directly in a browser. It embeds the frozen 20 WAVs and
compact output records for all 18 recipients and both silence donors. It needs
no network connection. The page links to the sibling independent audit memo,
summary, full JSONL record, and descriptive viewer for deeper inspection.
The four donor no-hook/read-only controls have a visible table with full token
paths, termination and saved comparison flags; together with the 324 recipient
arms, they account for all 328 saved forward calls.

Rebuild and check:

```sh
python3 models/second_architecture/nonspeech_probe_v1/explorer/build_causal_v1.py
python3 models/second_architecture/nonspeech_probe_v1/explorer/build_causal_v1.py --check
python3 -m unittest discover -s models/second_architecture/nonspeech_probe_v1/explorer -p 'test_build_causal_v1.py'
bash models/second_architecture/nonspeech_probe_v1/explorer/test_causal_viewer_browser_v1.sh
```

The builder pins SHA-256 of the immutable 328-call record, independent audit
summary and memo, original WAV manifest, and every embedded WAV. It also checks
the case and arm census, censor list, independent arm counts, and baseline donor
contrast. It only reads saved evidence; it never loads the model or writes to
evidence files. The generated HTML is a review aid, not a new security finding.

The optional browser test requires `agent-browser` and Chrome. It checks the
18-case inventory, stage tables, embedded paired audio, hash display, filters,
selection, and a censored arm.
