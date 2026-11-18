# Independent review: blinded Qwen v2 listening interface

**Verdict (2026-11-18): source review passed for local collection; no real listening has occurred.** I reviewed commit `92f32f8` without modifying the server, queue, browser files, or response data. The reviewed SHA-256 values are:

| Artifact | SHA-256 |
| --- | --- |
| `listening_server_v1.py` | `b20479096cd340d1c5a952b2b88ae6f42706b7caf73b0827367f39ac0adaff42` |
| `listening_v1.html` | `3c912bf01cac53f1c6b019e441a54a98294addbd2dcd814f4c7e27e675347cc3` |
| `listening_v1.js` | `9053555674e5a6eac6f0160fc5a85900dd38e9ba4a3dd8a8abf4a0d479575962` |
| `LISTENING_INTERFACE_V1.md` | `f9cc97c43ce9578ff749ec8db2516b82551ffd331aa2250e38ba9c64a5c23ff4` |

## Checks

- `python3 listening_server_v1.py --validate-only` passed and bound all **480** opaque IDs to the frozen private map and WAV hashes, without producing a response. `python3 -m unittest -v test_listening_server_v1.py` passed **5/5** synthetic tests; `node --check listening_v1.js` passed.
- The server binds only `127.0.0.1`. The only served resources are the HTML, JavaScript, opaque queue, progress by opaque ID, audio bytes at `/audio/{opaque_id}`, and a generic response receipt. There is no static-file route. A changed WAV is rejected by its SHA-256 before being served. The HTML, JavaScript, JSON queue, URL, and response headers contain no intended word, source filename, pair, split, or private-map path. The HTTP test also rejects a cross-origin request and direct private-map/clip-path requests.
- Initial responses are unique per opaque ID and review stage. The server prevents one declared reviewer ID from supplying two independent initial reviews for the same clip, requires primary before secondary and secondary before tertiary, and writes numbered, hash-chained JSONL records with `fsync`. Corrections retain the initial row and require a same-reviewer reason; an adjudication requires prior reviews and a distinct declared reviewer. Restart verifies the complete chain and source hashes. A filesystem lock prevents two collectors from owning one run.
- The collector records observations only. It has no ASR, reference-word comparison, quality gate, or Qwen inference path, and the synthetic tests create no real-corpus listener answers.

## Conditions for interpreting later responses

The service is local and has no authentication of a person's identity. Reviewer IDs and browser play counts are self-reported; a saved row is not independent proof that a person heard or understood the clip. Blinding requires listeners who lack repository/private-map access, since the opaque IDs are pseudonyms rather than secrets. The server permits second and third review collection, but the **later independent QA auditor must check the protocol's actual ASR/first-listener flags, reviewer separation, and required coverage** before any quality verdict. Real human submissions and the two pinned ASR passes remain outstanding. This review does not authorize Qwen v2 model inference or claim 480 intelligible clips.
