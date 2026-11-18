# Independent external-freeze review: four-context FP32 non-beaver pilot

18 November 2026. **Decision: GO for the exact external-freeze provenance below only.**

Decision: GO for non-beaver FP32 external freeze provenance; four reserved one-shot contexts may run.

This review author did not create a campaign reservation, load model weights, or run a context. The executable admission is the pinned launch gate, not this prose alone.

## Reviewed identity and order

- External freeze: `VOXTRAL_FP32_NONBEAVER_EXTERNAL_FREEZE_V1.json`, first committed as `ed2cc6377cdb9405385bc8b81e03a410721a507a`; SHA-256 `c28ffecbd7487b92ff98bc47bd2872af30f0a8400cbe65a4caac30b179cafb4f`.
- V5 conceptual GO first committed as `19083e1ed279ed5d29153ba671c659b1c34c3351`; combined runner/gate source GO first committed as `645a90a020959cb47bee46b50647b280fa3ba3f4`; auditor source rereview GO first committed as `a142526f2f089786fd05e1264b1a7bb2ca4acb73`. The latter two commits precede the freeze, and the committed reviewed source blobs equal current bytes.
- All 31 frozen file SHA-256 values match the current files and their Git `HEAD` blobs. All 13 runner/gate reviewed keys match the blobs at the combined source-review commit; both auditor reviewed keys match the blobs at its independent review commit. The V1/V2 combined-source HOLDs and V1 auditor HOLD remain present and hash-bound; the later GO reviews do not rewrite them.
- A read-only `--check-freeze` before this review reached exactly the absent independent freeze-review file, after all prior admission checks.

## Independently recomputed inputs and resources

- Recomputed `VOXTRAL_FP32_NONBEAVER_CPU_PREFLIGHT_V5.json` from the pinned source-only processor path: four exact batches, each with `[1,396]` IDs/mask, `[1,128,3000]` features, 375 audio placeholders, and the four frozen feature hashes. The fixed basis seed is `3693039655029897049` and its SHA-256 is `10c9132454a01e034886a3e14bf9b9670e75bb3befd58300b9b45f8d9ba7751f`. Frozen word masks have 60/15 Piper and 73/18 Kokoro encoder/projector rows. The four WAVs match the bound hashes.
- Independently SHA-256 checked the pinned local checkpoint config, weight index, two weight shards, processor config, tokenizer asset, and generation config against the committed CPU preflight receipt. The reviewed source dependency hashes and installed Transformers source receipt are rechecked by the CPU preflight and the read-only launch gate.
- Re-ran the full-size synthetic serialization probe without model weights. Piper serialized to **1,302,437,429 bytes**, Kokoro to **1,318,364,789 bytes**. The committed byte-budget receipt matched exactly; four projected sidecars total **5,241,604,436 bytes**. These are synthetic size checks, not model outputs.
- Rehashed the previous independently audited factorial attempt manifest, its 13 unchanged raw files totaling 854,641 bytes, and its local content-addressed 509,990,581-byte tensor sidecar. Prior cumulative model work is exactly **368.4824181678705 seconds**. The new freeze retains this value.
- The frozen order is Piper silver, Piper teal, Kokoro silver, Kokoro teal. Each context plans 96 patch forwards, eight VJPs, two G1 clean graphs, two N0 clean references, one donor and one passive recipient capture, and two natural generations. Limits are 300 seconds per model process, 120 seconds per CPU audit, 30 GiB MPS driver memory, 2 GiB combined output per context, 8 GiB combined campaign output, and at least 12 GiB free before reservation. The four-context model-work ceiling is 4,500 cumulative seconds.
- Neither the launch-root nor worker-root campaign directory exists. There is no campaign reservation or context attempt to reuse or overwrite.

## Scope

This GO admits only the four pinned one-shot contexts through the gate. Each context still needs its own completed start, claim, finish, result, content-addressed sidecar and independently recomputed CPU audit; a failed context stays failed and in the planned denominator. This review establishes source, input, resource and prior-result provenance. It cannot attest future forwards, validate a Jacobian result, or establish an audio-specific or security finding.
