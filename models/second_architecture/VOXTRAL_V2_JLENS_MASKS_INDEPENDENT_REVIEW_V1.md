# Independent CPU review: prospective Voxtral v2 audio-Jacobian masks

Review scope: `voxtral_v2_jlens_masks.py` at SHA-256 `e0a2d6b5ca3f52a9a426af1fc12ffdc9b50e60b3dd06a0282f560b2a020e5339`, its prospective protocol, four synthetic contract tests, and the exact pinned v2 inputs. This review opened no model, processor, ASR, answer output, or active development-run result.

## Result

**Pass for CPU source and geometry preflight; no blocking defect found.** The review authorizes freezing the nominal mask manifest after its freeze receipt binds this builder and its pinned inputs. It does not establish model tensor shapes, acoustic intelligibility, an audio Jacobian, causal behavior, or a security finding.

## Checks performed

- `--validate-only` passed: 240 matched records, 480 WAVs rehashed, 144 FIT records, 48 development records, and 48 locked-test records. Its result states `processor_shape_verified: false` and `intelligibility_verified: false`.
- All four CPU contract tests passed. They exercise full-population binding, paired PCM mismatch rejection, FIT-role and manifest-claim tampering, and failure on unsupported nominal geometry.
- Independently enumerated the pinned assignment: 60 primary pairs split 36/12/12, with 16 FIT basis and 20 FIT map pairs. Each of four categories has 4 basis and 5 map pairs; each category has 9/3/3 FIT/development/test pairs. The rendered manifest has 240 distinct `(pair_id, engine, phrasing_id)` keys and 480 distinct WAV paths. No reserve pair is selected.
- Reviewed clip verification: every WAV is confined to the v2 clip root, has a pinned file SHA-256 and PCM SHA-256, and must be mono, 16-kHz, 16-bit uncompressed PCM. For each matched pair, both PCM byte strings must have equal length, differ inside the half-open word slot, and agree byte for byte before and after that slot.
- Independently checked all 240 produced geometries: pre and post each have full word-slot width; four word quarters use the pinned floor-divided edges; word-overlap mel frames use 160-sample hops and 200-sample half-windows; encoder row centers are `320*j`; projector row centers are `160*(8*r+3)`. Both sites have six disjoint nonempty bins (minimum observed bin size: encoder 10 rows, projector 2 rows). Equal-count padding rows are outside the waveform by at least 200 samples and disjoint from all six bins. Mel frame 2999 is excluded from word overlap.
- Reviewed the output gate: `--validate-only` does not write the manifest. `--write-manifest` requires an absent output and a separate freeze receipt with the exact builder SHA-256, exact source-hash dictionary, and reviewed status. Neither the receipt nor output existed during this review.

## Conditions before interpretation

The nominal `[1,1500,1280]` encoder and `[375,3072]` projector shapes are explicit assumptions. A processor-only check and hook-shape check must verify the true sample-to-mel-to-row alignment before model intervention or interpretation. If either differs, stop and review a new mask design; this builder has no silent shape adaptation. Independently verify intelligibility before claiming spoken-word fidelity.

Minor documentation correction: the protocol says “three CPU tests”; the current test module has four. The freeze receipt should also name or hash this independent review for provenance, although the builder currently gates on builder and source hashes rather than review-file hash.
