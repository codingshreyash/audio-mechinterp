# Independent review: word-disjoint audio state validation V2

19 November 2026. Reviewed the frozen V2 plan at `1c86c5c` and the blinded-listening clarification at `7485626` against the V1 HOLD review. I checked the three cited corpus specification/manifest SHA-256 values and used metadata only; I did not open locked-test audio or answers, render clips, or run candidate-model inference.

**Decision: GO for source implementation and preflight preparation. Candidate inference remains gated on independent runner/auditor review and a frozen, passing no-weight preflight; Track B also requires its audited 32/32 human listening receipt.** This review closes the two V1 plan-level HOLD items. It is not an inference or listening GO.

## V1 blocker 1: exact transcript and listening rule

V2 now specifies Python NFKC normalization, `casefold()`, and `re.sub(r"[\W_]+", " ", ...).split()` for human and ASR transcripts. Strict clean-ASR eligibility compares both complete token lists to the frozen template with the word at a fixed position: Track B is exactly `i said {word} and then stopped`; Track A derives its list from the pinned `dev_listen` prefix/suffix. There is no outcome-dependent alignment or word search. The plan retains every planned case and raw output when a baseline is ineligible. I independently exercised three accepted parser examples (ordinary punctuation, Unicode compatibility/case, underscores) and six rejected examples (repeated target, paired alternative, missing template word, empty output, extra digit, altered word). The implementation still has to include its own edge-case tests before model output is inspected.

The `7485626` clarification closes an adjudication ambiguity: two independently blinded matching reviewers can be the two original listeners or one original listener and the triggered third. A third listener is blind to the earlier answers; one passing review alone cannot pass a clip. All first and adjudicating records remain saved.

## V1 blocker 2: output budget

V2 prohibits saving every patched full-state tensor, which made the previous Wav2Vec2 storage pattern exceed 512 MiB when scaled to 768 arms. It lists retained clean inputs, clean site states, Wav2Vec2 clean/arm CTC logits, Whisper complete generated IDs/strings and bounded scalar scores. Each patch records an independently reconstructible tensor hash, indices, state hashes, seed, and norms. The no-weight preflight must enumerate the full case matrix, derive shape bounds from pinned configs and WAV frame counts, include 256 bytes per `.npy` file, reserve 32 MiB for metadata and 64 MiB contingency, and reject a total above 512 MiB before a candidate forward. The saved-data auditor must reconstruct every patch and re-decode every full output. The concrete preflight must make the size estimate and random-generator recipe executable; this review does not assert that the not-yet-rendered Track B files fit.

The V1 review already checked the fixed 12-pair Track A selection, 576-arm arithmetic, 16-word disjoint Track B list, 32-WAV/two-engine fixture, and Track B's 768 Wav2Vec2 and 384 Whisper patch counts. V2 preserves them and the explicit exploratory label on the previously inspected v1 development split. Neither plan nor this review establishes a security finding.
