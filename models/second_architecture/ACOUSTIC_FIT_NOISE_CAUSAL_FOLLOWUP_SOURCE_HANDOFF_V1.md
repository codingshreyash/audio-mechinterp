# FIT word-noise causal follow-up: source-only handoff

**No new candidate forward or causal finding.** The [prospective protocol](ACOUSTIC_FIT_NOISE_CAUSAL_FOLLOWUP_PROSPECTIVE_V1.md) selects the one already observed Wav2Vec2 Kokoro/teal case where fixed word-slot noise changed the complete CTC transcript from TIL to TEAL. Selection is post hoc from FIT. The [source-only module](acoustic_fit_noise_causal_followup_v1.py) verifies the preserved failed V1 attempt and its read-only audit, plus the exact successful V2 parent result, worker JSONL, saved audit, source/no-weight receipt, selected PCM/state hashes, and word-only raw PCM change. It constructs the fixed one-site feature-projection word/complement/full/self/random patches, returns the realized edit-tensor SHA-256 with each arm, and contains a private fake-testable hook. Its public run command always raises HOLD.

Before a live attempt, independently review the protocol/source/tests and committed no-weight preflight. A separately reviewed worker must load the pinned Wav2Vec2 checkpoint only after that gate, capture clean and noisy baseline states/logits, compare exact saved state hashes and full CTC frame paths/transcripts, then run both directions and all eight arms without selecting new inputs. A parent must impose the protocol's wall, memory, disk and host limits, preserve raw logits/states and incomplete attempts, and provide a read-only saved auditor. Machine-readable worker and parent GO receipts must be in separate later commits. The existing V2 acoustic parent cannot run this intervention because it hard-pins the V2 descriptive worker; it must not be repurposed.

No claim about a user-supplied audio attack follows from an internal patch. Passing controls would establish only a causal boundary for this one selected, synthetic English FIT case. The separate held-out, human-intelligible, unmodified-assistant security study remains necessary.

Source-only verification:

~~~sh
models/second_architecture/.venv/bin/python -m unittest models/second_architecture/test_acoustic_fit_noise_causal_followup_v1.py -v
models/second_architecture/.venv/bin/python models/second_architecture/acoustic_fit_noise_causal_followup_v1.py preflight
~~~

The preflight command is valid only after the source/protocol have been committed. It reads prior evidence and cached model metadata, without model weights or a forward.
