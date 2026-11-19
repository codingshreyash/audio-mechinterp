# Independent executable review: Whisper Tiny causal tail pilot

19 November 2026. **GO for one bounded attempt per checkpoint, subject to the committed model-specific GO receipts and the live host gate.** This is an executable-source decision, not an inference result or a security finding. I did not load either checkpoint or run candidate inference in this review.

## Exact frozen inputs

- Source freeze: `db1951f` (after `4a8816b`, `37799e3`, and the independent host fault tests). Source-only admissions: `e18ba28a1838d95b46495d4075c473c8e0f89513`.
- Frozen plan SHA-256: `0b9117123e407ba2045ac535d1edacbb3fbfb8eb1bc47c522c3272320fa3c4eb`.
- Multilingual admission SHA-256: `6184868b7dd911b296969bd2535c0773e63f9c1b44e24487b96e54a123c50fa7`.
- English-only admission SHA-256: `d1eccb91685137b0445fc53a1c4c7461faae5b86fc07b9a2d07bf20361b1987b`.
- I independently recomputed `admission(model)` and compared it with each committed JSON receipt. Both match exactly. The prospective attempt directories are absent. The plan reconstructs from the passed V4 record/audit/inventory hashes and fixes the same 8-second non-speech WAV plus equal-duration zero/noise tails for both checkpoints.

## Executable checks

The schedule is 56 fresh `generate()` calls per checkpoint: two no-hook baselines, 48 local patches, and six full-donor controls. Three fixed encoder sites, two fixed time regions, both patch directions, and four local arms are recorded. The convolution receptive field for encoder row `r` is mel frames `2r-2..2r+2`; the boundary windows are wider by design. The pre-block hook edits its input Tensor; the other two hooks edit module output. The saved auditor rehashes all 394 expected sidecars per checkpoint, reconstructs each edited Tensor from saved donor/recipient states, checks unselected rows, and marks a target transfer interpretable only when its sham, source-row controls, and corresponding full-donor control behave as specified.

The worker checks a live parent pipe and one-use nonce before model import, then checks the fixed source/admission/GO ancestry. The parent begins its 1,800-second clock before its first host probe; samples process-tree RSS and evidence size while sending heartbeats; reserves time for the post-run power log; and fails closed on resource, host, pipe, or incomplete-output faults. Host probes run off the heartbeat loop. Battery must be at least 80% at each start and above 70% during the run; lid and thermal state must remain readable and acceptable with at most a 15-second sample gap. A late pending probe is consumed before normal success. Saved audit checks host and resource logs, the output census, phase counts, generation signature, complete token/EOS paths, first-step vector hashes, and unchanged multilingual language detection. It compares fresh baselines with V4 mel, mask, raw logits, processed scores, prefix, and complete sequence.

I ran the final five no-weight test modules with the project Python environment: **21 tests passed**. They include a 56-call/394-sidecar synthetic saved audit for both checkpoints, answer-tamper rejection, fake hook patch reconstruction, direct worker rejection before weights, slow host probing without heartbeat starvation, probe failure, and terminal pending-probe timeout. I separately inspected the pinned Transformers Whisper generation path: multilingual `language=None` performs a distinct language-detection encoder pass before transcription generation; the patch wrapper edits only the latter, and each arm invokes a fresh generation call without supplied cache or encoder outputs.

## Interpretation limits

This pilot follows one contrast pair already selected from V4. A successful patch can support local causal mediation of this observed output difference at a named hook and time region; it cannot establish a general audio mechanism, transfer to other checkpoints, or a security vulnerability. English-only is a same-WAV disruption control because its V4 complete answer did not change. A raw logit shift or a censored continuation is not a complete-answer rescue. Later encoder attention can spread information outside patched rows, so a row-local effect does not prove row-local upstream computation. Saved host evidence stores sanitized battery/lid summaries plus `pmset` text, not raw `ioreg` property lists; the live parser and source-bound parent enforce the gate, but a later auditor cannot independently reparse those raw properties.

No candidate inference was performed to make this GO decision. The attempts must remain one-shot and separately audited after execution.
