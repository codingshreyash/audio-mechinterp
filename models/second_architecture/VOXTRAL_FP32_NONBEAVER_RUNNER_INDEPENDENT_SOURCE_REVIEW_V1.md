# Independent source review: FP32 non-beaver V5 runner only

18 November 2026. I reviewed the five source-only files committed in
`36064b8` against the immutable V2–V5 protocol chain and its V5 conceptual
review. This review does **not** cover the evolving launcher or CPU auditor.

**Decision: GO for the V5 runner, CPU preflight, and byte-budget source only. Live model work remains HOLD until independent launcher and auditor source reviews, an external freeze, and its later review.**

| Reviewed file | SHA-256 |
|---|---|
| `voxtral_fp32_nonbeaver_linearity_runner_v1.py` | `91e622f7fd8f4d9dac00ca7a0baaf02d74faf9326109d10674ebf5673310bc04` |
| `test_voxtral_fp32_nonbeaver_linearity_runner_v1.py` | `a4d9600fcd0371f1c0cf11a6ea6133d9f5ee7d097d126ca6fe685e16a3749463` |
| `VOXTRAL_FP32_NONBEAVER_CPU_PREFLIGHT_V5.json` | `5ef97e47e66c37b6c8ecb64b366daf4a713cfea1dfdee2f7f59ca75324b08401` |
| `voxtral_fp32_nonbeaver_byte_budget_probe_v5.py` | `f666e1df888e7bd8c87b1772b8031bfbf3c819345b607ed3e165ddaf979666ec` |
| `VOXTRAL_FP32_NONBEAVER_BYTE_BUDGET_V1.json` | `53d19e065669cce63256f39c3b1112bdbbdb52691f2fe5ae475f865ad6191a7f` |

I verified that the current files match their blobs in commit `36064b8`.
All eight runner CPU tests pass. The CPU preflight recomputes the four
selected 16-kHz PCM WAV, mask, prompt/batch, checkpoint and source hashes,
and the fixed four-probe digest; its saved receipt matches the recomputation.
The two voices use the prespecified 60/15 and 73/18 encoder/projector rows.

The runner fixes 96 patched forwards and eight VJPs per context, with one
passive donor and one passive recipient capture, two site-specific G1 clean
graphs, two N0 clean references and two natural generations. The G1 source
leaf retains a gradient path at both encoder 31 and projector in fake-model
CPU checks. The alpha-zero branch clones source bytes exactly, including
negative zero, and the clean/paired-zero comparisons use tensor-byte hashes
rather than `torch.equal`. Each paired G1 forward releases its live leaf and
target references before the next forward. Full source, projector and LM
input tensors are retained for zero calls; edited calls retain selected rows
and full-state hashes, a limitation the V5 protocol discloses.

I independently reran the full-size synthetic serialization probe. It
reproduced its receipt exactly: `1,302,437,429` bytes for Piper and
`1,318,364,789` for Kokoro, below the `2,122,317,824`-byte worker cap.
This checks schema size and serialization, not model values or the eventual
auditor result. The four-context sidecar estimate is `5,241,604,436` bytes,
below the 8-GiB campaign output cap. I also checked a fake encoder-site G1
VJP separately: the LM29 target was connected and its source gradient was
finite and nonzero.

No Voxtral weights were loaded, no live model work was run, and no new
audio-model or security result follows from this source review. The launcher
must still bind the exact source hashes, one-shot sequence, time/memory/file
caps and reviewed auditor before an external freeze is eligible.
