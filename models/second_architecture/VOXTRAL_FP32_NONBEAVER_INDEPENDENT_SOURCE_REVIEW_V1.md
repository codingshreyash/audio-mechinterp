# Independent V5 four-context runner and launcher source review (V3)

18 November 2026. This independent source-only rereview covers the corrected
four-context Voxtral FP32 non-beaver runner and launcher. It follows and
preserves the V1 combined HOLD `a609b49803c21a9cfd58b91fb718a10b5d92270a`
and V2 combined HOLD `beb1bfc4185bfc1f8b1d8e01ccda466f346347dd`.

Decision: GO for non-beaver FP32 runner and launch gate source only; model HOLD until external freeze.

## Exact reviewed source

| Item | SHA-256 |
| --- | --- |
| V5 protocol | `99de95003190de6b5fb55c5ba8bed446dfea7cdd8cf53c0569d304bcb942b49b` |
| V5 conceptual review | `e5e706d8536754209c79eb4f458e8f7ecc90527e84a1b4198e0a0895fc13d6a3` |
| Runner, `voxtral_fp32_nonbeaver_linearity_runner_v1.py` | `91e622f7fd8f4d9dac00ca7a0baaf02d74faf9326109d10674ebf5673310bc04` |
| Runner tests | `a4d9600fcd0371f1c0cf11a6ea6133d9f5ee7d097d126ca6fe685e16a3749463` |
| Launch gate, `voxtral_fp32_nonbeaver_linearity_launch_gate_v1.py` | `14cd58e9f5706be0da2453664a14aef11d7f5cb5f55f354b1028aca717df0a9d` |
| Gate tests | `239fe2e40222ff31176fd379378f889fec2d8654504718567e1420efc23a8fa9` |
| Saved-data auditor | `70a1ac3c4e4da92bb1fa1034bd866195024dcd423ea35a2386e233bce5dca8e2` |
| Auditor tests | `3ffaa2224ca0c29ed5a49a985cf58b230aaaca9faa539214fa26548ca978c15a` |
| V5 CPU preflight | `5ef97e47e66c37b6c8ecb64b366daf4a713cfea1dfdee2f7f59ca75324b08401` |
| V5 byte budget | `53d19e065669cce63256f39c3b1112bdbbdb52691f2fe5ae475f865ad6191a7f` |
| V5 byte-budget probe | `f666e1df888e7bd8c87b1772b8031bfbf3c819345b607ed3e165ddaf979666ec` |
| V1 combined HOLD | `2e4c769a66e63597cac3180c4b3b92da5e30d003b7cc9a93c542909778252aa4` |
| V2 combined HOLD | `7aaeab158bdfb0e9ebfa6fc794099e0eabf86ac009ec72babf33942282dbab4c` |
| Auditor V2 rereview | `000ea1aa18170eeb3309189c9ecf1b06278815eaa47587e764a76fb099f2bf3a` |

The gate and test correction is commit
`a47eca8719e97c8604ab006a3acf6af2ed170eb9`. The independently reviewed
saved-data auditor V2 GO is commit
`a142526f2f089786fd05e1264b1a7bb2ca4acb73`. This combined review is
separate from both source corrections and the auditor review.

## Independent checks

- All 16 CPU gate tests passed with the pinned
  `models/second_architecture/.venv` interpreter. The real read-only
  `--check-freeze` stopped only because this combined source review had not
  yet been committed; it made no campaign reservation or model load.
- The V1 failures remain closed: forged minimal prior finishes and changed
  parent-measured elapsed time cannot admit the next context, and negative
  prior model work or a reservation predating the freeze is rejected.
- The V2 failure is closed: a successful prior entry requires the exact
  `meta.json`, `progress.jsonl`, `model_layout.json`, `result.json`,
  `audit.json`, and one content-addressed tensor sidecar. The gate checks
  file type, exclusivity, hashes, size, complete planned counts, and
  progress/time order. Missing meta, progress, sidecar, or all supporting
  files now rejects admission. A prior success also receives a separate
  read-only CPU process that reloads its saved tensor payload, calls the
  pinned auditor's `audit_payload`, and compares the complete recomputed
  report to the saved audit output. A forged audit cell is rejected.
  Revalidation has a hard 120-second timeout and does not write into the
  prior attempt. A corroborated failed prior attempt can advance without
  retry; its consumed model work remains in cumulative accounting and all
  four planned contexts remain in the denominator.
- The four fixed ordered contexts are Piper silver, Piper teal, Kokoro
  silver, and Kokoro teal. The exact WAV SHA-256 values are respectively
  `fe554972f68b470fb0f4a1c844b874e70758279c281bba47e963792bef2634a9`,
  `112ed36abf5fb2be1ce77dcfffa9d1bb6deba10b4ecb2d14fc5960b8106626e9`,
  `bcad702fa8b55c453286cb0e83ee3336a4bb6ade5e5ac9d4a42dc82c1a97c26e`,
  and `5038dbe167ae2ba6c9317f74c17bd9bb8ef7d07bbdee9da49f6dc0731e119365`.
  The runner CPU preflight recomputes frozen processor batches, masks, basis,
  and installed source/checkpoint pins before model load.
- The parent enforces a 300-second worker process limit and a 120-second
  auditor limit. The runner enforces 30 GiB MPS driver memory. The gate
  binds 2 GiB per context, 8 GiB per campaign, 12 GiB free before
  reservation, and a prior-work-inclusive 4,500-second model-work ceiling.
  It rechecks source and freeze pins after worker and auditor subprocesses.
  The source and freeze reviews must follow the earlier HOLD reviews and
  source commits in distinct commits before the one-shot reservation.

## Scope and limit

This GO concerns source admission only. Saved-data recomputation establishes
consistency of the recorded tensors and CPU audit; it cannot attest that a
real model forward occurred or that the measured system is linear. The
external freeze and its independent review remain mandatory. No model
weights, fresh corpus output, reservation, or live four-context pilot were
opened or run for this review.
