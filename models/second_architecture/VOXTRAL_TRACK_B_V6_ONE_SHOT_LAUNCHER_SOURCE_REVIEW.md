# V6 one-shot launcher source handoff

Status: **source-only HOLD**. This is executable wiring for a future, independently reviewed resource pilot. No candidate model weights or fresh 32-context Track B audio were loaded by this change.

## Frozen source and checks

- Launcher: `voxtral_track_b_v6_one_shot_launcher_source_v1.py`, SHA-256 `fb0124501fab2f92097dff64e9ed66f27e86743000dd1f491670369f39b19b52`.
- Fake-only tests: `test_voxtral_track_b_v6_one_shot_launcher_source_v1.py`, SHA-256 `811abe641f26170d52d1b2042586b26bc5f616917c044ccabbee0a2b7e7b7402`.
- Verification: 21 tests and 15 subtests passed across the new launcher, freeze, batch adapter, and composition source tests. The new launcher tests use mocks and never load candidate weights.

## Dependency order

The acyclic freeze remains W → A → R → L. W is the reviewed worker. A must externally bind the exact W, batch adapter, pinned loader, MPS runtime, composition source, Piper silver old-FIT input, synthetic geometry, human 32/32 QA receipt, and fresh preflight. R must independently approve A. L is this launcher and must freeze the exact SHA-256 bytes of W, A, and R in its own source. Its A/R constants are deliberately `None` now, so its public entry fails before starting a child process.

Filling A/R constants changes L's hash. Independent review must pin that final L hash after A and R exist. Do not revise W, the adapter, loader, runtime, composition source, or protocols casually: each revision changes a hash upstream of A and requires a new external admission and review. The new launcher additionally checks the generation config's pad token before load and requires the loaded model's pad token to match.

## Live-path review points

1. Confirm that the candidate's actual 32/32 human listening export, independent QA review, external A, and independent R exist and match their fixed hashes. There is no live GO today.
2. Review the child path: 30-GiB MPS allocator, pinned no-model processor batches, MPS transfer, pinned FP32 loader within the worker's 88-call schedule, 240-second parent watchdog, sampled MPS high-water, fresh pre/post host logs, and full 2-GiB output accounting.
3. The launcher writes a sidecar using the schema of the frozen composition shell, while the existing saved auditor binds that shell's source hash. Review this provenance relationship before a live pilot; the new launcher is the actual sidecar writer.
4. The saved audit returns `HOLD_no_post_pilot_go`. A measured resource result and independent post-pilot review are still required before the fresh Track B study.
