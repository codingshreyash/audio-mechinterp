# VoiceChat fixture child-environment independent source rereview v3

Decision: **GO for the exact bounded production-adapter source in `98291f99ae01dc1a539f071baad0669c20b65203`.** This resolves the direct-worker environment HOLD in `VOICECHAT_PRODUCTION_ADMISSION_INDEPENDENT_SOURCE_REREVIEW_HOLD_V2.md`. This is a source decision only: no VoiceChat weights, NeMo model, GPU, or live calendar fixture ran in this review.

| Changed source role | SHA-256 |
| --- | --- |
| Worker | `6f3664ed53ac9aac2a45e078f06adb3e11e0ce856a9cb9f820c2f4255d795a75` |
| Worker tests | `612e1d977b61caacd10226d310d1064276c02cfc15e972614d3524352ad7f658` |
| Production admission handoff | `3cb49d2b675adefa983e2fa110fe603386030cfb348140c2ce32480c86f24383` |

The canonical full 18-role source-map SHA-256 is `877b3a37eab653d3b534e1d0f5e79747730e30d9e7b6e6ac103998ff213b30ee`. `verify_source_commit(98291f99ae01dc1a539f071baad0669c20b65203, source_hashes())` passed. The admission and supervisor roles are unchanged from the prior rereview.

## Evidence

- The production wrapper computes the frozen public allowlist before admission. A `finally` restores exactly that dictionary after admission, removing both admission keys and **all** unrelated parent variables even if admission throws. It then compares the entire process environment with the admitted frozen public dictionary. A fake wrapper test with `UNRELATED_PARENT_SECRET` observed only the frozen dictionary at the fake arm boundary; a separate admission-exception test observed the unrelated variable and both keys removed.
- The direct `run_fixture_arm` production path checks `dict(os.environ)` against the admitted dictionary before QA or the official helper. An independent fake direct-worker probe with `UNRELATED_PARENT_SECRET` rejected the call before either factory. A fake helper that added `TRANSIENT_BEFORE_MODEL` was rejected after helper return and before the model factory. A clean fake production arm reached its model factory with the exact dictionary.
- The wrapper restores the frozen dictionary after Torch import and deterministic seeding. An independent fake `torch.manual_seed` mutation that added `TORCH_TRANSIENT` was removed before the fake arm. A fake arm that added `TRANSIENT_AFTER_FACTORY` was rejected by the post-arm full-environment comparison.
- The freeze still binds the public environment and its canonical SHA-256; the supervisor launches the child from that dictionary plus only the two required admission keys. The child scrubs those keys before the model boundary. Source/input rehashing and the prior tools-hash protection remain in place.

All **21** no-model admission, supervisor, and fixture-worker unit tests passed. `git diff --check 98291f9^ 98291f9` passed. The GO machine record beside this review binds all 18 exact source roles and the full commit.

This GO does **not** establish live fixture readiness or a security finding. Default external QA/anchor and sample-receipt hooks still fail closed; the all-12 human listening QA, independent anchor/grants, executable freeze/review, official sample receipt, and actual adapter capture remain required. This one-process slice does not run the three stock arms or compare final text/function/TTS parity.
