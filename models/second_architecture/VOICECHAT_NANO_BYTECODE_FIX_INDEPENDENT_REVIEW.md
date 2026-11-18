# Independent review: VoiceChat Nano bytecode gate

**18 November 2026.** Reviewed `e453914` against the remaining model-load blocker in [the prior independent review](VOICECHAT_NANO_CACHE_FIX_INDEPENDENT_REVIEW.md). The reviewed preflight SHA-256 is `a810aa54196fa4ac82653ff0a3569481564fd085e5ab31f5ebf2a55f28e02604`; the plan SHA-256 is `4e6e67c24ad765c2c6f5677b6718811102c467f01f1f943842bcc0b0e109bdc0`. This is a local source and synthetic-cache review. No remote instance, model, ASR, or active experiment output was accessed.

## Decision

**The reported bytecode blocker is closed in the documented `ready` → `preload` sequence.** The offline Nano resolver now starts Python with `-B`, sets `PYTHONDONTWRITEBYTECODE=1`, removes `PYTHONPYCACHEPREFIX`, and asserts `sys.dont_write_bytecode`. The generated-cache allowlist rejects every `__pycache__` directory and `.pyc` file, as well as extra modules and changed pinned `.py` files. The new `preload` phase rechecks the pinned NeMo tree, VoiceChat config/checkpoint, fixture files and receipts, six-file Nano hub cache, and two generated source files without importing Nano code. It also requires the canonical `HF_HOME`/`HF_HUB_CACHE`/`HF_MODULES_CACHE` paths, offline flags, `PYTHONDONTWRITEBYTECODE=1`, and no `PYTHONPYCACHEPREFIX` in the shell intended for the later model process.

`--self-test` returned `self_test_passed_no_model_or_network`. An additional independent synthetic cache accepted the two expected source files but rejected both a `__pycache__/modeling_nemotron_h.cpython-312.pyc` and a direct `modeling_nemotron_h.pyc`. The same synthetic check rejected `PYTHONPYCACHEPREFIX` in `preload`. These tests did not load a model.

The `preload` receipt is a point-in-time check: run it immediately before the model load in the same shell, preserve the cache and checked files, and rerun it after any change. The full VoiceChat load and no-hook/passive-hook identity pilot remain separate required gates. This review does not claim model feasibility, fixture inference, or a security finding.
