# J++ Lens source reproduction

**18 November 2026.** Cloned the released [J++ Lens repository](https://github.com/safety-research/jpp_lens) at commit `d7efc0de372e20589e899188c0cfe8434acc63ec` into the ignored local `jpp_lens/` directory. Its README describes FIT-time activation clustering and signed expert weights that produce one fixed inference map per layer, plus an LRP backward graph and token filtering.

From that repository's root, ran its CPU test tree with the existing Anthropic reproduction environment:

```sh
PYTHONPATH=src ../../anthropic_jlens/.venv/bin/python -m pytest -q src
```

Result: **264 passed**, 59 upstream `transformers` deprecation warnings, in 6.49 s. The environment used Python 3.12, torch 2.15.0, Transformers 5.23.0, and pytest 9.2.0, rather than the repository's exact pinned versions. An initial run from the parent working directory had two relative-path data-file failures; rerunning from the repository root passed all tests. This verifies the released CPU code path under the stated environment. It does **not** fit a lens, reproduce the published 55.2% recall@10 result, or validate an audio port.
