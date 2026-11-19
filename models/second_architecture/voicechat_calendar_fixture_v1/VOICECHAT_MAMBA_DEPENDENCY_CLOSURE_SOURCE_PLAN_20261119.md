# VoiceChat sample: Mamba dependency closure before another model run

19 November 2026. **Read-only source and package-metadata review; no A100 mutation or model inference in this review.** The host's fresh import-only smoke now passes with `pyarrow==25.0.1`, `numpy==1.26.4`, and `datasets==5.0.1`, including import of the official `offline_voicechat` module. This clears the observed attempt-002 import blocker. It does not prove that Mamba kernels or VoiceChat weights load.

## Exact candidate closure

Keep `mamba-ssm==2.3.2.post1`, Torch 2.10.0, and all VoiceChat card pins. The [Mamba v2.3.2.post1 package declaration](https://github.com/state-spaces/mamba/blob/v2.3.2.post1/pyproject.toml) requires `tilelang==0.1.8`, `apache-tvm-ffi<=0.1.9`, and `quack-kernels>=0.3.4`. The saved attempt-002 freeze has none of these because Mamba was installed with `--no-deps`. The following exact candidate closes their required dependency graph while preserving existing frozen versions:

| New package | Candidate pin | Why |
| --- | --- | --- |
| [`tilelang`](https://pypi.org/pypi/tilelang/0.1.8/json) | `0.1.8` | Exact Mamba requirement; Python 3.12 accepts its `cp38-abi3` Linux x86_64 wheel. |
| [`apache-tvm-ffi`](https://pypi.org/pypi/apache-tvm-ffi/0.1.9/json) | `0.1.9` | Satisfies Mamba `<=0.1.9`, TileLang `>=0.1.2,~=0.1.0`, and Quack `>=0.1.6,<0.2`; `cp312-abi3` Linux x86_64 wheel exists. |
| [`quack-kernels`](https://pypi.org/pypi/quack-kernels/0.3.4/json) | `0.3.4` | Lowest version satisfying Mamba's `>=0.3.4`; universal wheel. |
| [`torch-c-dlpack-ext`](https://pypi.org/pypi/torch-c-dlpack-ext/0.1.5/json) | `0.1.5` | Required by TileLang and Quack; Python 3.12 Linux x86_64 wheel; declares `torch` without a stricter version. |
| [`nvidia-cutlass-dsl`](https://pypi.org/pypi/nvidia-cutlass-dsl/4.4.2/json) | `4.4.2` | Lowest version meeting Quack's `>=4.4.2`; pins its base library to the same version. |
| [`nvidia-cutlass-dsl-libs-base`](https://pypi.org/pypi/nvidia-cutlass-dsl-libs-base/4.4.2/json) | `4.4.2` | Required exactly by CUTLASS DSL; Python 3.12 manylinux x86_64 wheel. |
| [`cuda-python`](https://pypi.org/pypi/cuda-python/12.9.4/json) | `12.9.4` | CUTLASS base requires `cuda-python>=12.8`; this version uses already frozen `cuda-bindings==12.9.4`. It does **not** establish kernel compatibility with the host's CUDA 12.8 Torch runtime. |
| [`z3-solver`](https://pypi.org/pypi/z3-solver/4.15.4.0/json) | `4.15.4.0` | Within TileLang's `>=4.13.0,<4.15.5`; `py3-none-manylinux` x86_64 wheel. |

The saved freeze already contains TileLang's other ordinary dependencies (`cloudpickle`, `ml_dtypes`, `numpy`, `psutil`, `torch`, `tqdm`, `typing_extensions`) and Mamba's other dependencies (`triton`, `ninja`, `einops`, `transformers`, `packaging`, `setuptools`). Extra groups such as CUDA 13, visualization, and development are not requested. Treat this as a **metadata-compatible candidate**, not a runtime compatibility claim.

## Installation and verification gate

Use a normal dependency-resolving install with all eight exact new pins, `--only-binary=:all:`, and a constraints file generated from the **fresh post-PyArrow host freeze**. Exclude its editable NeMo `-e git+...` line from the constraints file because it is not a version constraint; preserve the reviewed NeMo checkout separately. First run `pip install --dry-run --report` and inspect the proposed wheel files and SHA-256 values. The plan is acceptable only if the resolver changes no existing pinned package, including `numpy`, `pyarrow`, `torch`, `triton`, `cuda-bindings`, and the eleven card pins. If the resolver needs another change, stop for source review rather than silently broadening the repair. Avoid `--no-deps`: it would reproduce the incomplete declared environment.

After installation, capture full `pip freeze`, `pip check`, wheel hashes, and import stdout/stderr. In a fresh process with the reviewed environment and `PYTHONDONTWRITEBYTECODE=1`, import `mamba_ssm` and `Mamba3`, and re-import `nemo.collections.speechlm2.inference.utils.offline_voicechat`; use `importlib.metadata.version` to assert all eight pins and selected card pins. This check must not load the checkpoint or process audio. A passing `pip check` and these imports establish package closure only. Kernel execution can still fail due to CUDA, GPU, or ABI details; report such a failure as an environment finding. If the existing `decord` platform warning persists, preserve and review it rather than claiming `pip check` is clean.

Create a new host-specific signed freeze and a new output directory only after the dependency and import evidence pass independent review. Attempt 002 remains an immutable failed import attempt. A full sample retry remains sample-only, and no VoiceChat security behavior can be inferred from these dependency repairs.

Confidence: **high** in the declared dependency closure and Python 3.12 wheel availability from PyPI metadata; **moderate** in runtime compatibility until the A100 import smoke; **unknown** for native kernel execution until an independently reviewed sample run.
