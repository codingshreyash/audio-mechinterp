# VoiceChat public sample: PyArrow/NumPy import repair

19 November 2026. **Source-only proposal; no environment or model execution in this review.** Preserve attempt 002 as a failed import smoke. Its [independent audit](VOICECHAT_EXISTING_A100_SAMPLE_ATTEMPT_002_INDEPENDENT_AUDIT_20261119.md) shows that the official child returned code 1 before VoiceChat weights or audio were processed. The saved traceback goes through NeMo speech/ASR imports, `datasets`, and `pyarrow.lib` to `ImportError: pyarrow requires NumPy 2.0 or newer, found 1.26.4`. The saved freeze has `pyarrow==26.0.1`, `numpy==1.26.4`, `datasets==5.0.1`; the eleven selected VoiceChat card package versions passed the runner's pin check. This is an environment failure, with no VoiceChat behavior observed.

## Narrow repair

In the dedicated Python 3.12 VoiceChat environment, pin **`pyarrow==25.0.1`** while retaining **`numpy==1.26.4`**, **`datasets==5.0.1`**, and all selected card pins. Install the prebuilt PyArrow wheel without an opportunistic dependency upgrade, then record a fresh freeze. This is a one-package change from attempt 002. Do not edit the checkpoint, official sample source, historical output, or historical freeze.

The version boundary has direct upstream support. Apache Arrow's [25.0.1 installation guide](https://arrow.apache.org/docs/25.0/python/install.html) lists optional NumPy 1.21.2 or higher. Its [26.0 import issue](https://github.com/apache/arrow/issues/53954) reproduces the exact NumPy 1.26.4 error, reports that 25.0.1 imports and converts a NumPy array on Python 3.12 Linux, and explains why the optional NumPy lower bound is absent from wheel runtime metadata. The [PyArrow 25.0.1 PyPI release](https://pypi.org/project/pyarrow/25.0.1/) provides a Python 3.12 Linux x86_64 wheel. The [Datasets 5.0.1 dependency declaration](https://github.com/huggingface/datasets/blob/5.0.1/setup.py) requires `pyarrow>=21.0.0` and `numpy>=1.17`, so the proposed pair is inside its declared bounds. These sources support the pin; a successful import on the actual host remains to be measured.

Upgrading NumPy to 2.x would also satisfy Arrow 26, but changes a foundational ABI dependency across the NeMo environment. A PyArrow rollback is the smaller experiment. The absence of a resolver conflict alone is insufficient: Arrow 26's optional NumPy requirement is not expressed as `Requires-Dist`, and the saved environment already passed installation before failing at import.

## No-model verification before a new freeze

Use the existing dedicated VoiceChat venv on the A100, or a fresh equivalent with the same reviewed pins. Record the exact install command, wheel filename/hash, and before/after `python -m pip freeze` diff. Use a binary wheel for `pyarrow==25.0.1`; verify the resolver did not silently change NumPy, Datasets, Torch, NeMo, or any selected card pin. In a fresh Python process, with `PYTHONDONTWRITEBYTECODE=1` and the reviewed NeMo checkout on `PYTHONPATH`, run only import-level checks:

```python
import numpy as np
import pyarrow as pa
import datasets

assert np.__version__ == "1.26.4"
assert pa.__version__ == "25.0.1"
assert datasets.__version__ == "5.0.1"
assert pa.array(np.arange(3)).to_numpy().tolist() == [0, 1, 2]

from nemo.collections.speechlm2.inference.utils import offline_voicechat
```

The final import is the same entry module reached by the pinned official sample. It must be performed without loading the checkpoint or passing a WAV. A successful smoke establishes only that this import obstacle is cleared; later imports or model loading may reveal other defects. Record stdout/stderr, interpreter path, package versions, `pip check`, and the NeMo checkout status. If generated metadata or bytecode changes the checkout, account for it in the new source review.

## Separate Mamba dependency gate

The prior environment installed `mamba-ssm==2.3.2.post1` with `--no-deps`; `pip check` reported missing `tilelang`, `apache-tvm-ffi`, and `quack-kernels`. This is not the cause of the saved PyArrow traceback, which stops before Mamba is imported. It is an unresolved completeness issue: the [upstream Mamba v2.3.2.post1 package declaration](https://github.com/state-spaces/mamba/blob/v2.3.2.post1/pyproject.toml) lists `tilelang==0.1.8`, `apache-tvm-ffi<=0.1.9`, and `quack-kernels>=0.3.4` as normal dependencies. Because Mamba imports and kernel selection may depend on them, do not call them irrelevant based on the present traceback. Before model execution, either install versions satisfying these declarations in the reviewed environment and smoke the exact VoiceChat/Mamba import path, or document and independently review an explicit exception supported by source and import evidence. Record the complete `pip check` output, including the separate reported `decord` platform warning if it persists. Keep `mamba-ssm==2.3.2.post1` and the card pins fixed.

## Retry gate and interpretation

Only after the host import smoke and Mamba dependency decision, create a **new** host-specific signed freeze with the fresh full `pip freeze`, all changed wheel hashes, official sample command, source hashes, and a new output directory. Obtain the independent executable review required by the existing run process. Do not reuse attempt 002's freeze or output directory. The retry should remain sample-only; no model finding follows from this repair proposal or from an import PASS.

Confidence: **high** that `pyarrow==25.0.1` is the narrow compatible candidate for the observed error; **unverified on the actual A100** until the import smoke completes. The missing Mamba dependencies remain a separate gate.
