"""Portable Linux source/tokenizer audit; no checkpoint weights are imported."""
from __future__ import annotations

import importlib.metadata
import json
from hashlib import sha256
from pathlib import Path

from . import readout_v2
from .endpoint import SPEC_PATH, load_spec

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def audit() -> dict:
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer

    spec = load_spec()
    pins = json.loads((ROOT / "models/qwen_omni/pins.json").read_text())
    freeze = json.loads((HERE / "READOUT_FREEZE_V2.json").read_text())
    if (freeze.get("schema") != "qwen_action_method_readout_freeze_v2"
            or freeze.get("source_sites") != ["text.residual.17", "text.residual.34"]
            or freeze.get("target_dimension") != 3
            or freeze.get("fit_contexts") != readout_v2.FIT_CONTEXTS
            or freeze.get("fit_split") != "fit_text"
            or freeze.get("readout") != "affine_mean_target_vjp"
            or freeze.get("directional_intervention", {}).get(
                "alpha_times_recipient_state_l2") != 0.03):
        raise ValueError("frozen readout method changed")
    for key in ("model_id", "model_revision", "transformers_revision"):
        if spec[key] != pins[key]:
            raise ValueError(f"pinned {key} changed")
    if _sha(SPEC_PATH) != freeze["v1_spec_sha256"]:
        raise ValueError("frozen spec changed")
    if (spec["model_revision"] != freeze["model_revision"]
            or spec["transformers_revision"] != freeze["transformers_revision"]):
        raise ValueError("frozen model/source revision changed")
    distribution = importlib.metadata.distribution("transformers")
    base = Path(distribution.locate_file("transformers/models/qwen2_5_omni"))
    sources = {
        "modeling": (base / "modeling_qwen2_5_omni.py", spec["installed_modeling_sha256"]),
        "processing": (base / "processing_qwen2_5_omni.py", spec["installed_processing_sha256"]),
        "config": (ROOT / "models/qwen_omni/config.snapshot.json",
                   pins["config_snapshot_sha256"]),
    }
    for name, (path, expected) in sources.items():
        if not path.is_file() or _sha(path) != expected:
            raise ValueError(f"pinned {name} source bytes changed")
    files = {}
    for name, expected in freeze["tokenizer_files_sha256"].items():
        path = Path(hf_hub_download(spec["model_id"], name,
            revision=spec["model_revision"], local_files_only=True))
        files[name] = _sha(path)
        if files[name] != expected:
            raise ValueError(f"pinned tokenizer file changed: {name}")
    tokenizer = AutoTokenizer.from_pretrained(spec["model_id"],
        revision=spec["model_revision"], local_files_only=True)
    labels = {name: tokenizer.encode(name, add_special_tokens=False)
              for name in spec["labels"]}
    expected_labels = {name: [token] for name, token in
                       zip(spec["labels"], readout_v2.LABEL_TOKEN_IDS, strict=True)}
    if (labels != expected_labels or int(tokenizer.eos_token_id) != readout_v2.EOS_TOKEN_ID
            or readout_v2.EOS_TOKEN_ID not in tokenizer.all_special_ids):
        raise ValueError("pinned label/EOS tokenization changed")
    return {"status": "source_only_no_model_weights", "source_sha256":
            {name: _sha(path) for name, (path, _) in sources.items()},
            "tokenizer_file_sha256": files, "label_token_ids": labels}
