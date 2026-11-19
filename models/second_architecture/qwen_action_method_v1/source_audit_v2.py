"""Local-only tokenizer/source audit for the V2 three-target readout freeze."""
from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

from .endpoint import ROOT, SPEC_PATH, load_spec, validate_local_source
from .readout_v2 import EOS_TOKEN_ID, FIT_CONTEXTS, LABEL_TOKEN_IDS

FREEZE_PATH = Path(__file__).with_name("READOUT_FREEZE_V2.json")


def audit() -> dict:
    # Importing a tokenizer and downloading nothing loads no model weights.
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer

    freeze = json.loads(FREEZE_PATH.read_text())
    if freeze.get("schema") != "qwen_action_method_readout_freeze_v2":
        raise ValueError("readout freeze schema changed")
    if (freeze.get("source_sites") != ["text.residual.17", "text.residual.34"]
            or freeze.get("target_dimension") != 3
            or freeze.get("fit_contexts") != FIT_CONTEXTS
            or freeze.get("fit_split") != "fit_text"
            or freeze.get("readout") != "affine_mean_target_vjp"
            or freeze.get("directional_intervention", {}).get(
                "alpha_times_recipient_state_l2") != 0.03):
        raise ValueError("readout/causal freeze changed")
    if sha256(SPEC_PATH.read_bytes()).hexdigest() != freeze["v1_spec_sha256"]:
        raise ValueError("V1 wording/source binding changed")
    spec = load_spec()
    if (spec["model_revision"] != freeze["model_revision"]
            or spec["transformers_revision"] != freeze["transformers_revision"]):
        raise ValueError("model/source revision changed")
    source_hashes = validate_local_source(spec, ROOT)
    files = {}
    for name, expected in freeze["tokenizer_files_sha256"].items():
        path = Path(hf_hub_download(spec["model_id"], name,
                                    revision=spec["model_revision"],
                                    local_files_only=True))
        files[name] = sha256(path.read_bytes()).hexdigest()
        if files[name] != expected:
            raise ValueError(f"tokenizer file changed: {name}")
    tokenizer = AutoTokenizer.from_pretrained(spec["model_id"],
                                              revision=spec["model_revision"],
                                              local_files_only=True)
    labels = {label: tokenizer.encode(label, add_special_tokens=False)
              for label in spec["labels"]}
    expected = {label: [token_id] for label, token_id
                in zip(spec["labels"], LABEL_TOKEN_IDS, strict=True)}
    if (labels != expected or labels != {label: [token_id]
                                         for label, token_id in freeze["canonical_label_ids"].items()}
            or int(tokenizer.eos_token_id) != EOS_TOKEN_ID
            or int(freeze["eos_token_id"]) != EOS_TOKEN_ID
            or EOS_TOKEN_ID not in tokenizer.all_special_ids):
        raise ValueError("canonical label/EOS token path changed")
    return {"status": "source_only_no_model_weights", "v1_spec_sha256": freeze["v1_spec_sha256"],
            "installed_source_sha256": source_hashes,
            "tokenizer_file_sha256": files, "label_token_ids": labels,
            "eos_token_id": EOS_TOKEN_ID}


if __name__ == "__main__":
    print(json.dumps(audit(), sort_keys=True))
