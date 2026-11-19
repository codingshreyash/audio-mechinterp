"""Reconstruct the first Wav2Vec2 GroupNorm from independently audited FIT captures.

This is a posthoc algebraic diagnostic of saved tensors, not model inference or an
attacker-controlled audio experiment. It makes no claim about downstream CTC use.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open


HERE = Path(__file__).resolve().parent
RUN = HERE / "runs/wav2vec2_fit_baseline_stage_capture_v1/20261118T2352Z"
INVENTORY = HERE / "WAV2VEC2_FIT_BASELINE_V2_SIDECAR_MANIFEST_V1.json"
CHECKPOINT = Path.home() / ".cache/huggingface/hub/models--facebook--wav2vec2-base-960h/snapshots/22aad52d435eb6dbaf354bdad9b0da84ce7d6156/model.safetensors"
EXPECTED_RECORDS_SHA = "c9865f4ec306e57baf7dbfc7bd05037ec769034d95db70440972904fcab2507c"
EXPECTED_CHECKPOINT_SHA = "8aa76ab2243c81747a1f832954586bc566090c83a0ac167df6f31f0fa917d74a"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def compute() -> dict:
    records = RUN / "records.jsonl"
    audit = RUN / "audit.json"
    require(sha256(records) == EXPECTED_RECORDS_SHA, "baseline records hash changed")
    a = json.loads(audit.read_text())
    require(a["status"] == "independent_saved_data_audit_pass" and
            a["records_sha256"] == EXPECTED_RECORDS_SHA, "baseline audit failed or changed")
    receipt = json.loads(records.read_text().splitlines()[0])["receipt"]
    require(sha256(CHECKPOINT) == EXPECTED_CHECKPOINT_SHA == receipt["checkpoint_sha256"],
            "checkpoint hash changed")
    inventory = {f["path"]: f for f in json.loads(INVENTORY.read_text())["files"]}
    with safe_open(CHECKPOINT, framework="pt", device="cpu") as f:
        prefix = "wav2vec2.feature_extractor.conv_layers.0.layer_norm."
        gamma = f.get_tensor(prefix + "weight")[None, :, None]
        beta = f.get_tensor(prefix + "bias")[None, :, None]

    def capture(case_key: str, stage: str) -> torch.Tensor:
        rel = f"captures/C_{case_key}_{stage}.npy"
        path = RUN / rel
        item = inventory[rel]
        require(path.stat().st_size == item["size_bytes"] and sha256(path) == item["sha256"],
                f"capture changed: {rel}")
        arr = np.load(path, allow_pickle=False)
        require(arr.dtype == np.float32 and arr.ndim == 3 and arr.shape[0] == 1,
                f"capture dtype/shape changed: {rel}")
        return torch.from_numpy(arr)

    grouped: dict[tuple[str, str], list[dict]] = {}
    for c in receipt["cases"]:
        grouped.setdefault((c["pair_id"], c["engine"]), []).append(c)
    require(len(grouped) == 4 and all(len(v) == 2 for v in grouped.values()),
            "unexpected pair/voice population")
    rows = []
    torch.set_num_threads(4)
    for (pair, engine), cases in sorted(grouped.items()):
        cases.sort(key=lambda c: c["word"])
        require(cases[0]["slot_samples"] == cases[1]["slot_samples"],
                "paired slots differ")
        raw = [capture(c["key"], "first_conv_raw") for c in cases]
        norm = [capture(c["key"], "first_conv_norm") for c in cases]
        require(raw[0].shape == raw[1].shape == norm[0].shape == norm[1].shape,
                "paired first-conv shapes differ")
        start, end = cases[0]["slot_samples"]
        frame = torch.arange(raw[0].shape[-1])
        # Pinned conv0 kernel=10, stride=5, no padding. An output frame overlaps
        # the edited PCM slot iff its receptive interval intersects [start,end).
        overlap = (5 * frame < end) & (5 * frame + 9 >= start)
        outside = ~overlap
        require(bool(outside.any()) and bool(overlap.any()), "invalid overlap mask")
        raw_off = (raw[0] - raw[1]).abs()[:, :, outside].max().item()
        actual_off = (norm[0] - norm[1]).abs()[:, :, outside].max().item()
        require(raw_off == 0.0, "shared-normalized first-conv raw outside contrast is not exact zero")

        # This is the GroupNorm(num_groups=channels) formula, applied across time
        # independently to every channel, using the checkpoint's learned affine.
        errors = []
        for x, y in zip(raw, norm):
            mean = x.mean(dim=-1, keepdim=True)
            var = x.var(dim=-1, unbiased=False, keepdim=True)
            predicted = (x - mean) * torch.rsqrt(var + 1e-5) * gamma + beta
            errors.append((predicted - y).abs().max().item())
        require(max(errors) < 2e-5, "saved GroupNorm output does not match pinned formula")

        # Algebraic counterfactual only: use identical statistics from frames
        # whose raw convolution is exactly identical in both recordings.
        shared_mean = raw[0][:, :, outside].mean(dim=-1, keepdim=True)
        shared_var = raw[0][:, :, outside].var(dim=-1, unbiased=False, keepdim=True)
        cf = [(x - shared_mean) * torch.rsqrt(shared_var + 1e-5) * gamma + beta for x in raw]
        cf_off = (cf[0] - cf[1]).abs()[:, :, outside].max().item()
        require(cf_off == 0.0 and actual_off > 0, "unexpected outside contrast")
        rows.append({
            "pair_id": pair, "engine": engine, "words": [c["word"] for c in cases],
            "raw_frames": int(raw[0].shape[-1]), "overlap_frames": int(overlap.sum().item()),
            "raw_outside_max_abs_difference": raw_off,
            "saved_groupnorm_outside_max_abs_difference": actual_off,
            "formula_max_abs_error_by_word": errors,
            "shared_outside_stat_counterfactual_max_abs_difference": cf_off,
        })
    return {
        "status": "posthoc_saved_tensor_formula_reconstruction",
        "records_sha256": EXPECTED_RECORDS_SHA,
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA,
        "scope": "Four synthetic FIT pair/voice cells, common-normalized input. Algebraic first-GroupNorm reconstruction only; no downstream model run or security claim.",
        "rows": rows,
    }


if __name__ == "__main__":
    print(json.dumps(compute(), indent=2))
