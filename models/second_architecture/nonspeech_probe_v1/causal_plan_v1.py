"""Frozen, no-model schedule for the engineered non-speech Whisper causal assay.

This enumerates every source case and arm before any intervention forward. It
does not read the descriptive outputs or import/load a candidate model.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "runs/render_20261119_0740Z/manifest.json"
MANIFEST_SHA256 = "a37e6e6027611555cd65ceb1a8d9288b79c72460ae1155469eb1f094d7ddee90"

DONORS = ("d02_silence", "d08_silence")
MAX_NEW_TOKENS = 128
VOCAB_SIZE = 51865
MAX_EVIDENCE_BYTES = 2 * 1024**3
EVIDENCE_RESERVE_BYTES = 128 * 1024**2
METADATA_ALLOWANCE_BYTES = 256 * 1024**2
STAGE_A_ARMS = (
    "no_hook", "read_only", "self_detect", "self_transcribe", "self_both",
    "donor_detect", "donor_transcribe", "donor_both",
    "random_1_both", "random_2_both",
)
STAGE_B_ARMS = (
    "no_hook", "read_only", "self_both", "donor_both",
    "active_both", "displaced_both", "random_1_both", "random_2_both",
)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def frozen_cases() -> tuple[str, ...]:
    if digest(MANIFEST) != MANIFEST_SHA256:
        raise ValueError("engineered source manifest changed")
    rows = json.loads(MANIFEST.read_text())["rows"]
    names = tuple(sorted(row["case_id"] for row in rows))
    if len(rows) != 20 or len(set(names)) != 20 or set(DONORS) - set(names):
        raise ValueError("engineered case population changed")
    if sum(name.startswith("d02_") for name in names) != 10 or sum(
            name.startswith("d08_") for name in names) != 10:
        raise ValueError("engineered duration population changed")
    return names


def donor_for(case_id: str) -> str:
    if case_id.startswith("d02_"):
        return DONORS[0]
    if case_id.startswith("d08_"):
        return DONORS[1]
    raise ValueError(f"unknown case duration: {case_id}")


def encoder_active_rows(case_id: str) -> int:
    # Pinned processor gives 200 or 800 active mel frames; second conv stride 2.
    return {DONORS[0]: 100, DONORS[1]: 400}[donor_for(case_id)]


def random_seed(case_id: str, stage: str, arm: str, phase: str) -> int:
    if stage not in ("A", "B") or arm not in ("random_1_both", "random_2_both") \
            or phase not in ("detect", "transcribe"):
        raise ValueError("invalid random arm identity")
    payload = f"nonspeech-causal-v1|{case_id}|{stage}|{arm}|{phase}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def schedule() -> tuple[dict, ...]:
    cases = tuple(name for name in frozen_cases() if name not in DONORS)
    if len(cases) != 18:
        raise ValueError("non-silence population changed")
    rows: list[dict] = [
        {"stage": "donor", "case_id": name, "donor_id": name,
         "site": "none", "arm": "no_hook", "active_rows": encoder_active_rows(name),
         "save_full_raw_logits_each_step": True}
        for name in DONORS
    ]
    for stage, site, arms in (
        ("A", "final_encoder", STAGE_A_ARMS),
        ("B", "pre_block0", STAGE_B_ARMS),
    ):
        for name in cases:
            for arm in arms:
                row = {"stage": stage, "case_id": name,
                       "donor_id": donor_for(name), "site": site,
                       "arm": arm, "active_rows": encoder_active_rows(name),
                       "save_full_raw_logits_each_step": arm == "donor_both"}
                if arm.startswith("random_"):
                    row["random_support_rows"] = [0, 1500]
                    row["random_norm_reference"] = "full_donor_minus_recipient_l2_same_site_phase"
                    row["seeds"] = {phase: random_seed(name, stage, arm, phase)
                                    for phase in ("detect", "transcribe")}
                rows.append(row)
    if len(rows) != 326:
        raise ValueError("causal arm count changed")
    return tuple(rows)


def worst_case_core_evidence_bytes() -> int:
    """Conservative raw float32 sidecar budget, before metadata and reserve."""
    rows = schedule()
    full_steps = sum(row["save_full_raw_logits_each_step"] for row in rows)
    first_steps = len(rows)
    native_states = 20 * 2 * 2  # cases × sites × phase
    return ((full_steps * MAX_NEW_TOKENS + first_steps) * VOCAB_SIZE * 4
            + native_states * 1500 * 384 * 4)


def main() -> None:
    rows = schedule()
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    print(json.dumps({"schema": "nonspeech_whisper_causal_plan_v1",
                      "population": 18, "donor_baselines": 2,
                      "stage_a": 180, "stage_b": 144, "total": len(rows),
                      "worst_core_evidence_bytes": worst_case_core_evidence_bytes(),
                      "max_evidence_bytes": MAX_EVIDENCE_BYTES,
                      "schedule_sha256": hashlib.sha256(payload).hexdigest()},
                     sort_keys=True))


if __name__ == "__main__":
    main()
