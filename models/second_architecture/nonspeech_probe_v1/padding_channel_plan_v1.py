"""No-weight, frozen schedule for a prospective Whisper padding-channel test.

This module reads only the engineered WAV manifest. It never imports a model,
reads prior outputs, or performs candidate inference.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "runs/render_20261119_0740Z/manifest.json"
MANIFEST_SHA256 = "a37e6e6027611555cd65ceb1a8d9288b79c72460ae1155469eb1f094d7ddee90"
SILENCE = ("d02_silence", "d08_silence")
SITES = ("log_mel", "pre_block0")
PHASES = ("detect", "transcribe")
CELLS = ("RR", "RD", "DR", "DD")  # active source, then nominal-padding source
CONTROLS = (
    "read_only", "self", "random_active_1", "random_active_2",
    "random_padding_1", "random_padding_2",
)
PHASE_SPLITS = ("padding_detect_only", "padding_transcribe_only")
ARM_ORDER = ("RR", "read_only", "RD", "DR", "DD", "self",
             "random_active_1", "random_active_2", "random_padding_1",
             "random_padding_2", *PHASE_SPLITS)
MAX_NEW_TOKENS = 128
VOCAB_SIZE = 51865
DECODER_WIDTH = 384
WALL_SECONDS = 1800
RSS_LIMIT_BYTES = 4 * 1024**3
EVIDENCE_LIMIT_BYTES = 2 * 1024**3
EVIDENCE_RESERVE_BYTES = 128 * 1024**2
METADATA_ALLOWANCE_BYTES = 256 * 1024**2
SOURCE_SHA256 = {
    "feature_extraction_whisper.py": "dcce2e7820be059e657a9e12a60e8a55cfd37b1a82ee202ca15fc7d0934374cb",
    "modeling_whisper.py": "bf4f022d37ed523d09a472dae267b86007aaae0cd39d2ce37927de44bfc5b9e6",
    "generation_whisper.py": "e1531401f2cc2e389f13a7a55fa22542132164fefe9d431644c9368f3147583a",
}
PINNED_MODEL = "openai/whisper-tiny@169d4a4341b33bc18d8881c4b69c2e104e1cc0af"
PINNED_WEIGHT_SHA256 = "7ebd0e69e78190ffe1438491fa05cc1f5c1aa3a4c4db3bc1723adbb551ea2395"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def population() -> tuple[str, ...]:
    if digest(MANIFEST) != MANIFEST_SHA256:
        raise ValueError("frozen engineered manifest changed")
    rows = json.loads(MANIFEST.read_text())["rows"]
    names = tuple(sorted(row["case_id"] for row in rows))
    if len(names) != 20 or len(set(names)) != 20 or not set(SILENCE).issubset(names):
        raise ValueError("engineered population changed")
    for row in rows:
        wav = MANIFEST.parent / row["wav"]
        if not wav.is_file() or digest(wav) != row["wav_sha256"]:
            raise ValueError(f"frozen WAV changed: {row['case_id']}")
    for duration in (2, 8):
        prefix = f"d{duration:02d}_"
        if sum(name.startswith(prefix) for name in names) != 10:
            raise ValueError("duration stratum changed")
        if any(row["frames"] != duration * 16_000 for row in rows
               if row["case_id"].startswith(prefix)):
            raise ValueError("waveform length changed")
    return names


def donor_for(case_id: str) -> str:
    if case_id.startswith("d02_"):
        return SILENCE[0]
    if case_id.startswith("d08_"):
        return SILENCE[1]
    raise ValueError("unknown duration stratum")


def geometry(case_id: str, site: str) -> dict[str, int]:
    if site not in SITES:
        raise ValueError("unknown intervention site")
    seconds = 2 if donor_for(case_id) == SILENCE[0] else 8
    active = seconds * (100 if site == "log_mel" else 50)
    total = 3000 if site == "log_mel" else 1500
    return {"active": active, "padding": total - active, "total": total}


def seed(case_id: str, site: str, arm: str, phase: str) -> int:
    if site not in SITES or arm not in CONTROLS or not arm.startswith("random_") \
            or phase not in PHASES:
        raise ValueError("invalid random-arm identity")
    payload = f"nonspeech-padding-v1|{case_id}|{site}|{arm}|{phase}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def schedule() -> tuple[dict, ...]:
    cases = tuple(name for name in population() if name not in SILENCE)
    if len(cases) != 18:
        raise ValueError("non-silence population changed")
    rows: list[dict] = []
    for donor in SILENCE:
        rows.append({"site": "none", "case_id": donor, "donor_id": donor,
                     "arm": "no_hook", "hook_policy": "none",
                     "geometry": geometry(donor, "log_mel")})
        rows.append({"site": "both", "case_id": donor, "donor_id": donor,
                     "arm": "read_only_donor_capture", "hook_policy": "capture_both_sites_both_phases",
                     "geometry": geometry(donor, "log_mel")})
    for site in SITES:
        for case_id in cases:
            common = {"site": site, "case_id": case_id,
                      "donor_id": donor_for(case_id),
                      "geometry": geometry(case_id, site)}
            for arm in ARM_ORDER:
                policy = "none" if arm == "RR" else (
                    "capture_both_sites_both_phases" if arm == "read_only" else "edit")
                row = {**common, "arm": arm, "hook_policy": policy}
                if arm.startswith("random_"):
                    row["seeds"] = {phase: seed(case_id, site, arm, phase)
                                    for phase in PHASES}
                rows.append(row)
    if len(rows) != 436:
        raise ValueError("padding-channel arm count changed")
    validate_capture_dependencies(rows)
    return tuple(rows)


def validate_capture_dependencies(rows: list[dict] | tuple[dict, ...]) -> None:
    donor_baselines: set[str] = set()
    captured_donors: set[str] = set()
    recipient_baselines: set[tuple[str, str]] = set()
    captured_recipients: set[tuple[str, str]] = set()
    for row in rows:
        arm = row["arm"]
        key = (row["site"], row["case_id"])
        if arm == "no_hook":
            donor_baselines.add(row["case_id"])
        elif arm == "read_only_donor_capture":
            if row["case_id"] not in donor_baselines:
                raise ValueError("donor capture precedes its no-hook baseline")
            captured_donors.add(row["case_id"])
        elif arm == "RR":
            if row["donor_id"] not in captured_donors:
                raise ValueError("recipient baseline precedes donor capture")
            recipient_baselines.add(key)
        elif arm == "read_only":
            if row["donor_id"] not in captured_donors or key not in recipient_baselines:
                raise ValueError("recipient capture precedes donor capture or baseline")
            captured_recipients.add(key)
        elif row["hook_policy"] == "edit":
            if row["donor_id"] not in captured_donors or key not in captured_recipients:
                raise ValueError("edit precedes its donor or recipient capture")


def worst_case_core_evidence_bytes() -> int:
    """Conservative float32 core estimate; actual writes still need a hard cap."""
    rows = schedule()
    # Full per-step raw logits: both no-hook donors and all 36 DD calls.
    full_calls = sum(row["arm"] in ("no_hook", "DD") for row in rows)
    first_vectors = len(rows) * 2  # raw logits and processed scores
    # Conservative common tensor shape for both sites, both phases, all 20 files.
    native_states = 20 * len(SITES) * len(PHASES)
    return ((full_calls * MAX_NEW_TOKENS + first_vectors) * VOCAB_SIZE * 4
            + native_states * 1500 * 384 * 4
            + len(rows) * DECODER_WIDTH * 4
            + len(rows) * MAX_NEW_TOKENS * 4)  # selected-token score scalars


def resource_budget_fits() -> bool:
    return (worst_case_core_evidence_bytes() + METADATA_ALLOWANCE_BYTES
            + EVIDENCE_RESERVE_BYTES < EVIDENCE_LIMIT_BYTES)


def verify_installed_source_bytes() -> None:
    base = (HERE.parent / ".venv/lib/python3.12/site-packages/transformers/models/whisper")
    for filename, expected in SOURCE_SHA256.items():
        source = base / filename
        if not source.is_file() or digest(source) != expected:
            raise ValueError(f"pinned Whisper source changed: {filename}")


def schedule_sha256() -> str:
    raw = json.dumps(schedule(), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


if __name__ == "__main__":
    print(json.dumps({"schema": "nonspeech_padding_channel_plan_v3",
                      "non_silence_cases": 18, "silence_baselines": 2,
                      "donor_read_only_capture_calls": 2,
                      "per_case_site_arms": 12, "total_calls": len(schedule()),
                      "worst_case_core_evidence_bytes": worst_case_core_evidence_bytes(),
                      "resource_budget_fits": resource_budget_fits(),
                      "schedule_sha256": schedule_sha256()}, sort_keys=True))
