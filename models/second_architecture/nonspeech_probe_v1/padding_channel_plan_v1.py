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
                     "arm": "no_hook", "geometry": geometry(donor, "log_mel")})
    for site in SITES:
        for case_id in cases:
            common = {"site": site, "case_id": case_id,
                      "donor_id": donor_for(case_id),
                      "geometry": geometry(case_id, site)}
            for arm in CELLS + CONTROLS + PHASE_SPLITS:
                row = {**common, "arm": arm}
                if arm.startswith("random_"):
                    row["seeds"] = {phase: seed(case_id, site, arm, phase)
                                    for phase in PHASES}
                rows.append(row)
    if len(rows) != 434:
        raise ValueError("padding-channel arm count changed")
    return tuple(rows)


def schedule_sha256() -> str:
    raw = json.dumps(schedule(), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


if __name__ == "__main__":
    print(json.dumps({"schema": "nonspeech_padding_channel_plan_v1",
                      "non_silence_cases": 18, "silence_baselines": 2,
                      "per_case_site_arms": 12, "total_calls": len(schedule()),
                      "schedule_sha256": schedule_sha256()}, sort_keys=True))
