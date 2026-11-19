"""Source-only readiness check for the authentic-speech Whisper experiment.

This module never imports or runs the target model. It intentionally cannot
issue an executable GO: the independent source review and a later, separate
GO receipt remain necessary after blinded human review is complete.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import wave
from pathlib import Path

from . import human_review_intake_v2 as intake

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PLAN = HERE / "WHISPER_HUMAN_SPEECH_SOURCE_PLAN_V1.json"
PROTOCOL = HERE / "WHISPER_HUMAN_SPEECH_MECHANISM_PROSPECTIVE_V1.md"
MANIFEST = HERE / "source_v1/manifest.json"
PROXY_AUDIT = HERE / "runs/asr_proxy_v1/attempt_001_audit_v2.json"
PROXY_REVIEW = HERE / "ASR_PROXY_INDEPENDENT_RESULT_V1.md"
PINS = ROOT / "models/second_architecture/nonspeech_probe_v1/descriptive_pins_v1.json"
SOURCE_REVIEW = HERE / "WHISPER_HUMAN_SPEECH_PLAN_INDEPENDENT_SOURCE_REVIEW_V1.md"
PLAN_SHA256 = "711e75985ceecfd60fc2f3c7702473ab71c55bdb1446ef89f4d40d13c5b1af65"


def need(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1 << 20), b""):
            h.update(part)
    return h.hexdigest()


def _regular(path: Path) -> None:
    need(path.is_file() and not path.is_symlink(), f"missing or symlink: {path}")


def frozen_plan() -> dict:
    """Recheck the reviewed source plan and all 16 source WAV bytes."""
    for path in (PLAN, PROTOCOL, MANIFEST, PROXY_AUDIT, PROXY_REVIEW, PINS, SOURCE_REVIEW):
        _regular(path)
    need(sha(PLAN) == PLAN_SHA256, "reviewed plan bytes changed")
    plan = json.loads(PLAN.read_text())
    expected = {
        "manifest_sha256": sha(MANIFEST), "protocol_sha256": sha(PROTOCOL),
        "proxy_audit_sha256": sha(PROXY_AUDIT),
        "proxy_review_sha256": sha(PROXY_REVIEW), "pins_sha256": sha(PINS),
    }
    need(plan.get("bindings") == expected, "source plan binding changed")
    need(plan.get("status") == "source_only_hold_no_target_inference"
         and plan.get("counts") == {"base_replays": 4,
                                     "development_comparisons": 4,
                                     "primary_comparisons": 8,
                                     "sealed_test_comparisons": 4,
                                     "speakers": 4, "wav_calls": 16},
         "source plan census changed")
    calls = plan.get("calls")
    need(isinstance(calls, list) and len(calls) == 16, "16-call plan changed")
    need({r["split"] for r in calls} == {"development", "sealed_test"},
         "development/test split changed")
    for row in calls:
        path = ROOT / row["wav"]
        _regular(path)
        need(path.is_relative_to(HERE / "source_v1")
             and sha(path) == row["wav_sha256"], "source WAV hash changed")
        with wave.open(str(path), "rb") as clip:
            need((clip.getnchannels(), clip.getsampwidth(), clip.getframerate(),
                  clip.getnframes()) == (1, 2, 16000, row["frames"]),
                 "source WAV geometry changed")
    return plan


def reviewed_listening(bundle: Path, third: Path, a: Path, b: Path,
                       c: Path | None, receipt: Path) -> dict:
    """Recompute the human intake from actual exports and compare final bytes."""
    for path in (a, b, receipt):
        _regular(path)
    if c is not None:
        _regular(c)
    _regular(third / "ADJUDICATION_MAPPING.json")
    observed = json.loads(receipt.read_text())
    expected = intake.finalize(bundle, third, a, b, c)
    need(observed == expected and observed.get("status") == "REVIEWED"
         and observed.get("denominator") == 16
         and len(observed.get("cases", [])) == 16,
         "human review receipt does not match validated exports")
    return {"receipt_sha256": sha(receipt),
            "review_a_sha256": sha(a), "review_b_sha256": sha(b),
            "review_c_sha256": sha(c) if c is not None else None,
            "adjudication_sha256": sha(third / "ADJUDICATION_MAPPING.json"),
            "passed_clips": observed["passed"], "clips": 16,
            "disagreements": observed["disagreements"]}


def readiness(*, third: Path | None = None, a: Path | None = None,
              b: Path | None = None, c: Path | None = None,
              receipt: Path | None = None) -> dict:
    plan = frozen_plan()
    # Reuse the already pinned no-weight asset check. It hashes model bytes but
    # does not deserialize them or call the target model.
    from models.second_architecture.nonspeech_probe_v1 import descriptive_worker_v1
    assets = descriptive_worker_v1.verify_local_assets("whisper_multilingual")
    installed = descriptive_worker_v1.verify_installed_sources()
    need(assets["repo"] == plan["target"]["repo"]
         and assets["revision"] == plan["target"]["revision"]
         and assets["files_sha256"]["model.safetensors"] ==
             plan["target"]["weight_sha256"], "target asset identity changed")
    result = {"schema": "whisper_human_speech_preflight_v1",
              "status": "HOLD_NO_TARGET_INFERENCE", "plan_sha256": PLAN_SHA256,
              "target": plan["target"], "local_assets": assets,
              "installed": installed,
              "development_calls": sum(r["split"] == "development" for r in plan["calls"]),
              "sealed_calls": sum(r["split"] == "sealed_test" for r in plan["calls"])}
    if any(path is None for path in (third, a, b, receipt)):
        result["reason"] = "two blinded review exports and final intake receipt absent"
        return result
    result["listening"] = reviewed_listening(HERE / "blind_bundle_v1", third, a, b, c, receipt)
    result["reason"] = "executable runner, independent source review and separate GO remain required"
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--third", type=Path)
    parser.add_argument("--review-a", type=Path)
    parser.add_argument("--review-b", type=Path)
    parser.add_argument("--review-c", type=Path)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    print(json.dumps(readiness(third=args.third, a=args.review_a, b=args.review_b,
                               c=args.review_c, receipt=args.receipt), sort_keys=True))


if __name__ == "__main__":
    main()
