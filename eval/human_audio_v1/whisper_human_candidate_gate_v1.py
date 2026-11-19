"""Source-only admission and fail-closed GO gate for authentic-speech Whisper.

This is a candidate, not a GO. Importing it never deserializes target weights.
The independent reviewer must commit a review after the admission, and a
different, later commit must contain the one-attempt GO.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from . import whisper_human_preflight_v1 as frozen

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ADMISSION = HERE / "WHISPER_HUMAN_CANDIDATE_ADMISSION_V1.json"
REVIEW = HERE / "WHISPER_HUMAN_CANDIDATE_INDEPENDENT_REVIEW_V1.json"
GO = HERE / "WHISPER_HUMAN_CANDIDATE_EXECUTABLE_GO_V1.json"
ATTEMPT = HERE / "runs/whisper_human_development_v1/attempt_001"
SOURCES = (
    "whisper_human_candidate_gate_v1.py",
    "whisper_human_candidate_runner_v1.py",
    "test_whisper_human_candidate_v1.py",
    "whisper_human_preflight_v1.py",
    "WHISPER_HUMAN_SPEECH_MECHANISM_PROSPECTIVE_V1.md",
    "WHISPER_HUMAN_SPEECH_SOURCE_PLAN_V1.json",
    "WHISPER_HUMAN_CANDIDATE_HANDOFF_V1.md",
    "human_review_intake_v2.py",
    "blinded_listening_v1.py",
    "SOURCE_AUDIT_V1.json",
    "SOURCE_INDEPENDENT_TECHNICAL_REVIEW_V1.md",
    "WHISPER_HUMAN_SPEECH_PLAN_INDEPENDENT_SOURCE_REVIEW_V1.md",
)
EXTERNAL_SOURCES = (
    ROOT / "models/second_architecture/nonspeech_probe_v1/descriptive_worker_v1.py",
    ROOT / "models/second_architecture/nonspeech_probe_v1/descriptive_pins_v1.json",
    ROOT / "models/second_architecture/nonspeech_probe_v1/whisper_tiny_metadata_pin.json",
    ROOT / "models/second_architecture/nonspeech_probe_v1/whisper_tail_causal_host_v1.py",
    ROOT / "models/second_architecture/wav2vec2_fit_noise_norm_causal_host_v1.py",
)
LIMITS = {"wall_seconds": 7200, "worker_rss_bytes": 4 * 1024**3,
          "evidence_bytes": 1024**3, "max_model_calls": 106}
DECODE = {"task": "transcribe", "language": None, "do_sample": False,
          "num_beams": 1, "condition_on_prev_tokens": False,
          "return_timestamps": False, "max_new_tokens": 128}
SOURCE_RECEIPT_HASHES = {
    "SOURCE_AUDIT_V1.json": "f8b36f626dc3a813adcba8965f9cc0ada2c3b4406137f8085bee24a686ec0661",
    "SOURCE_INDEPENDENT_TECHNICAL_REVIEW_V1.md": "2ddd3d7d4d1a57f545a4d07572112db17a58c64da9c889d5f136c22e16fb90fe",
    "WHISPER_HUMAN_SPEECH_PLAN_INDEPENDENT_SOURCE_REVIEW_V1.md": "f94f2da35a6a620a535ce24d38b2df34cfae7a502f83598b9b21fc758bccd32e",
}


def need(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def rel(path: Path) -> str:
    need(path.is_relative_to(ROOT), "path outside repository")
    return str(path.relative_to(ROOT))


def git(*args: str) -> bytes:
    result = subprocess.run(("git", *args), cwd=ROOT, capture_output=True)
    need(result.returncode == 0, f"git provenance failed: {' '.join(args)}")
    return result.stdout


def tracked_clean(path: Path) -> None:
    need(path.is_file() and not path.is_symlink(), f"missing or symlink: {path}")
    name = rel(path)
    git("ls-files", "--error-unmatch", "--", name)
    git("diff", "--quiet", "--", name)
    git("diff", "--cached", "--quiet", "--", name)


def source_paths() -> list[Path]:
    return [*(HERE / name for name in SOURCES), *EXTERNAL_SOURCES]


def sources() -> dict[str, str]:
    for path in source_paths():
        tracked_clean(path)
    return {rel(path): sha(path) for path in source_paths()}


def source_receipts() -> dict[str, str]:
    for name, expected in SOURCE_RECEIPT_HASHES.items():
        path = HERE / name
        need(path.is_file() and not path.is_symlink() and sha(path) == expected,
             f"independent source receipt bytes changed: {name}")
    receipt = json.loads((HERE / "SOURCE_AUDIT_V1.json").read_text())
    need(receipt.get("schema") == "human_audio_source_audit_v1"
         and receipt.get("status") == "PASS"
         and receipt.get("wavs") == 16
         and receipt.get("source_recordings") == 4
         and receipt.get("manifest_sha256") == sha(frozen.MANIFEST),
         "source audit PASS/census or manifest changed")
    technical = (HERE / "SOURCE_INDEPENDENT_TECHNICAL_REVIEW_V1.md").read_text()
    review = (HERE / "WHISPER_HUMAN_SPEECH_PLAN_INDEPENDENT_SOURCE_REVIEW_V1.md").read_text()
    need("**PASS for source provenance" in technical
         and "**GO for the frozen source-only plan" in review
         and "HOLD target Whisper inference" in review,
         "independent source review status changed")
    return dict(SOURCE_RECEIPT_HASHES)


def admission(*, third: Path, a: Path, b: Path, c: Path | None,
              receipt: Path) -> dict:
    """Build only after real listening exports pass the existing V2 intake."""
    identity = sources()
    receipt_hashes = source_receipts()
    preflight = frozen.readiness(third=third, a=a, b=b, c=c, receipt=receipt)
    need(preflight["status"] == "HOLD_NO_TARGET_INFERENCE"
         and "listening" in preflight, "reviewed listening unavailable")
    return {
        "schema": "whisper_human_candidate_admission_v1",
        "status": "source_only_pending_independent_review_and_go",
        "split": "development", "sealed_test_inference": False,
        "source_identity": identity,
        "plan_sha256": frozen.PLAN_SHA256,
        "wav_sha256": {row["wav"]: row["wav_sha256"] for row in frozen.frozen_plan()["calls"]},
        "proxy_audit_sha256": sha(frozen.PROXY_AUDIT),
        "proxy_review_sha256": sha(frozen.PROXY_REVIEW),
        "source_receipt_sha256": receipt_hashes,
        "human_review": {"third_path": str(third.resolve()),
                         "a_path": str(a.resolve()), "b_path": str(b.resolve()),
                         "c_path": str(c.resolve()) if c else None,
                         "receipt_path": str(receipt.resolve()),
                         **preflight["listening"]},
        "local_assets": preflight["local_assets"],
        "installed": preflight["installed"],
        "attempt_path": str(ATTEMPT), "limits": LIMITS, "decode": DECODE,
        "call_plan": {"descriptive": 8, "base_replays": 2,
                      "conditional_replays_per_eligible_pair": 4,
                      "causal_cells_per_eligible_pair": 20},
    }


def _last_commit(path: Path, asof: str) -> str:
    result = git("log", "-1", "--format=%H", asof, "--", rel(path)).decode().strip()
    need(len(result) == 40, f"missing committed provenance: {path.name}")
    return result


def _ancestor(before: str, after: str) -> bool:
    return subprocess.run(("git", "merge-base", "--is-ancestor", before, after),
                          cwd=ROOT, capture_output=True).returncode == 0


def validate_go(*, parent_owned_attempt: bool = False) -> dict:
    """Require reviewed source/admission and an exact later committed GO."""
    for path in (ADMISSION, REVIEW, GO, *source_paths()):
        tracked_clean(path)
    stored = json.loads(ADMISSION.read_text())
    human = stored.get("human_review", {})
    expected = admission(third=Path(human["third_path"]), a=Path(human["a_path"]),
                         b=Path(human["b_path"]),
                         c=Path(human["c_path"]) if human["c_path"] else None,
                         receipt=Path(human["receipt_path"]))
    need(stored == expected, "admission differs from current source or human review")
    review = json.loads(REVIEW.read_text())
    need(review == {"schema": "whisper_human_candidate_independent_review_v1",
                    "status": "PASS_SOURCE_ONLY", "admission_sha256": sha(ADMISSION),
                    "source_identity": stored["source_identity"],
                    "split": "development"},
         "independent review receipt invalid")
    go = json.loads(GO.read_text())
    need(go == {"schema": "whisper_human_candidate_executable_go_v1",
                "status": "GO", "admission_sha256": sha(ADMISSION),
                "review_sha256": sha(REVIEW),
                "source_identity": stored["source_identity"],
                "split": "development", "attempt_path": str(ATTEMPT),
                "max_model_calls": LIMITS["max_model_calls"]},
         "executable GO fields invalid")
    go_commit = _last_commit(GO, "HEAD")
    for path in (ADMISSION, REVIEW, GO, *source_paths()):
        need(hashlib.sha256(git("show", f"{go_commit}:{rel(path)}")).hexdigest()
             == sha(path), f"GO-time/current bytes differ: {path.name}")
    admission_commit = _last_commit(ADMISSION, go_commit)
    review_commit = _last_commit(REVIEW, go_commit)
    need(admission_commit != review_commit != go_commit
         and _ancestor(admission_commit, review_commit)
         and _ancestor(review_commit, go_commit),
         "source admission, independent review and GO need separate ordered commits")
    for path in source_paths():
        prior = _last_commit(path, go_commit)
        need(prior != admission_commit and _ancestor(prior, admission_commit),
             f"source must precede admission: {path.name}")
    if parent_owned_attempt:
        need(ATTEMPT.is_dir() and not ATTEMPT.is_symlink()
             and (ATTEMPT / "parent_launch.json").is_file(),
             "parent-owned attempt missing")
    else:
        need(not ATTEMPT.exists() and not ATTEMPT.is_symlink(),
             "one-shot attempt already exists")
    return {"admission": stored, "go_sha256": sha(GO), "go_commit": go_commit}
