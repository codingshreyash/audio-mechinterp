"""Content-bound, independently reviewed GO gate for the ASR proxy."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from . import asr_proxy_parent_v1 as old_parent
from . import asr_proxy_worker_v1 as worker

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ADMISSION_V2 = HERE / "ASR_PROXY_ADMISSION_V2.json"
REVIEW = HERE / "ASR_PROXY_EXECUTABLE_REVIEW_V2.md"
GO = HERE / "ASR_PROXY_EXECUTABLE_GO_V2.json"
SOURCES = ("ASR_PROXY_EXECUTABLE_AMENDMENT_V2.md", "asr_proxy_gate_v2.py",
           "asr_proxy_parent_v2.py", "asr_proxy_worker_v2.py",
           "asr_proxy_saved_audit_v2.py",
           "test_asr_proxy_gate_v2.py")
HOST_SOURCE = Path(old_parent.host.__file__).resolve()


def source_paths() -> list[Path]:
    return [*(HERE / name for name in SOURCES), HOST_SOURCE]


def need(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha(path: Path) -> str:
    return sha_bytes(path.read_bytes())


def relative(path: Path) -> str:
    need(path.is_relative_to(ROOT), "path outside repository")
    return str(path.relative_to(ROOT))


def tracked_clean(path: Path) -> None:
    need(path.is_file() and not path.is_symlink(), f"file missing: {path.name}")
    rel = relative(path)
    need(subprocess.run(("git", "ls-files", "--error-unmatch", "--", rel),
                        cwd=ROOT, capture_output=True).returncode == 0
         and subprocess.run(("git", "diff", "--quiet", "--", rel),
                            cwd=ROOT, capture_output=True).returncode == 0
         and subprocess.run(("git", "diff", "--cached", "--quiet", "--", rel),
                            cwd=ROOT, capture_output=True).returncode == 0,
         f"file untracked or modified: {rel}")


def source_identity() -> dict[str, str]:
    return {relative(path): sha(path) for path in source_paths()}


def admission() -> dict:
    v1 = old_parent.validate_admission()  # no model deserialization
    need(v1["status"] == "source_only_no_model_inference"
         and len(v1["plan"]) == 16, "V1 source admission changed")
    return {"schema": "human_asr_proxy_admission_v2",
            "status": "source_only_no_model_inference",
            "model": v1["model"], "model_revision": v1["model_revision"],
            "attempt": str(worker.ATTEMPT),
            "admission_v1_sha256": sha(worker.ADMISSION),
            "source_identity": source_identity(),
            "limits": v1["limits"], "calls": 16}


def _git_output(argv: tuple[str, ...]) -> bytes:
    result = subprocess.run(("git", *argv), cwd=ROOT, capture_output=True)
    need(result.returncode == 0, f"git provenance failed: {argv[0]}")
    return result.stdout


def _last_asof(path: Path, commit: str) -> str:
    value = _git_output(("log", "-1", "--format=%H", commit, "--", relative(path))).decode().strip()
    need(bool(value), f"no GO-time commit: {path.name}")
    return value


def _ancestor(before: str, after: str) -> bool:
    return subprocess.run(("git", "merge-base", "--is-ancestor", before, after),
                          cwd=ROOT, capture_output=True).returncode == 0


def validate_go() -> dict:
    for path in [worker.ADMISSION, ADMISSION_V2, REVIEW, GO, *source_paths()]:
        tracked_clean(path)
    expected = admission()
    need(json.loads(ADMISSION_V2.read_text()) == expected, "V2 admission changed")
    go = json.loads(GO.read_text())
    need(go.get("schema") == "human_asr_proxy_executable_go_v2"
         and go.get("status") == "GO"
         and go.get("model") == expected["model"]
         and go.get("model_revision") == expected["model_revision"]
         and go.get("attempt") == str(worker.ATTEMPT)
         and go.get("calls") == 16
         and go.get("admission_v1_sha256") == sha(worker.ADMISSION)
         and go.get("admission_v2_sha256") == sha(ADMISSION_V2)
         and go.get("source_identity") == expected["source_identity"]
         and go.get("review_sha256") == sha(REVIEW),
         "independent executable GO fields invalid")
    go_commit = _git_output(("log", "-1", "--format=%H", "--", relative(GO))).decode().strip()
    need(bool(go_commit), "GO commit missing")
    paths = [worker.ADMISSION, ADMISSION_V2, REVIEW, GO, *source_paths()]
    for path in paths:
        snapshot = _git_output(("show", f"{go_commit}:{relative(path)}"))
        need(sha_bytes(snapshot) == sha(path), f"GO-time/current bytes differ: {path.name}")
    review_commit = _last_asof(REVIEW, go_commit)
    need(review_commit != go_commit and _ancestor(review_commit, go_commit),
         "review must precede GO")
    admit_commit = _last_asof(ADMISSION_V2, go_commit)
    need(admit_commit != review_commit and _ancestor(admit_commit, review_commit),
         "V2 admission must precede independent review")
    for path in [worker.ADMISSION, *source_paths()]:
        prior = _last_asof(path, go_commit)
        need(prior != admit_commit and _ancestor(prior, admit_commit),
             f"source/V1 admission must precede V2 admission: {path.name}")
    need(_last_asof(GO, go_commit) == go_commit, "GO commit identity invalid")
    return {"schema": "human_asr_proxy_go_validation_v2", "status": "GO_VALID",
            "go_sha256": sha(GO), "go_commit": go_commit,
            "admission_v2_sha256": sha(ADMISSION_V2),
            "admission_v1_sha256": sha(worker.ADMISSION),
            "source_identity": expected["source_identity"],
            "attempt": str(worker.ATTEMPT), "model_revision": expected["model_revision"]}
