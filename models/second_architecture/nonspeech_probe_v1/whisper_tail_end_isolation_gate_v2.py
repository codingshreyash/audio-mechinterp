"""Source-only admission and one-attempt GO for Whisper tail/end V2.

Importing this module never loads candidate weights. A separate independent
review and committed GO are required before the parent or worker may proceed.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from . import natural_tail_runner_v4 as v4
from . import whisper_tail_end_isolation_plan_v2 as plan_source
from . import whisper_time_translation_source_v1 as pinned_source

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RUNS = HERE / "runs/whisper_tail_end_isolation_v2"
STAGES = ("original", "offset_8000", "offset_16000")
ATTEMPTS = {stage: RUNS / f"{stage}_attempt_001" for stage in STAGES}
ADMISSIONS = {stage: HERE / f"WHISPER_TAIL_END_ISOLATION_{stage.upper()}_ADMISSION_V2.json"
              for stage in STAGES}
GOS = {stage: HERE / f"WHISPER_TAIL_END_ISOLATION_{stage.upper()}_GO_V2.json"
       for stage in STAGES}
REVIEW = HERE / "WHISPER_TAIL_END_ISOLATION_EXECUTABLE_REVIEW_V2.md"
PLAN = plan_source.PLAN
SOURCE_MANIFEST = pinned_source.OUTPUT / "manifest.json"
SOURCE_NAMES = (
    "whisper_tail_end_isolation_gate_v2.py",
    "whisper_tail_end_isolation_parent_v2.py",
    "whisper_tail_end_isolation_worker_v2.py",
    "whisper_tail_end_isolation_saved_audit_v2.py",
    "whisper_tail_end_isolation_plan_v2.py",
    "test_whisper_tail_end_isolation_gate_v2.py",
    "test_whisper_tail_end_isolation_parent_v2.py",
    "test_whisper_tail_end_isolation_worker_v2.py",
    "test_whisper_tail_end_isolation_saved_audit_v2.py",
    "test_whisper_tail_end_isolation_plan_v2.py",
    "WHISPER_TAIL_END_ISOLATION_PROSPECTIVE_V2.md",
    "WHISPER_TAIL_END_ISOLATION_EXECUTABLE_HANDOFF_V2.md",
    "WHISPER_TAIL_END_ISOLATION_PLAN_V2.json",
    "whisper_time_translation_gate_v1.py",
    "whisper_time_translation_parent_v1.py",
    "whisper_time_translation_worker_v1.py",
    "whisper_time_translation_saved_audit_v1.py",
    "whisper_time_translation_plan_v1.py",
    "whisper_time_translation_source_v1.py",
    "whisper_time_translation_independent_preprocessing_check_v1.py",
    "test_whisper_time_translation_gate_v1.py",
    "test_whisper_time_translation_parent_v1.py",
    "test_whisper_time_translation_worker_v1.py",
    "test_whisper_time_translation_saved_audit_v1.py",
    "test_whisper_time_translation_plan_v1.py",
    "test_whisper_time_translation_source_v1.py",
    "WHISPER_TIME_TRANSLATION_PROSPECTIVE_V1.md",
    "WHISPER_TIME_TRANSLATION_EXECUTABLE_HANDOFF_V1.md",
    "WHISPER_TIME_TRANSLATION_PLAN_V1.json",
    "runs/whisper_time_translation_v1/source_v1/manifest.json",
    "whisper_tail_causal_host_v1.py",
    "whisper_tail_causal_worker_v3.py",
    "whisper_tail_causal_saved_audit_v3.py",
    "whisper_tail_causal_parent_v3.py",
    "whisper_tail_causal_plan_v1.py",
    "whisper_tail_causal_plan_v3.py",
    "natural_tail_runner_v4.py",
    "causal_parent_v2.py",
)
WALL_SECONDS = 1800
RSS_BYTES = 4 * 1024**3
EVIDENCE_BYTES = 1024**3


def need(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1 << 20), b""):
            digest.update(part)
    return digest.hexdigest()


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(("git", *args), cwd=ROOT, text=True,
                          capture_output=True, check=True, timeout=20)


def _tracked_clean(path: Path) -> None:
    need(path.is_file() and not path.is_symlink(), "missing or linked source")
    relative = str(path.relative_to(ROOT))
    need(subprocess.run(("git", "ls-files", "--error-unmatch", "--", relative),
                        cwd=ROOT, capture_output=True).returncode == 0,
         f"untracked source: {relative}")
    for flags in ((), ("--cached",)):
        need(subprocess.run(("git", "diff", *flags, "--quiet", "--", relative),
                            cwd=ROOT, capture_output=True).returncode == 0,
             f"modified source: {relative}")


def source_identity() -> dict[str, str]:
    identity: dict[str, str] = {}
    for name in SOURCE_NAMES:
        path = HERE / name
        _tracked_clean(path)
        identity[name] = sha(path)
    return identity


def _commit(path: Path) -> str:
    return _git("log", "-1", "--format=%H", "--", str(path.relative_to(ROOT))).stdout.strip()


def _original_go_commit(path: Path) -> str:
    commits = _git("log", "--diff-filter=A", "--reverse", "--format=%H", "--",
                   str(path.relative_to(ROOT))).stdout.splitlines()
    need(len(commits) == 1, "GO must have exactly one original creation commit")
    return commits[0]


def _sha_at(path: Path, revision: str) -> str:
    data = _git("show", f"{revision}:{path.relative_to(ROOT)}").stdout.encode()
    return hashlib.sha256(data).hexdigest()


def _ancestor(before: str, after: str) -> bool:
    return subprocess.run(("git", "merge-base", "--is-ancestor", before, after),
                          cwd=ROOT, capture_output=True).returncode == 0


def admission(stage: str) -> dict:
    need(stage in STAGES, "unknown isolated tail/end stage")
    frozen = json.loads(PLAN.read_text())
    need(frozen == plan_source.build(), "frozen plan differs from source builder")
    need(frozen["stages"][stage]["max_calls"] == (7 if stage == "original" else 24),
         "stage call cap changed")
    return {"schema": "whisper_tail_end_isolation_admission_v2",
            "status": "source_only_no_weights", "stage": stage,
            "attempt": str(ATTEMPTS[stage]), "plan_sha256": sha(PLAN),
            "source_manifest_sha256": sha(SOURCE_MANIFEST),
            "source_identity": source_identity(),
            "v4_admission": v4.admission("whisper_multilingual"),
            "limits": {"whole_wall_seconds": WALL_SECONDS,
                       "tree_rss_bytes": RSS_BYTES,
                       "evidence_bytes": EVIDENCE_BYTES,
                       "battery_start_percent": 80,
                       "battery_abort_percent": 70,
                       "max_power_probe_gap_seconds": 15}}


def validate_go(stage: str, admission_path: Path, go_path: Path) -> dict:
    need(stage in STAGES and admission_path == ADMISSIONS[stage]
         and go_path == GOS[stage], "fixed stage admission and GO paths required")
    for path in (admission_path, go_path, REVIEW, PLAN, SOURCE_MANIFEST):
        _tracked_clean(path)
    expected = admission(stage)
    need(json.loads(admission_path.read_text()) == expected,
         "committed admission differs from current source")
    go = json.loads(go_path.read_text())
    need(go == {"schema": "whisper_tail_end_isolation_go_v2",
                "status": "independent_executable_go", "stage": stage,
                "attempt": str(ATTEMPTS[stage]),
                "admission_sha256": sha(admission_path),
                "plan_sha256": sha(PLAN),
                "source_identity": expected["source_identity"],
                "review_sha256": sha(REVIEW)}, "GO does not bind reviewed source")
    go_commit = _original_go_commit(go_path)
    need(_sha_at(go_path, go_commit) == sha(go_path)
         and _sha_at(admission_path, go_commit) == sha(admission_path)
         and _sha_at(REVIEW, go_commit) == sha(REVIEW),
         "GO/admission/review differs from original GO tree")
    for name, digest in expected["source_identity"].items():
        need(_sha_at(HERE / name, go_commit) == digest,
             f"GO tree source differs: {name}")
    need(_ancestor(_commit(PLAN), _commit(admission_path))
         and _ancestor(_commit(admission_path), _commit(REVIEW))
         and _ancestor(_commit(REVIEW), go_commit),
         "source, admission, independent review, GO order invalid")
    return expected
