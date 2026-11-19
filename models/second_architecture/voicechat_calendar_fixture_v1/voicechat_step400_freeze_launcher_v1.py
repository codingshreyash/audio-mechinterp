"""Fresh, externally reviewed H200 admission for the step-400 diagnostic.

This module is source-only until an independent reviewer signs a host freeze.
It never imports VoiceChat or loads weights itself. A successful invocation
permits one child under the bounded parent; it does not assert a model result.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import hmac
import json
import os
import re
import shutil
import stat
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from models.second_architecture import voicechat_h100_local_preflight as pinned
from models.second_architecture.voicechat_calendar_fixture_v1 import public_sample_smoke_v1 as sample
from models.second_architecture.voicechat_calendar_fixture_v1 import voicechat_step400_supervisor_v1 as parent
from models.second_architecture.voicechat_calendar_fixture_v1 import voicechat_step400_worker_v1 as worker

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
SCHEMA = "voicechat_step400_external_freeze_v1"
REVIEW_SCHEMA = "voicechat_step400_external_freeze_review_v1"
KEY_ENV = "VOICECHAT_STEP400_REVIEW_KEY_HEX"
MAX_FREEZE_BYTES = 262_144
MAX_REVIEW_BYTES = 16_384
SOURCE_NAMES = (
    "voicechat_step400_freeze_launcher_v1.py",
    "voicechat_step400_worker_v1.py",
    "voicechat_step400_probe_core_v1.py",
    "voicechat_step400_supervisor_v1.py",
    "voicechat_step400_saved_auditor_v1.py",
    "test_voicechat_step400_freeze_launcher_v1.py",
    "test_voicechat_step400_worker_v1.py",
    "test_voicechat_step400_probe_core_v1.py",
    "test_voicechat_step400_supervisor_v1.py",
    "test_voicechat_step400_saved_auditor_v1.py",
)
CAPS = {"wall_seconds": 1_800, "gpu_mib": 130_000,
        "evidence_bytes": 64 * 1024**2, "post_staging_disk_bytes": 30 * 1024**3,
        "checkpoint_steps": [1, 50, 100, 200, 300, 400]}


def need(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       allow_nan=False) + "\n").encode()


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_json(path: Path, cap: int) -> tuple[dict[str, Any], bytes]:
    need(type(cap) is int and 0 < cap <= MAX_FREEZE_BYTES, "invalid read cap")
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError as error:
        raise ValueError("review/freeze not a readable regular file") from error
    try:
        before = os.fstat(fd)
        need(stat.S_ISREG(before.st_mode) and 0 < before.st_size <= cap,
             "missing or oversized review/freeze")
        chunks: list[bytes] = []
        remaining = cap + 1
        while remaining:
            block = os.read(fd, remaining)
            if not block:
                break
            chunks.append(block)
            remaining -= len(block)
        raw = b"".join(chunks)
        after = os.fstat(fd)
        named = path.lstat()
        need(len(raw) <= cap and len(raw) == before.st_size == after.st_size
             and before.st_dev == after.st_dev == named.st_dev
             and before.st_ino == after.st_ino == named.st_ino
             and before.st_mtime_ns == after.st_mtime_ns
             and before.st_ctime_ns == after.st_ctime_ns,
             "review/freeze changed during bounded read")
    finally:
        os.close(fd)
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result = dict(pairs)
        need(len(result) == len(pairs), "duplicate JSON field")
        return result
    value = json.loads(raw, object_pairs_hook=unique)
    need(type(value) is dict and canonical(value) == raw, "noncanonical review/freeze")
    return value, raw


def check_paths(*, nemo_dir: Path, checkpoint_dir: Path, nano_home: Path,
                output_dir: Path, freeze_path: Path, review_path: Path) -> None:
    paths = (nemo_dir, checkpoint_dir, nano_home, output_dir, freeze_path, review_path)
    need(all(path.is_absolute() and path == path.resolve() for path in paths),
         "all paths must be absolute and resolved")
    need(nemo_dir.is_dir() and checkpoint_dir.is_dir() and nano_home.is_dir(),
         "staged inputs missing")
    need(REPO_ROOT.is_dir() and not REPO_ROOT.is_symlink(), "project source missing")
    need(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{3,63}", output_dir.name) is not None,
         "invalid attempt ID")
    need(not output_dir.exists() and not output_dir.is_symlink(),
         "attempt already consumed")
    need(output_dir.parent.is_dir() and not output_dir.parent.is_symlink(),
         "attempt parent missing or linked")
    claim = claim_path(output_dir)
    need(not claim.exists() and not claim.is_symlink(),
         "one-attempt GO already consumed")
    need(freeze_path != review_path and freeze_path != output_dir
         and review_path != output_dir and freeze_path != claim
         and review_path != claim
         and freeze_path.parent.is_dir() and review_path.parent.is_dir(),
         "external admission paths missing or collide")
    need(freeze_path.parent != output_dir and review_path.parent != output_dir,
         "admission must be external to attempt")
    need(shutil.disk_usage(checkpoint_dir).free >= CAPS["post_staging_disk_bytes"],
         "less than 30 GiB free after staging")


def claim_path(output_dir: Path) -> Path:
    return output_dir.parent / (output_dir.name + ".step400-go-consumed")


def consume_go(output_dir: Path, freeze_raw: bytes, review_raw: bytes) -> Path:
    """Atomically reserve this attempt outside its removable evidence tree."""
    path = claim_path(output_dir)
    receipt = {"schema": "voicechat_step400_go_consumed_v1",
               "attempt_id": output_dir.name, "output_dir": str(output_dir),
               "freeze_sha256": digest(freeze_raw),
               "review_sha256": digest(review_raw), "status": "consumed_before_spawn"}
    data = canonical(receipt)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        written = 0
        while written < len(data):
            count = os.write(fd, data[written:])
            need(count > 0, "GO claim short write")
            written += count
        os.fsync(fd)
    finally:
        os.close(fd)
    parent_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)
    return path


def project_source() -> dict[str, Any]:
    """Hash executable local source and reject ignored Python bytecode nearby."""
    need(Path.cwd().resolve() == REPO_ROOT, "launcher must run from source root")
    files = {name: sample.hash_file(HERE / name, cap=1 << 20)[0]
             for name in SOURCE_NAMES}
    for package_dir in (REPO_ROOT / "models", REPO_ROOT / "models/second_architecture", HERE):
        need(not (package_dir / "__init__.py").exists()
             and not (package_dir / "__init__.py").is_symlink(),
             "unreviewed package initializer in project import path")
        need(not any(package_dir.glob("*.pyc"))
             and not (package_dir / "__pycache__").exists(),
             "ignored local bytecode present in executable package path")
    head = subprocess.run(("git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"),
                          check=True, capture_output=True, text=True, timeout=10).stdout.strip()
    need(re.fullmatch(r"[0-9a-f]{40}", head) is not None, "project commit unavailable")
    return {"project_root": str(REPO_ROOT), "project_head": head,
            "local_source_sha256": files}


def command(nemo_dir: Path, checkpoint_dir: Path, output_dir: Path) -> list[str]:
    # Preseed the three absent project packages with exact paths. This prevents
    # an installed regular `models` package from shadowing our namespace tree.
    # Remove the child CWD from sys.path before importing any project module.
    bootstrap = ("import sys\n"
                 "sys.path[:] = [p for p in sys.path if p]\n"
                 "assert sys.path[0] == " + repr(str(nemo_dir)) + "\n"
                 "sys.path.insert(1, " + repr(str(REPO_ROOT)) + ")\n"
                 "import types, importlib.machinery\n"
                 "for name, path in " + repr([
                     ("models", str(REPO_ROOT / "models")),
                     ("models.second_architecture", str(REPO_ROOT / "models/second_architecture")),
                     ("models.second_architecture.voicechat_calendar_fixture_v1", str(HERE)),
                 ]) + ":\n"
                 "    assert name not in sys.modules\n"
                 "    module = types.ModuleType(name)\n"
                 "    module.__path__ = [path]\n"
                 "    module.__package__ = name\n"
                 "    module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)\n"
                 "    sys.modules[name] = module\n"
                 "from models.second_architecture.voicechat_calendar_fixture_v1."
                 "voicechat_step400_worker_v1 import main\n"
                 "main()\n")
    return [sys.executable, "-B", "-c", bootstrap,
            "--nemo-dir", str(nemo_dir), "--checkpoint-dir", str(checkpoint_dir),
            "--output-dir", str(output_dir)]


def _official_defaults(nemo_dir: Path) -> tuple[str, list[dict], str]:
    """Extract literal defaults without importing the official model example."""
    tree = ast.parse((nemo_dir / sample.SAMPLE_SCRIPT).read_text())
    found: dict[str, Any] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in {
                    "DEFAULT_SYSTEM_MESSAGE", "DEFAULT_TOOLS"}:
                    found[target.id] = ast.literal_eval(node.value)
    message, tools = found["DEFAULT_SYSTEM_MESSAGE"], found["DEFAULT_TOOLS"]
    need(type(message) is str and type(tools) is list and len(tools) == 1
         and tools[0]["function"]["name"] == "generate_random_number",
         "official prompt defaults changed")
    from jinja2 import Environment
    template = (nemo_dir / "examples/speechlm2/function_calling/template.jinja").read_text()
    rendered = Environment().from_string(template).render(system_message=message, tools=tools)
    return message, tools, rendered


def worker_identity(nemo_dir: Path) -> dict[str, Any]:
    message, tools, rendered = _official_defaults(nemo_dir)
    return {"nemo_revision": pinned.NEMO_COMMIT,
            "voicechat_revision": pinned.VOICECHAT_COMMIT,
            "nano_revision": pinned.NANO_COMMIT,
            "source_sha256": pinned.SOURCE_SHA256,
            "sample_sha256": sample.SAMPLE_SHA256,
            "sample_bytes": sample.SAMPLE_BYTES,
            "config_sha256": pinned.CONFIG_SHA256,
            "weight_sha256": pinned.WEIGHT_SHA256,
            "weight_bytes": pinned.WEIGHT_BYTES,
            "template_sha256": pinned.SOURCE_SHA256[
                "examples/speechlm2/function_calling/template.jinja"],
            "system_message_sha256": digest(message.encode()),
            "tools_sha256": digest(worker.canonical({"tools": tools})),
            "prompt_sha256": digest(rendered.encode()),
            "sample_id": "sample_fc"}


def expected(nemo_dir: Path, checkpoint_dir: Path, nano_home: Path,
             output_dir: Path) -> dict[str, Any]:
    need(os.environ.get("PYTHONPATH") == str(nemo_dir),
         "PYTHONPATH must select pinned NeMo checkout only")
    base = sample.expected_freeze_fields(nemo_dir, checkpoint_dir, nano_home,
                                         output_dir, "h200-141gb")
    runtime = sample.check_h200_sample_runtime(checkpoint_dir)
    need(runtime["gpu_policy"] == "h200-141gb", "H200 host policy changed")
    base["command"] = command(nemo_dir, checkpoint_dir, output_dir)
    base["project"] = project_source()
    base["worker_identity"] = worker_identity(nemo_dir)
    base["caps"] = CAPS
    return base


def stable_phase(receipt: dict[str, Any]) -> dict[str, Any]:
    """Discard timestamps and free-space fluctuation, retaining identities."""
    fields = {name: receipt[name] for name in
              ("phase", "source", "sample_sha256", "nano", "nano_ref_sha256",
               "runtime", "offline", "dynamic")}
    fields["runtime"] = {name: value for name, value in fields["runtime"].items()
                         if name != "disk_free_gib"}
    return fields


def launch_preload(nemo_dir: Path, checkpoint_dir: Path,
                   nano_home: Path) -> dict[str, Any]:
    """Recheck the generated offline Nano cache without requiring it pristine."""
    need(os.environ.get("PYTHONPATH") == str(nemo_dir),
         "PYTHONPATH must select pinned NeMo checkout only")
    receipt = {"phase": "preload",
               "source": pinned.check_source(nemo_dir, checkpoint_dir),
               "sample_sha256": sample.check_sample(nemo_dir),
               "nano": pinned.check_nano_cache(nano_home),
               "nano_ref_sha256": sample.check_exact_nano_ref(nano_home),
               "runtime": sample.check_h200_sample_runtime(checkpoint_dir),
               "offline": pinned.check_preload_environment(nano_home),
               "dynamic": pinned.check_generated_modules_cache(nano_home)}
    return stable_phase(receipt)


def build(*, nemo_dir: Path, checkpoint_dir: Path, nano_home: Path,
          output_dir: Path, freeze_path: Path, review_path: Path) -> dict[str, Any]:
    # Build has no reason to possess a review key. V5's pip/Nano preflight
    # subprocesses inherit os.environ, so fail before invoking either one.
    need(KEY_ENV not in os.environ,
         "external review key must be absent during freeze build")
    check_paths(nemo_dir=nemo_dir, checkpoint_dir=checkpoint_dir,
                nano_home=nano_home, output_dir=output_dir,
                freeze_path=freeze_path, review_path=review_path)
    need(not freeze_path.exists() and not freeze_path.is_symlink()
         and not review_path.exists() and not review_path.is_symlink(),
         "freeze or review path consumed")
    preflight = sample.sample_ready_preload(nemo_dir, checkpoint_dir,
                                           nano_home, "h200-141gb")
    need(preflight["status"] == "sample_only_ready_preload_passed_no_model_work",
         "ready/preload source check failed")
    record = {"schema": SCHEMA, "status": "unsigned_source_and_host_freeze",
              "attempt_id": output_dir.name,
              "expected": expected(nemo_dir, checkpoint_dir, nano_home, output_dir),
              "ready_preload": [stable_phase(row) for row in preflight["receipts"]]}
    raw = canonical(record)
    need(len(raw) <= MAX_FREEZE_BYTES, "freeze exceeds size cap")
    with freeze_path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return {"freeze_sha256": digest(raw), "freeze_bytes": len(raw),
            "status": "unsigned_requires_independent_review"}


def verify_review(freeze: dict[str, Any], raw: bytes, review: dict[str, Any],
                  key: bytes) -> None:
    need(len(key) >= 32, "external review key missing or short")
    required = {"schema", "status", "freeze_sha256", "project_source_sha256",
                "reviewer_id", "reviewed_utc", "hmac_sha256"}
    need(set(review) == required and review["schema"] == REVIEW_SCHEMA
         and review["status"] == "independent_one_attempt_go"
         and review["freeze_sha256"] == digest(raw)
         and review["project_source_sha256"] == digest(canonical(
             freeze["expected"]["project"]["local_source_sha256"])),
         "independent review scope changed")
    need(type(review["reviewer_id"]) is str and review["reviewer_id"].strip()
         and type(review["reviewed_utc"]) is str, "reviewer identity missing")
    try:
        moment = datetime.fromisoformat(review["reviewed_utc"].replace("Z", "+00:00"))
        need(moment.tzinfo is not None, "review timestamp lacks timezone")
    except ValueError as error:
        raise ValueError("review timestamp invalid") from error
    unsigned = {name: value for name, value in review.items() if name != "hmac_sha256"}
    actual = hmac.new(key, canonical(unsigned), hashlib.sha256).hexdigest()
    need(type(review["hmac_sha256"]) is str
         and hmac.compare_digest(review["hmac_sha256"], actual),
         "external review signature invalid")


def launch(*, nemo_dir: Path, checkpoint_dir: Path, nano_home: Path,
           output_dir: Path, freeze_path: Path, review_path: Path,
           review_key: bytes) -> dict[str, Any]:
    # Keep the HMAC key out of V5's pip-freeze and Nano resolver subprocesses.
    # It is never restored: a single launch process owns one review.
    os.environ.pop(KEY_ENV, None)
    check_paths(nemo_dir=nemo_dir, checkpoint_dir=checkpoint_dir,
                nano_home=nano_home, output_dir=output_dir,
                freeze_path=freeze_path, review_path=review_path)
    freeze, raw = read_json(freeze_path, MAX_FREEZE_BYTES)
    review, review_raw = read_json(review_path, MAX_REVIEW_BYTES)
    need(set(freeze) == {"schema", "status", "attempt_id", "expected", "ready_preload"}
         and freeze["schema"] == SCHEMA
         and freeze["status"] == "unsigned_source_and_host_freeze"
         and freeze["attempt_id"] == output_dir.name,
         "freeze scope changed")
    verify_review(freeze, raw, review, review_key)
    current = expected(nemo_dir, checkpoint_dir, nano_home, output_dir)
    need(freeze["expected"] == current, "host/source freeze changed")
    need(type(freeze["ready_preload"]) is list
         and len(freeze["ready_preload"]) == 2
         and freeze["ready_preload"][0]["phase"] == "ready"
         and freeze["ready_preload"][1]["phase"] == "preload"
         and freeze["ready_preload"][1] == launch_preload(
             nemo_dir, checkpoint_dir, nano_home),
         "Nano/source/runtime preload identity changed")
    child_env = dict(current["child_environment"])
    need(KEY_ENV not in child_env and child_env.get("PYTHONPATH") == str(nemo_dir),
         "review key or unpinned Python path reached child")
    consume_go(output_dir, raw, review_raw)
    return parent.supervise_candidate(
        command=current["command"], environment=child_env,
        output_dir=output_dir, staged_checkpoint_dir=checkpoint_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("build", "launch"))
    for name in ("nemo-dir", "checkpoint-dir", "nano-home", "output-dir",
                 "freeze-path", "review-path"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    kwargs = {"nemo_dir": args.nemo_dir, "checkpoint_dir": args.checkpoint_dir,
              "nano_home": args.nano_home, "output_dir": args.output_dir,
              "freeze_path": args.freeze_path, "review_path": args.review_path}
    if args.mode == "build":
        result = build(**kwargs)
    else:
        raw_key = os.environ.pop(KEY_ENV, "")
        need(re.fullmatch(r"[0-9a-fA-F]{64,}", raw_key) is not None
             and len(raw_key) % 2 == 0, "external review key absent")
        result = launch(**kwargs, review_key=bytes.fromhex(raw_key))
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
