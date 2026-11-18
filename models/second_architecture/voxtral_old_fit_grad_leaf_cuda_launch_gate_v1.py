"""Fail-closed, two-attempt CUDA launcher for the prospective old-FIT study.

Source only: missing independent reviews and a later external freeze mean HOLD.
This module imports neither Torch nor the CUDA worker before admission.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RUNS = HERE / "runs/voxtral_old_fit_grad_leaf_cuda_launch_v1"
WORKER_RUNS = HERE / "runs/voxtral_old_fit_grad_leaf_cuda_v1"
RUNNER = HERE / "voxtral_old_fit_grad_leaf_cuda_runner_v1.py"
RUNNER_TESTS = HERE / "test_voxtral_old_fit_grad_leaf_cuda_runner_v1.py"
AUDITOR = HERE / "voxtral_old_fit_grad_leaf_cuda_audit_v1.py"
AUDITOR_TESTS = HERE / "test_voxtral_old_fit_grad_leaf_cuda_audit_v1.py"
GATE = Path(__file__).resolve()
GATE_TESTS = HERE / "test_voxtral_old_fit_grad_leaf_cuda_launch_gate_v1.py"
PROTOCOL_V1 = HERE / "VOXTRAL_OLD_FIT_GRAD_LEAF_CUDA_REPLICATION_PROSPECTIVE_V1.md"
PROTOCOL_V2 = HERE / "VOXTRAL_OLD_FIT_GRAD_LEAF_CUDA_REPLICATION_PROSPECTIVE_V2.md"
RESULT_SCHEMA = HERE / "VOXTRAL_OLD_FIT_GRAD_LEAF_CUDA_RESULT_SCHEMA_V1.md"
ENV_SCHEMA = HERE / "VOXTRAL_OLD_FIT_GRAD_LEAF_CUDA_ENVIRONMENT_RECEIPT_SCHEMA_V1.json"
CONCEPT_REVIEW = HERE / "VOXTRAL_OLD_FIT_GRAD_LEAF_CUDA_REPLICATION_CONCEPTUAL_REREVIEW_V2.md"
PILOT = HERE / "VOXTRAL_OLD_FIT_GRAD_LEAF_CUDA_BYTE_PILOT_V1.json"
PILOT_SOURCE = HERE / "voxtral_old_fit_grad_leaf_cuda_byte_pilot_v1.py"
MPS_MANIFEST = HERE / "VOXTRAL_OLD_FIT_GRAD_LEAF_FACTORIAL_ATTEMPT_MANIFEST_V1.json"
MPS_REVIEW = HERE / "VOXTRAL_OLD_FIT_GRAD_LEAF_FACTORIAL_RESULT_INDEPENDENT_REVIEW_V1.md"
SOURCE_REVIEW = HERE / "VOXTRAL_OLD_FIT_GRAD_LEAF_CUDA_INDEPENDENT_SOURCE_REVIEW_V1.md"
AUDITOR_REVIEW = HERE / "VOXTRAL_OLD_FIT_GRAD_LEAF_CUDA_AUDITOR_INDEPENDENT_SOURCE_REVIEW_V1.md"
ENV_REVIEW = HERE / "VOXTRAL_OLD_FIT_GRAD_LEAF_CUDA_ENVIRONMENT_INDEPENDENT_REVIEW_V1.md"
ENV_RECEIPT = HERE / "VOXTRAL_OLD_FIT_GRAD_LEAF_CUDA_ENVIRONMENT_RECEIPT_V1.json"
FREEZE = HERE / "VOXTRAL_OLD_FIT_GRAD_LEAF_CUDA_EXTERNAL_FREEZE_V1.json"
FREEZE_REVIEW = HERE / "VOXTRAL_OLD_FIT_GRAD_LEAF_CUDA_EXTERNAL_FREEZE_INDEPENDENT_REVIEW_V1.md"
CHECKPOINT = HERE / "hf_cache/voxtral"
CASE_IDS = ("animal_08_kokoro_fit_chosen_beaver", "nature_09_piper_fit_chosen_waterfall")
STATUS = "frozen_voxtral_old_fit_grad_leaf_cuda_v1"
GO_SOURCE = "Decision: GO for CUDA runner and launch gate source only; live model HOLD until external freeze."
GO_AUDITOR = "Decision: GO for CUDA CPU auditor source only; live model HOLD until external freeze."
GO_ENV = "Decision: GO for pinned CUDA environment receipt only; live model HOLD until external freeze."
GO_FREEZE = "Decision: GO for CUDA external freeze provenance; two fixed no-retry attempts may run under the pinned gate."
EXPECTED = {
    "protocol_v1_sha256": "188925bc4022aa9841502d310899b5f9b28bf8ec0ba32ab7f7acb3fc45579050",
    "protocol_v2_sha256": "c86537552b55f6542c88d2be9a7c71d5e1cecd589c6a11cc89eec514c49419ca",
    "result_schema_sha256": "c90f46adcf73a8466232be15ba37d9a1d18b0eca234f2b4648f480bff6b7ad4d",
}
BOUND = {
    "protocol_v1_sha256": PROTOCOL_V1, "protocol_v2_sha256": PROTOCOL_V2,
    "result_schema_sha256": RESULT_SCHEMA, "environment_schema_sha256": ENV_SCHEMA,
    "concept_review_sha256": CONCEPT_REVIEW, "pilot_sha256": PILOT,
    "pilot_source_sha256": PILOT_SOURCE, "mps_manifest_sha256": MPS_MANIFEST,
    "mps_review_sha256": MPS_REVIEW, "runner_sha256": RUNNER,
    "runner_tests_sha256": RUNNER_TESTS, "gate_sha256": GATE,
    "gate_tests_sha256": GATE_TESTS, "auditor_sha256": AUDITOR,
    "auditor_tests_sha256": AUDITOR_TESTS, "environment_receipt_sha256": ENV_RECEIPT,
    "source_review_sha256": SOURCE_REVIEW, "auditor_review_sha256": AUDITOR_REVIEW,
    "environment_review_sha256": ENV_REVIEW,
}
MAX_CUDA = 42 * 1024**3
MAX_RAM = 110 * 1024**3
MAX_WORKER_SECONDS = 600
MAX_AUDIT_SECONDS = 120
MAX_CASE_BYTES = 1024**3
MAX_CAMPAIGN_BYTES = 5 * 1024**3 // 2
MAX_STREAM_BYTES = 8 * 1024**2
MIN_FREE_BYTES = 30 * 1024**3
FINISH_RESERVE = 64 * 1024
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
ATTEMPT = re.compile(r"[0-9]{8}T[0-9]{12}Z\Z")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_bytes(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def put(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def tree_bytes(path: Path) -> int:
    if path.is_symlink():
        raise ValueError(f"Symlink output root: {path}")
    if not path.exists():
        return 0
    total = 0
    for child in path.rglob("*"):
        if child.is_symlink():
            raise ValueError(f"Symlink in output: {child}")
        if child.is_file():
            total += child.stat().st_size
    return total


def git(*args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(REPO), *args], stderr=subprocess.DEVNULL)


def committed_same(path: Path) -> None:
    if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(REPO):
        raise ValueError(f"Missing or unsafe committed file: {path}")
    rel = str(path.relative_to(REPO))
    if git("show", f"HEAD:{rel}") != path.read_bytes():
        raise ValueError(f"Uncommitted or changed file: {path}")


def first_add(path: Path) -> str:
    rel = str(path.relative_to(REPO))
    commits = git("log", "--diff-filter=A", "--format=%H", "--", rel).decode().splitlines()
    if len(commits) != 1:
        raise ValueError(f"Missing/ambiguous commit for {path}")
    return commits[0]


def ancestor(before: str, after: str) -> None:
    if before == after or subprocess.run(["git", "-C", str(REPO), "merge-base", "--is-ancestor", before, after],
                                         check=False, stdout=subprocess.DEVNULL,
                                         stderr=subprocess.DEVNULL).returncode != 0:
        raise ValueError("Required independent commit chronology absent")


def reviewed_source(paths: tuple[Path, ...], review: Path, phrase: str) -> str:
    committed_same(review)
    require_decision(review, phrase)
    commit = first_add(review)
    for path in paths:
        committed_same(path)
        rel = str(path.relative_to(REPO))
        if git("show", f"{commit}:{rel}") != path.read_bytes():
            raise ValueError(f"Review predates/does not pin {path.name}")
        source_commit = git("log", "-1", "--format=%H", "HEAD", "--", rel).decode().strip()
        ancestor(source_commit, commit)
    return commit


def require_decision(review: Path, phrase: str) -> None:
    """Only one standalone decision line counts; prose quotations never do."""
    content = review.read_text()
    if re.search(r"\bDecision:\s*HOLD\b", content, flags=re.IGNORECASE):
        raise ValueError(f"Contradictory HOLD decision: {review.name}")
    lines = [line.strip() for line in content.splitlines()]
    decisions = [line for line in lines if line.startswith("Decision: ")]
    if decisions != [phrase]:
        raise ValueError(f"Independent exact GO decision absent/contradicted: {review.name}")


def checkpoint_tree(path: Path = CHECKPOINT) -> str:
    """Verify each file against the pinned Hugging Face revision-tree manifest."""
    if not path.is_dir() or path.is_symlink():
        raise ValueError("Checkpoint directory absent/unsafe")
    pinned = "e8e4220e9f8a1ed5fd4eb7c640c7061d10516530476438cd7f57b555b7e1e88a"
    tree = path / ".cache/huggingface/trees/3060fe34b35ba5d44202ce9ff3c097642914f8f3.json"
    if tree.is_symlink() or not tree.is_file() or sha(tree) != pinned:
        raise ValueError("Pinned checkpoint tree manifest differs")
    files = read_json(tree)["files"]
    required = ("config.json", "preprocessor_config.json", "generation_config.json",
                "tekken.json", "model.safetensors.index.json",
                "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors")
    if any(name not in files for name in required):
        raise ValueError("Pinned checkpoint tree missing required files")
    if {p.name for p in path.glob("*.safetensors")} != {
            "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"}:
        raise ValueError("Checkpoint contains missing/extra weight shard")
    if any(p.is_symlink() for p in path.rglob("*")):
        raise ValueError("Checkpoint contains symlink")
    for relative in required:
        item = files[relative]
        file = path / relative
        if not file.resolve().is_relative_to(path.resolve()) or file.is_symlink() or not file.is_file():
            raise ValueError("Checkpoint file absent/unsafe")
        if file.stat().st_size != item["size"]:
            raise ValueError(f"Checkpoint file changed: {relative}")
        if "lfs_sha256" in item:
            if sha(file) != item["lfs_sha256"]:
                raise ValueError(f"Checkpoint file hash changed: {relative}")
        else:
            content = file.read_bytes()
            blob = hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()
            if blob != item["blob_id"]:
                raise ValueError(f"Checkpoint Git blob changed: {relative}")
    return pinned


def validate_receipt(receipt: dict) -> None:
    import jsonschema  # CPU-only preflight dependency, never a model import.
    schema = read_json(ENV_SCHEMA)
    jsonschema.Draft202012Validator(schema, format_checker=jsonschema.FormatChecker()).validate(receipt)
    if receipt["sources"]["runner"] != sha(RUNNER) or receipt["sources"]["auditor"] != sha(AUDITOR) or \
            receipt["sources"]["gate"] != sha(GATE):
        raise ValueError("Environment source hashes differ")
    if receipt["checkpoint"]["tree_sha256"] != checkpoint_tree():
        raise ValueError("Checkpoint tree differs")
    bundle = Path(receipt["storage"]["input_bundle_path"])
    if (not bundle.is_absolute() or bundle.is_symlink() or not bundle.is_file() or
            bundle.stat().st_size != receipt["storage"]["input_bundle_bytes"] or
            sha(bundle) != receipt["storage"]["input_bundle_sha256"] or
            sha(bundle) != "0bc13e5831d3c4c325ad903bcbbdadf9f40d2e38eb67346ac68847bb1840dea2"):
        raise ValueError("Input bundle bytes/path differ")
    if Path(receipt["storage"]["output_root"]).resolve() != WORKER_RUNS.resolve():
        raise ValueError("Environment output root differs")


def verify_live_environment(receipt: dict) -> None:
    """Pre-weight installed-code and device probe; no model construction."""
    probe = r'''
import hashlib, inspect, json, torch, transformers, accelerate, safetensors
from transformers.models.voxtral import modeling_voxtral, processing_voxtral
from transformers.models.whisper import feature_extraction_whisper
from transformers.models.llama import modeling_llama
modules = {"modeling_voxtral": modeling_voxtral, "processing_voxtral": processing_voxtral,
           "feature_extraction_whisper": feature_extraction_whisper, "modeling_llama": modeling_llama}
def digest(module):
    path = inspect.getfile(module)
    with open(path, "rb") as stream: return hashlib.sha256(stream.read()).hexdigest()
print(json.dumps({"torch": torch.__version__, "torch_cuda": torch.version.cuda,
 "cudnn": torch.backends.cudnn.version(), "transformers": transformers.__version__,
 "accelerate": accelerate.__version__, "safetensors": safetensors.__version__,
 "gpu_name": torch.cuda.get_device_name(0),
 "compute_capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
 "gpu_count": torch.cuda.device_count(),
 "sources": {name: digest(module) for name, module in modules.items()}}))
'''
    result = subprocess.run([sys.executable, "-c", probe], check=True, capture_output=True,
                            text=True, timeout=30, env={**os.environ, "HF_HUB_OFFLINE": "1",
                                                     "TRANSFORMERS_OFFLINE": "1"})
    live = json.loads(result.stdout)
    for key in ("torch", "torch_cuda", "cudnn", "transformers", "accelerate", "safetensors"):
        if live[key] != receipt["software"][key]:
            raise ValueError(f"Live installed {key} differs")
    if live["sources"] != {k: receipt["sources"][k] for k in live["sources"]} or \
            live["gpu_count"] != 1 or live["gpu_name"] != receipt["machine"]["gpu_name"] or \
            live["compute_capability"] != receipt["machine"]["compute_capability"]:
        raise ValueError("Live source/device probe differs")
    gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid,memory.total,driver_version",
                                   "--format=csv,noheader,nounits"], timeout=5).decode().strip().splitlines()
    if len(gpu) != 1:
        raise ValueError("Live GPU count differs")
    uuid, memory_mib, driver = [x.strip() for x in gpu[0].split(",")]
    if uuid != receipt["machine"]["gpu_uuid"] or \
            int(memory_mib) * 1024**2 != receipt["machine"]["gpu_memory_bytes"] or \
            driver != receipt["machine"]["driver_version"]:
        raise ValueError("Live GPU identity differs")


def verify_freeze() -> tuple[dict, dict]:
    """Read-only admission; any missing receipt/review/freeze means HOLD."""
    for key, path in BOUND.items():
        committed_same(path)
        if key in EXPECTED and sha(path) != EXPECTED[key]:
            raise ValueError(f"Fixed protocol hash differs: {key}")
    pilot = read_json(PILOT)
    if pilot.get("schema") != "voxtral_old_fit_grad_leaf_cuda_byte_pilot_receipt_v1" or \
            pilot.get("pilot_source_sha256") != sha(PILOT_SOURCE) or \
            pilot.get("runner_source_sha256") != sha(RUNNER) or \
            pilot.get("result_schema_sha256") != sha(RESULT_SCHEMA) or \
            pilot.get("max_worker_bytes") != 1048576000 or \
            [x.get("case_id") for x in pilot.get("measurements", [])] != list(CASE_IDS) or \
            any(x.get("within_worker_cap") is not True or x.get("sidecar_bytes", MAX_CASE_BYTES) >= 1048576000
                for x in pilot["measurements"]):
        raise ValueError("CPU byte-budget pilot invalid")
    source_commit = reviewed_source((RUNNER, RUNNER_TESTS, GATE, GATE_TESTS, RESULT_SCHEMA,
                                     PROTOCOL_V1, PROTOCOL_V2, PILOT, PILOT_SOURCE),
                                    SOURCE_REVIEW, GO_SOURCE)
    auditor_commit = reviewed_source((AUDITOR, AUDITOR_TESTS), AUDITOR_REVIEW, GO_AUDITOR)
    env_commit = reviewed_source((ENV_RECEIPT, ENV_SCHEMA), ENV_REVIEW, GO_ENV)
    if len({source_commit, auditor_commit, env_commit}) != 3:
        raise ValueError("Independent source reviews share a commit")
    receipt = read_json(ENV_RECEIPT)
    validate_receipt(receipt)
    freeze = read_json(FREEZE)
    expected_keys = {"status", "source_review_commit", "auditor_review_commit",
                     "environment_review_commit", "reviewed_utc", "attempt_ids", "caps",
                     "checkpoint_tree_sha256", "input_bundle_sha256", "input_bundle_path",
                     "case_ids", "planned_counts", *BOUND}
    if set(freeze) != expected_keys or freeze["status"] != STATUS or \
            freeze["source_review_commit"] != source_commit or \
            freeze["auditor_review_commit"] != auditor_commit or \
            freeze["environment_review_commit"] != env_commit or \
            freeze["case_ids"] != list(CASE_IDS) or \
            freeze["caps"] != {"cuda_bytes": MAX_CUDA, "ram_bytes": MAX_RAM,
                                "worker_seconds": MAX_WORKER_SECONDS, "audit_seconds": MAX_AUDIT_SECONDS,
                                "case_output_bytes": MAX_CASE_BYTES, "campaign_output_bytes": MAX_CAMPAIGN_BYTES,
                                "stream_bytes": MAX_STREAM_BYTES, "minimum_free_bytes": MIN_FREE_BYTES} or \
            freeze["planned_counts"] != {"cases": 2, "full_forwards": 32,
                                          "direct_forwards": 32, "passive_forwards": 2,
                                          "vjps": 0, "generations": 0}:
        raise ValueError("External freeze scope/caps differ")
    attempts = freeze["attempt_ids"]
    if not isinstance(attempts, list) or len(attempts) != 2 or len(set(attempts)) != 2 or \
            any(not isinstance(x, str) or ATTEMPT.fullmatch(x) is None for x in attempts):
        raise ValueError("Two fixed distinct attempt IDs required")
    if freeze["checkpoint_tree_sha256"] != receipt["checkpoint"]["tree_sha256"] or \
            freeze["input_bundle_sha256"] != receipt["storage"]["input_bundle_sha256"] or \
            freeze["input_bundle_path"] != receipt["storage"]["input_bundle_path"]:
        raise ValueError("Frozen checkpoint/input differs")
    for key, path in BOUND.items():
        if freeze[key] != sha(path):
            raise ValueError(f"Frozen file changed: {key}")
    freeze_commit = first_add(FREEZE)
    for earlier in (source_commit, auditor_commit, env_commit):
        ancestor(earlier, freeze_commit)
    committed_same(FREEZE_REVIEW)
    review_commit = first_add(FREEZE_REVIEW)
    require_decision(FREEZE_REVIEW, GO_FREEZE)
    if sha(FREEZE) not in FREEZE_REVIEW.read_text():
        raise ValueError("Independent freeze GO/hash absent")
    ancestor(freeze_commit, review_commit)
    datetime.fromisoformat(freeze["reviewed_utc"].replace("Z", "+00:00"))
    return freeze, receipt


def verify_worker_start(case_index: int, attempt_id: str, freeze_sha256: str) -> dict:
    """Atomic no-retry worker claim, called before model construction."""
    freeze, receipt = verify_freeze()
    if case_index not in (0, 1) or attempt_id != freeze["attempt_ids"][case_index] or \
            freeze_sha256 != sha(FREEZE):
        raise ValueError("Worker case/attempt/freeze differs")
    verify_live_environment(receipt)
    reservation = read_json(RUNS / "reservation.json")
    start = read_json(RUNS / attempt_id / "launch_start.json")
    if reservation != {"status": "cuda_campaign_reserved_no_retry", "attempt_ids": freeze["attempt_ids"],
                       "freeze_sha256": freeze_sha256} or \
            start.get("status") != "cuda_case_started" or start.get("case_index") != case_index or \
            start.get("attempt_id") != attempt_id or start.get("freeze_sha256") != freeze_sha256 or \
            (WORKER_RUNS / attempt_id).exists():
        raise ValueError("Worker launch chronology/output differs")
    if case_index == 1:
        prior = read_json(RUNS / freeze["attempt_ids"][0] / "launch_finish.json")
        if prior.get("status") != "cuda_case_completed_cpu_audited_only":
            raise ValueError("Prior case has no validated CPU audit")
    put(RUNS / attempt_id / "worker_claim.json", {
        "status": "cuda_worker_claimed_no_retry", "case_index": case_index,
        "attempt_id": attempt_id, "freeze_sha256": freeze_sha256,
        "wall_utc": datetime.now(timezone.utc).isoformat()})
    return {"environment_receipt_sha256": sha(ENV_RECEIPT),
            "checkpoint_tree_sha256": receipt["checkpoint"]["tree_sha256"],
            "input_bundle_path": receipt["storage"]["input_bundle_path"],
            "input_bundle_sha256": receipt["storage"]["input_bundle_sha256"]}


def resource_usage(process_group: int) -> tuple[int, int]:
    """Conservative Linux external watchdog: system RAM and total device use."""
    if sys.platform != "linux":
        raise ValueError("CUDA watchdog requires Linux /proc")
    memory = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        fields = line.split()
        if len(fields) >= 2 and fields[0].rstrip(":") in ("MemTotal", "MemAvailable"):
            memory[fields[0].rstrip(":")] = int(fields[1]) * 1024
    if set(memory) != {"MemTotal", "MemAvailable"}:
        raise ValueError("Cannot measure system RAM")
    # Whole-device use is intentionally conservative when another process exists.
    output = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used",
                                      "--format=csv,noheader,nounits"], timeout=3).decode().splitlines()
    if len(output) != 1:
        raise ValueError("Expected exactly one monitored GPU")
    cuda_bytes = int(output[0].strip()) * 1024**2
    return cuda_bytes, memory["MemTotal"] - memory["MemAvailable"]


def bounded(command: list[str], capture: Path, names: tuple[str, str], output: Path,
            seconds: int, *, campaign: bool = False) -> tuple[int | None, float, str | None]:
    began = time.monotonic()
    reason = None
    with (capture / names[0]).open("xb") as stdout, (capture / names[1]).open("xb") as stderr:
        proc = subprocess.Popen(command, stdout=stdout, stderr=stderr, start_new_session=True,
                                env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                                     "HF_HUB_DISABLE_TELEMETRY": "1", "PYTHONNOUSERSITE": "1"})
        while True:
            code = proc.poll()
            elapsed = time.monotonic() - began
            if elapsed >= seconds:
                reason = "process_time_cap"
            elif max((capture / names[0]).stat().st_size,
                     (capture / names[1]).stat().st_size) >= MAX_STREAM_BYTES:
                reason = "capture_byte_cap"
            elif tree_bytes(capture) + tree_bytes(output) >= MAX_CASE_BYTES - FINISH_RESERVE:
                reason = "case_output_byte_cap"
            elif campaign and tree_bytes(RUNS) + tree_bytes(WORKER_RUNS) >= MAX_CAMPAIGN_BYTES - FINISH_RESERVE:
                reason = "campaign_output_byte_cap"
            elif code is None:
                try:
                    gpu, ram = resource_usage(proc.pid)
                    if gpu >= MAX_CUDA:
                        reason = "external_cuda_memory_cap"
                    elif ram >= MAX_RAM:
                        reason = "external_system_ram_cap"
                except (OSError, ValueError, subprocess.SubprocessError) as error:
                    reason = f"resource_watchdog_unavailable:{type(error).__name__}"
            if reason and code is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    code = proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    code = proc.wait(timeout=3)
                break
            if reason or code is not None:
                break
            time.sleep(0.1)
    return code, time.monotonic() - began, reason


def _mark_stream_overshoot(capture: Path, finish: dict) -> None:
    for name in ("worker_stdout.txt", "worker_stderr.txt", "audit_stdout.txt", "audit_stderr.txt",
                 "campaign_audit_stdout.txt", "campaign_audit_stderr.txt"):
        path = capture / name
        if path.is_file() and path.stat().st_size >= MAX_STREAM_BYTES:
            finish["status"] = "cuda_case_failed_no_retry"
            finish["failure_reason"] = "capture_byte_cap"
            original_size, original_sha = path.stat().st_size, sha(path)
            with path.open("r+b") as stream:
                prefix = stream.read(512)
                stream.truncate(0)
                stream.seek(0)
                stream.write(prefix)
            finish.setdefault("trimmed_streams", {})[name] = {
                "original_bytes": original_size, "original_sha256": original_sha,
                "retained_prefix_bytes": len(prefix)}


def _case_receipts(case_index: int, attempt_id: str, freeze_sha: str,
                   receipt: dict) -> tuple[Path, Path]:
    capture = RUNS / attempt_id
    output = WORKER_RUNS / attempt_id
    claim = read_json(capture / "worker_claim.json")
    if claim.get("status") != "cuda_worker_claimed_no_retry" or \
            claim.get("case_index") != case_index or claim.get("attempt_id") != attempt_id or \
            claim.get("freeze_sha256") != freeze_sha:
        raise ValueError("Worker atomic claim missing/invalid")
    if (output / "error.json").exists() or not (output / "result.json").is_file():
        raise ValueError("Worker failed or result missing")
    result = read_json(output / "result.json")
    if result.get("status") != "completed" or result.get("schema") != \
            "voxtral_old_fit_grad_leaf_cuda_result_v1" or \
            (result.get("case_id"), result.get("case_index"), result.get("attempt_id"),
             result.get("freeze_sha256")) != (CASE_IDS[case_index], case_index, attempt_id, freeze_sha) or \
            result.get("runner_sha256") != sha(RUNNER) or \
            result.get("protocol_v1_sha256") != sha(PROTOCOL_V1) or \
            result.get("protocol_v2_sha256") != sha(PROTOCOL_V2) or \
            result.get("schema_sha256") != sha(RESULT_SCHEMA):
        raise ValueError("Worker result provenance differs")
    meta = read_json(output / "meta.json")
    if meta.get("environment_receipt_sha256") != sha(ENV_RECEIPT) or \
            meta.get("checkpoint_tree_sha256") != receipt["checkpoint"]["tree_sha256"]:
        raise ValueError("Worker environment/checkpoint receipt differs")
    for key, limit in (("elapsed_seconds", MAX_WORKER_SECONDS),
                       ("peak_cuda_allocated_bytes", MAX_CUDA),
                       ("peak_cuda_reserved_bytes", MAX_CUDA),
                       ("peak_process_rss_bytes", MAX_RAM)):
        value = result.get(key)
        if type(value) not in (float, int) or not math.isfinite(value) or not 0 <= value < limit:
            raise ValueError(f"Worker {key} cap/receipt invalid")
    if tree_bytes(output) + tree_bytes(capture) >= MAX_CASE_BYTES:
        raise ValueError("Case output cap reached")
    return capture, output


def _audit_one(case_index: int, capture: Path, output: Path, freeze_sha: str) -> dict:
    audit_path = capture / "case_audit.json"
    command = [sys.executable, str(AUDITOR), "--case-index", str(case_index),
               "--run-dir", str(output), "--output", str(audit_path),
               "--freeze", str(FREEZE), "--expected-freeze-sha256", freeze_sha,
               "--environment-receipt", str(ENV_RECEIPT)]
    code, elapsed, reason = bounded(command, capture, ("audit_stdout.txt", "audit_stderr.txt"),
                                    output, MAX_AUDIT_SECONDS, campaign=True)
    if reason or code != 0 or elapsed >= MAX_AUDIT_SECONDS or not audit_path.is_file():
        raise ValueError(reason or "Single-case CPU audit failed")
    audit = read_json(audit_path)
    if audit.get("status") != "audited_saved_data_only" or \
            audit.get("result_sha256") != sha(output / "result.json") or \
            audit.get("case_index") != case_index or audit.get("case_id") != CASE_IDS[case_index]:
        raise ValueError("Single-case CPU audit rejected provenance")
    return audit


def _audit_campaign(attempt_ids: list[str], capture: Path, output: Path,
                    freeze_sha: str) -> dict:
    audit_path = RUNS / "campaign_audit.json"
    command = [sys.executable, str(AUDITOR), "--beaver-dir", str(WORKER_RUNS / attempt_ids[0]),
               "--waterfall-dir", str(WORKER_RUNS / attempt_ids[1]), "--output", str(audit_path),
               "--freeze", str(FREEZE), "--expected-freeze-sha256", freeze_sha,
               "--environment-receipt", str(ENV_RECEIPT)]
    code, elapsed, reason = bounded(command, capture, ("campaign_audit_stdout.txt", "campaign_audit_stderr.txt"),
                                    output, 2 * MAX_AUDIT_SECONDS, campaign=True)
    if reason or code != 0 or elapsed >= 2 * MAX_AUDIT_SECONDS or not audit_path.is_file():
        raise ValueError(reason or "Two-case CPU audit failed")
    audit = read_json(audit_path)
    expected_case_ids = list(CASE_IDS)
    expected_cells = [(case_id, site) for case_id in CASE_IDS for site in ("encoder31", "projector")]
    if audit.get("schema") != "voxtral_old_fit_grad_leaf_cuda_audit_v1" or \
            audit.get("planned_case_count") != 2 or audit.get("planned_forward_count") != 66 or \
            audit.get("saved_forward_count") != 66 or audit.get("planned_cell_count") != 4 or \
            audit.get("direct_full_comparison_count") != 96 or \
            len(audit.get("cases", [])) != 2 or \
            [x.get("case_id") for x in audit["cases"]] != expected_case_ids or \
            [x.get("case_index") for x in audit["cases"]] != [0, 1] or \
            [x.get("saved_forwards") for x in audit["cases"]] != [32, 34] or \
            [x.get("direct_full_comparison_count") for x in audit["cases"]] != [48, 48] or \
            [x.get("result_sha256") for x in audit["cases"]] != [
                sha(WORKER_RUNS / attempt_ids[0] / "result.json"),
                sha(WORKER_RUNS / attempt_ids[1] / "result.json")] or \
            any(x.get("status") != "audited_saved_data_only" for x in audit["cases"]) or \
            len(audit.get("cells", [])) != 4 or \
            [(x.get("case_id"), x.get("site")) for x in audit["cells"]] != expected_cells or \
            any(x.get("label") not in ("same_pattern", "different_pattern") for x in audit["cells"]):
        raise ValueError("Two-case CPU audit invalid/incomplete")
    return audit


def launch(case_index: int) -> dict:
    if sys.platform != "linux":
        raise ValueError("CUDA launch requires Linux")
    freeze, receipt = verify_freeze()
    if case_index not in (0, 1):
        raise ValueError("Case index must be 0 or 1")
    attempt_id = freeze["attempt_ids"][case_index]
    freeze_sha = sha(FREEZE)
    verify_live_environment(receipt)
    if RUNS.is_symlink() or WORKER_RUNS.is_symlink():
        raise ValueError("Run root is symlinked")
    RUNS.mkdir(parents=True, exist_ok=True)
    WORKER_RUNS.mkdir(parents=True, exist_ok=True)
    if case_index == 0:
        if (RUNS / "reservation.json").exists() or \
                shutil.disk_usage(WORKER_RUNS).free < MIN_FREE_BYTES:
            raise ValueError("Campaign already reserved or under 30 GiB free")
        put(RUNS / "reservation.json", {"status": "cuda_campaign_reserved_no_retry",
                                        "attempt_ids": freeze["attempt_ids"],
                                        "freeze_sha256": freeze_sha})
    else:
        reserved = read_json(RUNS / "reservation.json")
        if reserved != {"status": "cuda_campaign_reserved_no_retry",
                        "attempt_ids": freeze["attempt_ids"], "freeze_sha256": freeze_sha} or \
                read_json(RUNS / freeze["attempt_ids"][0] / "launch_finish.json").get("status") != \
                "cuda_case_completed_cpu_audited_only":
            raise ValueError("Case order or prior CPU audit differs")
    capture = RUNS / attempt_id
    output = WORKER_RUNS / attempt_id
    if output.exists():
        raise ValueError("Attempt output already exists")
    capture.mkdir(exist_ok=False)
    command = [sys.executable, str(RUNNER), "--case-index", str(case_index),
               "--attempt-id", attempt_id, "--freeze-sha256", freeze_sha]
    put(capture / "launch_start.json", {
        "status": "cuda_case_started", "case_index": case_index, "case_id": CASE_IDS[case_index],
        "attempt_id": attempt_id, "freeze_sha256": freeze_sha,
        "wall_utc": datetime.now(timezone.utc).isoformat(),
        "monotonic_start": time.monotonic(), "worker_command": command,
        "worker_output": str(output)})
    finish = {"status": "cuda_case_failed_no_retry", "case_index": case_index,
              "case_id": CASE_IDS[case_index], "attempt_id": attempt_id,
              "freeze_sha256": freeze_sha, "worker_returncode": None,
              "audit_sha256": None, "campaign_audit_sha256": None,
              "result_sha256": None, "post_run_source_verified": False}
    began = time.monotonic()
    try:
        code, elapsed, reason = bounded(command, capture, ("worker_stdout.txt", "worker_stderr.txt"),
                                        output, MAX_WORKER_SECONDS, campaign=True)
        finish["worker_returncode"] = code
        finish["model_work_elapsed_seconds"] = elapsed
        if reason or code != 0 or elapsed >= MAX_WORKER_SECONDS:
            raise ValueError(reason or "Worker failed")
        _case_receipts(case_index, attempt_id, freeze_sha, receipt)
        finish["result_sha256"] = sha(output / "result.json")
        if sha(FREEZE) != freeze_sha or verify_freeze()[0] != freeze:
            raise ValueError("Source/freeze changed after worker")
        audit = _audit_one(case_index, capture, output, freeze_sha)
        finish["audit_sha256"] = sha(capture / "case_audit.json")
        if case_index == 1:
            _audit_campaign(freeze["attempt_ids"], capture, output, freeze_sha)
            finish["campaign_audit_sha256"] = sha(RUNS / "campaign_audit.json")
        if sha(FREEZE) != freeze_sha or verify_freeze()[0] != freeze:
            raise ValueError("Source/freeze changed after audit")
        if tree_bytes(capture) + tree_bytes(output) + FINISH_RESERVE >= MAX_CASE_BYTES or \
                tree_bytes(RUNS) + tree_bytes(WORKER_RUNS) + FINISH_RESERVE >= MAX_CAMPAIGN_BYTES:
            raise ValueError("Output cap reached before finish")
        finish["post_run_source_verified"] = True
        finish["status"] = "cuda_case_completed_cpu_audited_only"
    except Exception as error:
        finish["failure_reason"] = str(error)
        finish["exception_type"] = type(error).__name__
        if (output / "error.json").is_file():
            finish["worker_error_sha256"] = sha(output / "error.json")
    finally:
        finish["wall_utc"] = datetime.now(timezone.utc).isoformat()
        finish["elapsed_seconds"] = time.monotonic() - began
        _mark_stream_overshoot(capture, finish)
        put(capture / "launch_finish.json", finish)
    return finish


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-freeze", action="store_true")
    mode.add_argument("--run-case", type=int, choices=(0, 1))
    args = parser.parse_args()
    if args.check_freeze:
        verify_freeze()
        print(json.dumps({"status": "cuda_freeze_verified_only", "freeze_sha256": sha(FREEZE)}))
        return
    result = launch(args.run_case)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    if result["status"] != "cuda_case_completed_cpu_audited_only":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
