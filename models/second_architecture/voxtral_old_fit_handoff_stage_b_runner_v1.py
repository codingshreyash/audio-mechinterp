"""Prospective Stage-B Voxtral handoff driver; source for independent review.

The live entry point is deliberately closed until a separate Stage-B source,
auditor, launcher and external freeze have been reviewed. The phase routines
are complete enough to review and CPU-fake-test without loading model weights.
Each eventual phase is one fresh process, BF16 direct LM before FP32 replay.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
import math
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

import voxtral_fit_pilot_capability_v1 as fixture
import voxtral_old_fit_handoff_native_stage_a_v1 as native
import voxtral_old_fit_handoff_replay_stage_b_v1 as replay
import voxtral_old_fit_handoff_stage_a_runner_v1 as stage_a_runner
import voxtral_v1_fit_backward_numerical_followup_v2 as prior


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
STAGE_A_RUNS = HERE / "runs/voxtral_old_fit_handoff_stage_a_launch_v1"
RUNS = HERE / "runs/voxtral_old_fit_handoff_stage_b_v1"
STAGE_A_FREEZE = HERE / "VOXTRAL_OLD_FIT_HANDOFF_STAGE_A_EXTERNAL_FREEZE_V1.json"
STAGE_A_MANIFESTS = {
    "bf16": HERE / "VOXTRAL_OLD_FIT_HANDOFF_STAGE_A_BF16_ATTEMPT_MANIFEST_V1.json",
    "fp32": HERE / "VOXTRAL_OLD_FIT_HANDOFF_STAGE_A_FP32_ATTEMPT_MANIFEST_V1.json",
}
STAGE_B_BF16_MANIFEST = HERE / "VOXTRAL_OLD_FIT_HANDOFF_STAGE_B_BF16_ATTEMPT_MANIFEST_V1.json"
PHASES = ("bf16_direct_lm", "fp32_replay")
STAGE_A_RUNNER_SHA = "9f8f6b52d3e218acc496b78b35247cfdf322672cd487e585b92d58f08205f1dc"
STAGE_A_AUDITOR_SHA = "4f800ed604295ae0e4fbd53c893fccd14b8968badb43b835e6ec975e49b0cf3d"
STAGE_B_PRIMITIVES_SHA = "bef337c90a024e034e60818761b4be79371ca0767f83d28efe29269ba0cc89e1"
STAGE_A_LAUNCH_STATUS = "launch_completed_audited_old_fit_handoff_stage_a_native_only"
STAGE_A_AUDIT_STATUS = "audited_completed_old_fit_handoff_stage_a_native_only"
STATUS = "completed_old_fit_handoff_stage_b_replay_only"
MAX_B_BYTES = 1024**3
MAX_TOTAL_SECONDS = 4500
MAX_MPS_DRIVER_BYTES = 30 * 1024**3
CAP_SECONDS = dict(replay.STAGE_B_CAP_SECONDS)
COUNTS = {
    "bf16_direct_lm": {"direct_lm_forwards": 24, "source_pair_forwards": 0,
                         "source_parity_forwards": 0, "source_vjps": 0,
                         "full_model_override_forwards": 0},
    "fp32_replay": {"direct_lm_forwards": 24, "source_pair_forwards": 24,
                    "source_parity_forwards": 6, "source_vjps": 68,
                    "full_model_override_forwards": 4},
}
# An external Stage-B freeze and independent auditor/launcher review must set
# this in a separate reviewed change. No local result or CLI flag can set it.
LIVE_STAGE_B_SOURCE_PINS: dict[str, str] | None = None


def sha(path: Path) -> str:
    return replay.sha_file(path)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes(), parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v)))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path.name}")
    return value


def committed_same(repo: Path, path: Path) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Required committed file absent: {path.name}")
    relative = path.resolve().relative_to(repo.resolve()).as_posix()
    result = subprocess.run(["git", "-C", str(repo), "show", f"HEAD:{relative}"],
                            capture_output=True, check=False)
    if result.returncode or path.read_bytes() != result.stdout:
        raise ValueError(f"Uncommitted or changed file: {path.name}")


def first_add(repo: Path, path: Path) -> str:
    relative = path.resolve().relative_to(repo.resolve()).as_posix()
    result = subprocess.run(["git", "-C", str(repo), "log", "--diff-filter=A",
                             "--format=%H", "HEAD", "--", relative],
                            capture_output=True, text=True, check=False)
    lines = result.stdout.splitlines()
    if result.returncode or len(lines) != 1 or not re.fullmatch(r"[0-9a-f]{40}", lines[0]):
        raise ValueError("Receipt must have one Git first-add commit")
    return lines[0]


def strictly_later(repo: Path, earlier: Path, later: Path) -> None:
    before, after = first_add(repo, earlier), first_add(repo, later)
    if before == after or subprocess.run(["git", "-C", str(repo), "merge-base",
                                         "--is-ancestor", before, after],
                                        capture_output=True, check=False).returncode:
        raise ValueError(f"Receipt chronology invalid: {later.name}")


def _one_attempt(runs: Path, arm: str, name: str) -> Path:
    if not name or Path(name).name != name or name in (".", ".."):
        raise ValueError("Attempt basename invalid")
    folder = runs / arm
    attempt = folder / name
    if not folder.is_dir() or sorted(p for p in folder.iterdir() if p.is_dir()) != [attempt]:
        raise ValueError(f"Stage-A {arm} attempt missing or retried")
    return attempt


def verify_stage_a_manifests(*, repo: Path = REPO, runs: Path = STAGE_A_RUNS,
                             freeze: Path = STAGE_A_FREEZE,
                             manifests: dict[str, Path] = STAGE_A_MANIFESTS) -> dict[str, Any]:
    """Bind both external manifests to audited full Stage-A byte chains.

    This intentionally rechecks the sidecars instead of trusting manifest
    paths or a runner-authored `parity_passed` flag on its own.
    """
    replay.require_protocol()
    if sha(Path(stage_a_runner.__file__)) != STAGE_A_RUNNER_SHA or \
            sha(HERE / "voxtral_old_fit_handoff_stage_a_audit_v1.py") != STAGE_A_AUDITOR_SHA or \
            sha(Path(replay.__file__)) != STAGE_B_PRIMITIVES_SHA:
        raise ValueError("Reviewed Stage-A/Stage-B dependency bytes changed")
    for file in (freeze, manifests["bf16"], manifests["fp32"]):
        committed_same(repo, file)
    strictly_later(repo, freeze, manifests["bf16"])
    strictly_later(repo, manifests["bf16"], manifests["fp32"])
    freeze_sha = sha(freeze)
    frozen = read_json(freeze)
    if (frozen.get("status") != "frozen_old_fit_handoff_stage_a_v1" or
            frozen.get("protocol_sha256") != replay.PROTOCOL_SHA256 or
            frozen.get("runner_sha256") != STAGE_A_RUNNER_SHA or
            frozen.get("native_source_sha256") != replay.NATIVE_SOURCE_SHA256 or
            frozen.get("auditor_sha256") != STAGE_A_AUDITOR_SHA):
        raise ValueError("Stage-A external freeze does not bind reviewed source")
    bound: dict[str, Any] = {"freeze_sha256": freeze_sha, "arms": {}}
    previous = None
    for arm in ("bf16", "fp32"):
        path = manifests[arm]
        manifest = read_json(path)
        expected = {"status", "arm", "attempt_relative_path", "freeze_receipt_sha256",
                    "launch_start_sha256", "launch_finish_sha256", "result_sha256",
                    "audit_sha256", "model_work_elapsed_seconds"}
        if (set(manifest) != expected or manifest["status"] != STAGE_A_LAUNCH_STATUS or
                manifest["arm"] != arm or manifest["freeze_receipt_sha256"] != freeze_sha):
            raise ValueError(f"Stage-A {arm} manifest schema/status invalid")
        attempt = _one_attempt(runs, arm, manifest["attempt_relative_path"])
        files = {"start": attempt / "launch_start.json", "finish": attempt / "launch_finish.json",
                 "result": attempt / "runner/result.json", "audit": attempt / "audit.json"}
        for key, field in (("start", "launch_start_sha256"), ("finish", "launch_finish_sha256"),
                           ("result", "result_sha256"), ("audit", "audit_sha256")):
            if files[key].is_symlink() or not files[key].is_file() or sha(files[key]) != manifest[field]:
                raise ValueError(f"Stage-A {arm} {key} receipt bytes differ")
        start, finish, result, audit = (read_json(files[key]) for key in files)
        if (start.get("arm") != arm or start.get("freeze_receipt_sha256") != freeze_sha or
                start.get("runner_sha256") != STAGE_A_RUNNER_SHA or
                start.get("auditor_sha256") != STAGE_A_AUDITOR_SHA or
                finish.get("status") != STAGE_A_LAUNCH_STATUS or finish.get("arm") != arm or
                finish.get("freeze_receipt_sha256") != freeze_sha or
                finish.get("post_run_source_verified") is not True or
                finish.get("runner_returncode") != 0 or finish.get("auditor_returncode") != 0 or
                finish.get("runner_result_sha256") != manifest["result_sha256"] or
                finish.get("audit_sha256") != manifest["audit_sha256"] or
                result.get("status") != stage_a_runner.RESULT_STATUS or result.get("arm") != arm or
                result.get("script_sha256") != STAGE_A_RUNNER_SHA or
                result.get("native_source_sha256") != replay.NATIVE_SOURCE_SHA256 or
                result.get("protocol_sha256") != replay.PROTOCOL_SHA256 or
                result.get("parity_passed") is not True or
                result.get("planned_counts") != stage_a_runner.planned_counts() or
                result.get("actual_counts") != stage_a_runner.planned_counts() or
                audit.get("status") != STAGE_A_AUDIT_STATUS or audit.get("arm") != arm or
                audit.get("result_sha256") != manifest["result_sha256"] or
                audit.get("auditor_sha256") != STAGE_A_AUDITOR_SHA or
                audit.get("runner_sha256") != STAGE_A_RUNNER_SHA or
                audit.get("native_source_sha256") != replay.NATIVE_SOURCE_SHA256 or
                audit.get("protocol_sha256") != replay.PROTOCOL_SHA256):
            raise ValueError(f"Stage-A {arm} source/hook/parity/audit chain invalid")
        for stem in ("meta", "model_layout", "progress"):
            file = attempt / "runner" / ("progress.jsonl" if stem == "progress" else f"{stem}.json")
            if file.is_symlink() or not file.is_file() or sha(file) != result.get(f"{stem}_sha256"):
                raise ValueError(f"Stage-A {arm} {stem} hash differs")
        for stem in ("probes", "details"):
            digest = result.get(f"{stem}_sha256")
            if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest) or \
                    result.get(f"{stem}_path") != f"blobs/{digest}.pt":
                raise ValueError(f"Stage-A {arm} {stem} blob descriptor differs")
            file = attempt / "runner/blobs" / f"{digest}.pt"
            if (file.is_symlink() or not file.is_file() or sha(file) != digest or
                    file.stat().st_size != result.get(f"{stem}_bytes") or
                    audit.get(f"{stem}_sha256") != digest):
                raise ValueError(f"Stage-A {arm} {stem} blob differs")
        elapsed = manifest["model_work_elapsed_seconds"]
        if (isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)) or
                not math.isfinite(elapsed) or not 0 < elapsed < stage_a_runner.MAX_SECONDS[arm] or
                elapsed != finish.get("model_work_elapsed_seconds") or
                not isinstance(result.get("elapsed_seconds"), (int, float)) or
                not 0 < result["elapsed_seconds"] <= elapsed):
            raise ValueError(f"Stage-A {arm} model-work time invalid")
        if previous is not None and not strictly_after_utc(previous, start.get("wall_utc")):
            raise ValueError("FP32 Stage-A started before BF16 Stage-A finished")
        previous = finish.get("wall_utc")
        bound["arms"][arm] = {"manifest": manifest, "manifest_sha256": sha(path),
                              "attempt": attempt, "result": result, "audit": audit,
                              "elapsed_seconds": float(elapsed)}
    bound["stage_a_model_work_seconds"] = sum(bound["arms"][arm]["elapsed_seconds"]
                                             for arm in ("bf16", "fp32"))
    if bound["stage_a_model_work_seconds"] + sum(CAP_SECONDS.values()) > MAX_TOTAL_SECONDS:
        raise ValueError("Cumulative four-process cap lacks full Stage-B reservation")
    return bound


def strictly_after_utc(before: str, after: str) -> bool:
    from datetime import datetime
    try:
        left = datetime.fromisoformat(before.replace("Z", "+00:00"))
        right = datetime.fromisoformat(after.replace("Z", "+00:00"))
        return bool(left.utcoffset().total_seconds() == 0 and
                    right.utcoffset().total_seconds() == 0 and right > left)
    except (TypeError, AttributeError, ValueError):
        return False


def required_live_freeze() -> None:
    if LIVE_STAGE_B_SOURCE_PINS is None:
        raise ValueError("Stage-B live source/auditor/launcher review and external freeze absent")
    raise ValueError("Stage-B external freeze verifier not yet implemented")


def planned_counts(phase: str) -> dict[str, int]:
    if phase not in COUNTS:
        raise ValueError("Unknown Stage-B phase")
    return dict(COUNTS[phase])


def checked_pair_order(details: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    pairs = replay.cycle_zero_pairs(details)
    if [(p["site"], p["alpha"]) for p in pairs] != [
            (site, alpha) for site in ("encoder31", "projector") for alpha in replay.ALPHAS]:
        raise ValueError("Stage-B cycle-0 replay order differs")
    return pairs


def direct_replay(model, batch, pairs, dtype, *, device="mps", parity_native=False,
                  receipt=lambda _phase: None) -> tuple[list[dict], int]:
    records = []
    count = 0
    for pair in pairs:
        record = {"site": pair["site"], "cycle": 0, "alpha": pair["alpha"]}
        for label in ("zero", "edited"):
            lm_input = pair[label]["lm_input_full"]
            replayed = replay.direct_lm_forward(model, batch, lm_input.to(dtype), dtype,
                                                device=device)
            count += 1
            receipt(f"after_direct_lm_{pair['site']}_{pair['alpha']}_{label}")
            if parity_native:
                replay.assert_direct_parity(pair[label], replayed)
            record[label] = {"input": lm_input, "input_sha256": native.first.tensor_sha(lm_input),
                             "replayed": replayed}
        records.append(record)
    if count != 24:
        raise ValueError("Stage-B direct LM forward count changed")
    return records, count


def _same_replay(left: dict, right: dict) -> None:
    replay.assert_direct_parity(left["saved"], right["saved"])
    if left["saved"]["audio_placeholder_positions"] != right["saved"]["audio_placeholder_positions"]:
        raise ValueError("Source replay placeholder positions changed")
    for key in ("projector_word", "lm_input_word"):
        if not torch.equal(left["saved"][key], right["saved"][key]):
            raise ValueError(f"Source replay {key} parity failed")


def fp32_source_replays(model, batch, pairs, probes, *, device="mps",
                        receipt=lambda _phase: None) -> tuple[dict, dict[str, int]]:
    output = {"parity": {}, "vjps": {}, "pairs": []}
    counts = {"source_pair_forwards": 0, "source_parity_forwards": 0, "source_vjps": 0}
    for site in ("encoder31", "projector"):
        site_pairs = [pair for pair in pairs if pair["site"] == site]
        clean = site_pairs[0]["zero"]["source_full_post"][site]
        if any(not torch.equal(clean, pair["zero"]["source_full_post"][site])
               for pair in site_pairs):
            raise ValueError("BF16 clean replay source varies across signed steps")
        first = replay.source_replay_forward(model, batch, site=site, full_source=clean,
                                             probes=probes, grad=True, device=device)
        counts["source_parity_forwards"] += 1
        receipt(f"after_source_{site}_clean_graph")
        second = replay.source_replay_forward(model, batch, site=site, full_source=clean,
                                              probes=probes, device=device)
        counts["source_parity_forwards"] += 1
        receipt(f"after_source_{site}_clean_repeat")
        self_patch = replay.source_replay_forward(model, batch, site=site, full_source=clean,
                                                  probes=probes, self_clone=True, device=device)
        counts["source_parity_forwards"] += 1
        receipt(f"after_source_{site}_self")
        _same_replay(first, second)
        _same_replay(first, self_patch)
        gradients = replay.replay_source_vjps(first, probes, site)
        counts["source_vjps"] += sum(t.shape[0] for t in gradients.values())
        receipt(f"after_source_{site}_vjps")
        output["parity"][site] = {"clean": first["saved"], "repeat": second["saved"],
                                  "self": self_patch["saved"]}
        output["vjps"][site] = gradients
        del first
        for pair in site_pairs:
            record = {"site": site, "cycle": 0, "alpha": pair["alpha"]}
            for label in ("zero", "edited"):
                source = pair[label]["source_full_post"][site]
                trace = replay.source_replay_forward(model, batch, site=site,
                                                     full_source=source, probes=probes, device=device)
                counts["source_pair_forwards"] += 1
                receipt(f"after_source_{site}_{pair['alpha']}_{label}")
                record[label] = {"input": source, "input_sha256": native.first.tensor_sha(source),
                                 "trace": trace["saved"]}
            verified = replay.verify_source_replay_pair(
                {"saved": record["zero"]["trace"]}, {"saved": record["edited"]["trace"]},
                record["zero"]["input"], record["edited"]["input"], site)
            record["effective_edit"] = verified["source_effective_edit"]
            record["terminal_delta_l2"] = verified["terminal_delta_l2"]
            output["pairs"].append(record)
    if counts != {"source_pair_forwards": 24, "source_parity_forwards": 6, "source_vjps": 68}:
        raise ValueError("Stage-B FP32 source replay counts changed")
    return output, counts


def fp32_lm_validation(model, batch, pairs, direct_records, *, device="mps",
                       receipt=lambda _phase: None) -> tuple[list[dict], int]:
    checks = []
    count = 0
    for site in ("encoder31", "projector"):
        pair = next(p for p in pairs if p["site"] == site)
        direct = next(r for r in direct_records if r["site"] == site)
        for label in ("zero", "edited"):
            full = replay.full_model_lm_override(model, batch,
                                                 pair[label]["lm_input_full"].float(),
                                                 torch.float32, device=device)
            count += 1
            receipt(f"after_full_model_lm_override_{site}_{label}")
            replay.assert_direct_parity(full, direct[label]["replayed"])
            checks.append({"site": site, "alpha": pair["alpha"], "label": label,
                           "input_sha256": direct[label]["input_sha256"], "full": full})
    if count != 4:
        raise ValueError("Stage-B FP32 full-model override count changed")
    return checks, count


def phase_preflight(phase: str, *, stage_a: dict) -> tuple[dict, tuple, dict]:
    if phase not in PHASES:
        raise ValueError("Unknown Stage-B phase")
    bf = stage_a["arms"]["bf16"]
    result, details = replay.load_audited_native_attempt(
        bf["attempt"] / "runner", bf["attempt"] / "audit.json",
        expected_audit_sha256=bf["manifest"]["audit_sha256"])
    if sha(bf["attempt"] / "runner/result.json") != bf["manifest"]["result_sha256"] or \
            result["details_sha256"] != bf["result"]["details_sha256"]:
        raise ValueError("Audited BF16 result changed during Stage-B preflight")
    pairs = checked_pair_order(details)
    estimate = replay.estimate_replay_payload_bytes(details)
    # The primitive estimate counts both phase products together; headroom is
    # intentionally conservative. It is still below each phase's 1-GiB cap.
    if estimate["estimated_tensor_payload_bytes"] > MAX_B_BYTES * .9:
        raise ValueError("Stage-B tensor payload lacks output headroom")
    return details, pairs, estimate


def verify_bf16_stage_b_manifest(stage_a: dict, *, repo: Path = REPO,
                                 runs: Path = RUNS,
                                 manifest_path: Path = STAGE_B_BF16_MANIFEST) -> dict:
    """A later FP32 worker requires an externally committed, audited BF16 B arm."""
    committed_same(repo, manifest_path)
    strictly_later(repo, STAGE_A_MANIFESTS["fp32"], manifest_path)
    value = read_json(manifest_path)
    expected = {"status", "phase", "attempt_relative_path", "stage_a_bf16_manifest_sha256",
                "stage_a_fp32_manifest_sha256", "result_sha256", "audit_sha256",
                "model_work_elapsed_seconds"}
    if (set(value) != expected or value["status"] !=
            "audited_completed_old_fit_handoff_stage_b_bf16_direct_lm_only" or
            value["phase"] != "bf16_direct_lm" or
            value["stage_a_bf16_manifest_sha256"] != stage_a["arms"]["bf16"]["manifest_sha256"] or
            value["stage_a_fp32_manifest_sha256"] != stage_a["arms"]["fp32"]["manifest_sha256"]):
        raise ValueError("Stage-B BF16 external manifest does not bind Stage A")
    attempt = _one_attempt(runs, "bf16_direct_lm", value["attempt_relative_path"])
    result_path, audit_path = attempt / "result.json", attempt / "audit.json"
    if (result_path.is_symlink() or audit_path.is_symlink() or
            not result_path.is_file() or not audit_path.is_file() or
            sha(result_path) != value["result_sha256"] or sha(audit_path) != value["audit_sha256"]):
        raise ValueError("Stage-B BF16 result/audit bytes differ")
    result, audit = read_json(result_path), read_json(audit_path)
    if (result.get("status") != STATUS or result.get("phase") != "bf16_direct_lm" or
            result.get("actual_counts") != COUNTS["bf16_direct_lm"] or
            result.get("stage_a_manifest_sha256") != {
                arm: stage_a["arms"][arm]["manifest_sha256"] for arm in ("bf16", "fp32")} or
            audit.get("status") != "audited_completed_old_fit_handoff_stage_b_bf16_direct_lm_only" or
            audit.get("result_sha256") != value["result_sha256"]):
        raise ValueError("Stage-B BF16 auditor did not bind complete replay")
    elapsed = value["model_work_elapsed_seconds"]
    if (isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)) or
            not math.isfinite(elapsed) or not 0 < elapsed < CAP_SECONDS["bf16_direct_lm"] or
            result.get("elapsed_seconds") != elapsed):
        raise ValueError("Stage-B BF16 model-work time invalid")
    for stem in ("meta", "progress", "details"):
        filename = "progress.jsonl" if stem == "progress" else f"{stem}.json"
        if stem == "details":
            digest = result.get("details_sha256")
            if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest) or \
                    result.get("details_path") != f"blobs/{digest}.pt":
                raise ValueError("Stage-B BF16 details descriptor invalid")
            path = attempt / "blobs" / f"{digest}.pt"
        else:
            path = attempt / filename
        if path.is_symlink() or not path.is_file() or sha(path) != result.get(f"{stem}_sha256"):
            raise ValueError(f"Stage-B BF16 {stem} bytes differ")
    if stage_a_runner.directory_bytes(attempt) >= MAX_B_BYTES:
        raise ValueError("Stage-B BF16 output cap reached")
    return {"manifest_sha256": sha(manifest_path), "elapsed_seconds": float(elapsed),
            "attempt": attempt, "result": result, "audit": audit}


def _output_receipt(path: Path, value: dict) -> None:
    fixture.exclusive_json(path, value)
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def run_phase(phase: str, output: Path, *, model_factory=None,
              device: str = "mps") -> dict:
    """One bounded phase body for a future external launcher, one process only.

    The source-only CLI does not call this without an independently reviewed
    freeze. `model_factory` exists solely for CPU fake tests; the live path
    must pass None and must use the pinned local Voxtral snapshot.
    """
    if phase not in PHASES:
        raise ValueError("Unknown Stage-B phase")
    if output.exists():
        raise FileExistsError("Stage-B attempt already exists; no retry")
    started = time.monotonic()
    counts = {key: 0 for key in COUNTS[phase]}
    prior_seconds = 0.0
    # Admission is read-only. A missing/failed Stage-A result leaves no Stage-B
    # attempt directory and never reaches model construction.
    stage_a = verify_stage_a_manifests()
    if phase == "bf16_direct_lm":
        if (STAGE_B_BF16_MANIFEST.exists() or
                any((RUNS / p).is_dir() and list((RUNS / p).iterdir()) for p in PHASES)):
            raise ValueError("Stage-B prior attempt exists; no retry")
    else:
        previous = verify_bf16_stage_b_manifest(stage_a)
        if (RUNS / "fp32_replay").is_dir() and list((RUNS / "fp32_replay").iterdir()):
            raise ValueError("Stage-B FP32 prior attempt exists; no retry")
        prior_seconds = previous["elapsed_seconds"]
    prior_seconds += stage_a["stage_a_model_work_seconds"]
    if prior_seconds + CAP_SECONDS[phase] > MAX_TOTAL_SECONDS:
        raise ValueError("Cumulative cap lacks full Stage-B phase reservation")
    details, pairs, estimate = phase_preflight(phase, stage_a=stage_a)
    prepared = native.preflight_cpu()
    dtype = torch.bfloat16 if phase == "bf16_direct_lm" else torch.float32
    output.mkdir(parents=True, exist_ok=False)
    progress = output / "progress.jsonl"
    progress.touch(exist_ok=False)
    stack = ExitStack()
    memory = None
    try:
        meta = {"status": "started_old_fit_handoff_stage_b_replay_only", "phase": phase,
                "protocol_sha256": replay.PROTOCOL_SHA256,
                "script_sha256": sha(Path(__file__)),
                "replay_primitives_sha256": sha(Path(replay.__file__)),
                "stage_a_manifest_sha256": {arm: stage_a["arms"][arm]["manifest_sha256"]
                                             for arm in ("bf16", "fp32")},
                "stage_a_result_sha256": {arm: stage_a["arms"][arm]["manifest"]["result_sha256"]
                                           for arm in ("bf16", "fp32")},
                "stage_a_audit_sha256": {arm: stage_a["arms"][arm]["manifest"]["audit_sha256"]
                                          for arm in ("bf16", "fp32")},
                "source_rows": replay.ROWS, "cycle": 0, "alphas": list(replay.ALPHAS),
                "planned_counts": planned_counts(phase), "archive_estimate": estimate,
                "prior_model_work_seconds": prior_seconds,
                "process_cap_seconds": CAP_SECONDS[phase], "cumulative_cap_seconds": MAX_TOTAL_SECONDS,
                "max_mps_driver_bytes": MAX_MPS_DRIVER_BYTES, "max_output_bytes": MAX_B_BYTES,
                "fresh_v2_corpus_opened": False}
        _output_receipt(output / "meta.json", meta)

        def receipt(label: str) -> None:
            elapsed = time.monotonic() - started
            driver = int(torch.mps.driver_allocated_memory()) if device == "mps" else 0
            if memory is not None:
                driver = max(driver, memory.driver_max)
            replay.check_replay_limits(phase, elapsed_seconds=elapsed,
                                       previous_model_work_seconds=prior_seconds,
                                       mps_driver_bytes=driver,
                                       output_bytes=stage_a_runner.directory_bytes(output))
            fixture.append_line(progress, {"phase": label, "elapsed_seconds": elapsed,
                                           "mps_driver_bytes": driver})

        receipt("preflight_complete_before_model_load")
        if device == "mps":
            if not torch.backends.mps.is_available():
                raise RuntimeError("MPS unavailable")
            recommended = int(torch.mps.recommended_max_memory())
            if recommended <= 0:
                raise RuntimeError("MPS recommended memory unavailable")
            torch.mps.set_per_process_memory_fraction(min(1.0, MAX_MPS_DRIVER_BYTES / recommended))
            memory = stack.enter_context(fixture.MemorySampler(torch))
        if model_factory is None:
            from transformers import VoxtralForConditionalGeneration
            model = VoxtralForConditionalGeneration.from_pretrained(
                str(fixture.SNAPSHOT), local_files_only=True, dtype=dtype,
                low_cpu_mem_usage=True, device_map="mps", attn_implementation="sdpa").eval()
        else:
            if device != "cpu":
                raise ValueError("Injected model factory is CPU-test-only")
            model = model_factory(dtype)
        model.requires_grad_(False)
        layout = prior.parameter_layout(model, "bf16" if dtype == torch.bfloat16 else "fp32")
        _output_receipt(output / "model_layout.json", layout)
        if not layout["valid"]:
            raise ValueError("Stage-B parameter layout invalid")
        receipt("model_loaded")
        batch = prepared["batches"]["beaver"]
        records = {"source_replay": None, "direct_lm": None, "lm_override_validation": None}
        if phase == "fp32_replay":
            source, used = fp32_source_replays(model, batch, pairs, details["probes"],
                                               device=device, receipt=receipt)
            records["source_replay"] = source
            counts.update(used)
        direct, count = direct_replay(model, batch, pairs, dtype, device=device,
                                      parity_native=phase == "bf16_direct_lm", receipt=receipt)
        records["direct_lm"] = direct
        counts["direct_lm_forwards"] = count
        if phase == "fp32_replay":
            validations, count = fp32_lm_validation(model, batch, pairs, direct,
                                                     device=device, receipt=receipt)
            records["lm_override_validation"] = validations
            counts["full_model_override_forwards"] = count
        if counts != COUNTS[phase]:
            raise ValueError("Stage-B realized forward/VJP counts differ")
        receipt("all_planned_model_work_complete")
        remaining = MAX_B_BYTES - stage_a_runner.directory_bytes(output) - 1_048_576
        if remaining <= 0:
            raise OSError("Stage-B output cap lacks details headroom")
        with (output / "details.pt").open("xb") as raw:
            limited = stage_a_runner.LimitedWriter(raw, remaining)
            torch.save(records, limited)
            limited.flush()
            os.fsync(limited.fileno())
        details_sha, details_path = stage_a_runner.freeze_content_addressed(output / "details.pt")
        receipt("details_saved")
        elapsed = time.monotonic() - started
        driver = int(torch.mps.driver_allocated_memory()) if device == "mps" else 0
        if memory is not None:
            driver = max(driver, memory.driver_max)
        replay.check_replay_limits(phase, elapsed_seconds=elapsed,
                                   previous_model_work_seconds=prior_seconds,
                                   mps_driver_bytes=driver,
                                   output_bytes=stage_a_runner.directory_bytes(output))
        result = {"status": STATUS, "phase": phase, "protocol_sha256": replay.PROTOCOL_SHA256,
                  "script_sha256": sha(Path(__file__)),
                  "stage_a_manifest_sha256": meta["stage_a_manifest_sha256"],
                  "stage_a_result_sha256": meta["stage_a_result_sha256"],
                  "stage_a_audit_sha256": meta["stage_a_audit_sha256"],
                  "meta_sha256": sha(output / "meta.json"),
                  "model_layout_sha256": sha(output / "model_layout.json"),
                  "progress_sha256": sha(progress),
                  "details_sha256": details_sha, "details_path": details_path,
                  "details_bytes": (output / details_path).stat().st_size,
                  "planned_counts": planned_counts(phase), "actual_counts": counts,
                  "prior_model_work_seconds": prior_seconds, "elapsed_seconds": elapsed,
                  "peak_mps_driver_bytes": driver,
                  "scope": "posthoc_one_old_fit_case_replay_only_no_security_claim"}
        payload = json.dumps(result, allow_nan=False).encode()
        if stage_a_runner.directory_bytes(output) + len(payload) + 4096 >= MAX_B_BYTES:
            raise OSError("Stage-B output cap reached before result")
        _output_receipt(output / "result.json", result)
        return result
    except Exception as exc:
        _output_receipt(output / "error.json", {"status": "invalid_stage_b_replay_attempt",
                        "phase": phase, "error_type": type(exc).__name__, "error": str(exc),
                        "elapsed_seconds": time.monotonic() - started,
                        "prior_model_work_seconds": prior_seconds,
                        "progress_sha256": sha(progress)})
        raise
    finally:
        stack.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.validate_only and args.output_dir:
        parser.error("--output-dir requires --run")
    if args.run:
        required_live_freeze()
        raise AssertionError("unreachable until separate Stage-B launch gate is reviewed")
    stage_a = verify_stage_a_manifests()
    _, _, estimate = phase_preflight(args.phase, stage_a=stage_a)
    print(json.dumps({"status": "validated_stage_b_cpu_only", "phase": args.phase,
                      "stage_a_manifest_sha256": {arm: stage_a["arms"][arm]["manifest_sha256"]
                                                   for arm in ("bf16", "fp32")},
                      "stage_a_model_work_seconds": stage_a["stage_a_model_work_seconds"],
                      "planned_counts": planned_counts(args.phase), "estimate": estimate}, indent=2))


if __name__ == "__main__":
    main()
