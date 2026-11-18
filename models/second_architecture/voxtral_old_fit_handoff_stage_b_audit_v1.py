"""CPU-only independent saved-data audit for the prospective Voxtral Stage B.

No model is constructed here. A successful audit establishes byte provenance,
saved-tensor consistency, and arithmetic; it cannot attest that forwards or
backward calls actually occurred. The external launcher must bind chronology.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
PROTOCOL = HERE / "VOXTRAL_OLD_FIT_NUMERICAL_HANDOFF_PROSPECTIVE_V1.md"
DRIVER = HERE / "voxtral_old_fit_handoff_stage_b_runner_v1.py"
PRIMITIVES = HERE / "voxtral_old_fit_handoff_replay_stage_b_v1.py"
PROTOCOL_SHA = "61a67ac87a4e4e17cb0b1f7ad972f25fda4e0d90e0b70798e7144b3fd6f2484b"
DRIVER_SHA = "f685a920a61bb1885ce86df44e41899adca886f42b90f5a6577ae7e6cffaf71e"
PRIMITIVES_SHA = "bef337c90a024e034e60818761b4be79371ca0767f83d28efe29269ba0cc89e1"
PHASES = ("bf16_direct_lm", "fp32_replay")
SITES = ("encoder31", "projector")
ALPHAS = (.00625, -.00625, .025, -.025, .1, -.1)
ROWS = {"encoder31": list(range(96, 108)), "projector": [24, 25, 26]}
SHAPES = {"encoder31": (1, 1500, 1280), "projector": (375, 3072)}
LMS = ("lm0", "lm7", "lm14", "lm22", "lm29")
TARGETS = {"encoder31": {"projector_word": 4, "lm_input_word": 4, "lm0": 4,
                          "lm14": 4, "lm29": 16, "lm29_raw": 4},
           "projector": {"lm_input_word": 4, "lm0": 4, "lm14": 4,
                         "lm29": 16, "lm29_raw": 4}}
COUNTS = {"bf16_direct_lm": {"direct_lm_forwards": 24, "source_pair_forwards": 0,
                              "source_parity_forwards": 0, "source_vjps": 0,
                              "full_model_override_forwards": 0},
          "fp32_replay": {"direct_lm_forwards": 24, "source_pair_forwards": 24,
                          "source_parity_forwards": 6, "source_vjps": 68,
                          "full_model_override_forwards": 4}}
CAP_SECONDS = {"bf16_direct_lm": 600, "fp32_replay": 1800}
BYTE_CAP = 1024**3
DRIVER_CAP = 30 * 1024**3
TOTAL_SECONDS = 4500
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
REPO = HERE.parents[1]
FAILED_MANIFEST = HERE / "VOXTRAL_STAGE_A_BF16_FAILED_ATTEMPT_MANIFEST_V1.json"
POSTHOC_RECEIPT = HERE / "VOXTRAL_STAGE_A_BF16_POSTHOC_AUDIT_RECEIPT_V1.json"
POSTHOC_AUDIT = HERE / "VOXTRAL_STAGE_A_BF16_POSTHOC_AUDIT_V2.json"
POSTHOC_REVIEW = HERE / "VOXTRAL_STAGE_A_BF16_POSTHOC_AUDIT_RECEIPT_INDEPENDENT_REVIEW_V1.md"
POSTHOC_RESULT_REVIEW = HERE / "VOXTRAL_STAGE_A_BF16_POSTHOC_AUDIT_RESULT_INDEPENDENT_REVIEW_V1.md"
ORIGINAL_FREEZE = HERE / "VOXTRAL_OLD_FIT_HANDOFF_STAGE_A_EXTERNAL_FREEZE_V1.json"
STAGE_A_RUNS = HERE / "runs/voxtral_old_fit_handoff_stage_a_launch_v1"
STAGE_A_RUNNER_SHA = "9f8f6b52d3e218acc496b78b35247cfdf322672cd487e585b92d58f08205f1dc"
STAGE_A_NATIVE_SHA = "dc63b8c10ef605e1cc8538e3d37d8d2531e50ca7a46140ee1ee551054e3de978"
STAGE_A_V1_AUDITOR_SHA = "4f800ed604295ae0e4fbd53c893fccd14b8968badb43b835e6ec975e49b0cf3d"
STAGE_A_V2_AUDITOR_SHA = "8d5f8b23bd32add8464bd976d42cd0225fbc5ee86510adc441b714de5a8fd275"


def need(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha(value: torch.Tensor) -> str:
    value = value.detach().cpu().contiguous()
    if value.dtype == torch.bfloat16:
        value = value.view(torch.uint16)
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_bytes(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    need(isinstance(value, dict), f"{path.name} must be a JSON object")
    return value


def committed_same(path: Path, *, repo: Path = REPO) -> None:
    need(path.is_file() and not path.is_symlink(), f"Committed evidence absent: {path.name}")
    relative = path.resolve().relative_to(repo.resolve()).as_posix()
    result = subprocess.run(["git", "-C", str(repo), "show", f"HEAD:{relative}"],
                            capture_output=True, check=False)
    need(result.returncode == 0 and path.read_bytes() == result.stdout,
         f"Committed evidence changed: {path.name}")


def first_add(path: Path, *, repo: Path = REPO) -> str:
    relative = path.resolve().relative_to(repo.resolve()).as_posix()
    result = subprocess.run(["git", "-C", str(repo), "log", "--diff-filter=A",
                             "--format=%H", "HEAD", "--", relative],
                            capture_output=True, text=True, check=False)
    lines = result.stdout.splitlines()
    need(result.returncode == 0 and len(lines) == 1 and
         re.fullmatch(r"[0-9a-f]{40}", lines[0]) is not None,
         f"Receipt first-add history invalid: {path.name}")
    added = subprocess.run(["git", "-C", str(repo), "show", f"{lines[0]}:{relative}"],
                           capture_output=True, check=False)
    need(added.returncode == 0 and path.read_bytes() == added.stdout,
         f"Receipt changed after first-add: {path.name}")
    return lines[0]


def strictly_later(earlier: Path, later: Path, *, repo: Path = REPO) -> None:
    before, after = first_add(earlier, repo=repo), first_add(later, repo=repo)
    need(before != after and subprocess.run(["git", "-C", str(repo), "merge-base",
                                             "--is-ancestor", before, after],
                                            capture_output=True, check=False).returncode == 0,
         f"Receipt chronology differs: {later.name}")


def verify_manifest_files(manifest: dict, *, here: Path = HERE,
                          repo: Path = REPO) -> dict[str, Path]:
    """Check every raw byte, including the two local, uncommitted tensors."""
    need(manifest.get("schema") == "voxtral_stage_a_bf16_failed_attempt_manifest_v1" and
         manifest.get("status") == "frozen_auditor_rejected_no_scientific_interpretation" and
         manifest.get("arm") == "bf16" and
         manifest.get("local_large_sidecars_required") is True and
         isinstance(manifest.get("files"), list) and len(manifest["files"]) == 12,
         "Original BF16 failed-attempt manifest schema differs")
    attempt_id = manifest.get("attempt_id")
    need(attempt_id == "20261118T131755298331Z", "Original BF16 attempt ID differs")
    prefix = f"runs/voxtral_old_fit_handoff_stage_a_launch_v1/bf16/{attempt_id}/"
    paths = {}
    for item in manifest["files"]:
        need(isinstance(item, dict) and set(item) ==
             {"path", "bytes", "sha256", "git_committed"},
             "Original BF16 file record schema differs")
        relative = item["path"]
        need(isinstance(relative, str) and relative.startswith(prefix) and
             ".." not in Path(relative).parts and relative not in paths and
             type(item["bytes"]) is int and item["bytes"] >= 0 and
             isinstance(item["sha256"], str) and HEX64.fullmatch(item["sha256"]) is not None and
             type(item["git_committed"]) is bool,
             "Original BF16 file record invalid")
        path = here / relative
        need(path.is_file() and not path.is_symlink() and
             path.stat().st_size == item["bytes"] and sha(path) == item["sha256"],
             f"Original BF16 raw file differs: {relative}")
        if item["git_committed"]:
            committed_same(path, repo=repo)
        else:
            need(path.suffix == ".pt" and "/runner/blobs/" in relative,
                 "Unexpected uncommitted BF16 raw file")
        paths[relative] = path
    need(sum(not row["git_committed"] for row in manifest["files"]) == 2,
         "Original BF16 local sidecar count differs")
    attempt = here / prefix
    actual = {p for p in attempt.rglob("*") if p.is_file()}
    need(actual == set(paths.values()) and
         sum(p.stat().st_size for p in actual) < int(1.5 * 1024**3),
         "Original BF16 attempt file set or output cap differs")
    return paths


def verify_bf16_posthoc() -> dict:
    """Preserve the original failed launch while binding its later CPU audit."""
    for path in (FAILED_MANIFEST, POSTHOC_RECEIPT, POSTHOC_AUDIT,
                 POSTHOC_REVIEW, POSTHOC_RESULT_REVIEW, ORIGINAL_FREEZE):
        committed_same(path)
    strictly_later(ORIGINAL_FREEZE, FAILED_MANIFEST)
    strictly_later(FAILED_MANIFEST, POSTHOC_AUDIT)
    strictly_later(POSTHOC_AUDIT, POSTHOC_RESULT_REVIEW)
    strictly_later(POSTHOC_RESULT_REVIEW, POSTHOC_RECEIPT)
    strictly_later(POSTHOC_RECEIPT, POSTHOC_REVIEW)
    need("Decision: GO for receipt provenance; FP32 HOLD pending launcher review and freeze."
         in POSTHOC_REVIEW.read_text(), "Posthoc BF16 independent review lacks GO marker")
    manifest, receipt, audit = (read_json(p) for p in
                                (FAILED_MANIFEST, POSTHOC_RECEIPT, POSTHOC_AUDIT))
    paths = verify_manifest_files(manifest)
    attempt = STAGE_A_RUNS / "bf16" / manifest["attempt_id"]
    need(sorted(p for p in (STAGE_A_RUNS / "bf16").iterdir() if p.is_dir()) == [attempt],
         "Original BF16 attempt was retried")
    finish_path = attempt / "launch_finish.json"
    result_path = attempt / "runner/result.json"
    layout_path = attempt / "runner/model_layout.json"
    finish, result = read_json(finish_path), read_json(result_path)
    need(finish.get("status") == "launch_failed" and finish.get("arm") == "bf16" and
         finish.get("runner_returncode") == 0 and finish.get("auditor_returncode") == 1 and
         finish.get("audit_sha256") is None and
         result.get("status") == "completed_old_fit_handoff_stage_a_native_only" and
         result.get("arm") == "bf16" and result.get("script_sha256") == STAGE_A_RUNNER_SHA and
         result.get("native_source_sha256") == STAGE_A_NATIVE_SHA and
         result.get("protocol_sha256") == PROTOCOL_SHA and
         result.get("planned_counts") == result.get("actual_counts") and
         result.get("actual_counts", {}).get("native_pairs") == 36 and
         result.get("actual_counts", {}).get("vjps") == 44,
         "Original BF16 failed-launch/native receipt differs")
    expected = {
        "schema": "voxtral_stage_a_bf16_posthoc_audit_receipt_v1",
        "status": "posthoc_saved_tensor_audit_passed_original_launch_failed",
        "arm": "bf16", "attempt_id": manifest["attempt_id"],
        "original_launch_status": "launch_failed",
        "failed_attempt_manifest_sha256": sha(FAILED_MANIFEST),
        "original_freeze_sha256": sha(ORIGINAL_FREEZE),
        "original_launch_finish_sha256": sha(finish_path),
        "original_result_sha256": sha(result_path),
        "original_layout_sha256": sha(layout_path),
        "original_details_sha256": result["details_sha256"],
        "original_probes_sha256": result["probes_sha256"],
        "original_v1_auditor_sha256": STAGE_A_V1_AUDITOR_SHA,
        "v2_auditor_sha256": STAGE_A_V2_AUDITOR_SHA,
        "v2_auditor_source_review_commit": "a8b6c248f285485702ef46dca08f91012e51dbbf",
        "posthoc_audit_sha256": sha(POSTHOC_AUDIT),
        "model_work_elapsed_seconds": finish["model_work_elapsed_seconds"],
    }
    need(receipt == expected and audit.get("status") ==
         "audited_completed_old_fit_handoff_stage_a_native_only" and
         audit.get("arm") == "bf16" and audit.get("auditor_sha256") == STAGE_A_V2_AUDITOR_SHA and
         audit.get("result_sha256") == sha(result_path) and
         audit.get("details_sha256") == result["details_sha256"] and
         audit.get("probes_sha256") == result["probes_sha256"] and
         audit.get("audited_native_pairs") == 36 and
         audit.get("audited_patch_forwards") == 72 and
         audit.get("audited_vjps") == 44 and
         audit.get("audited_lm0_control_cells") == 24 and
         audit.get("observed_progress", {}).get("progress_lines") == 214 and
         audit.get("scope") ==
         "saved_tensor_and_receipt_arithmetic_only_no_model_execution_attestation",
         "Posthoc BF16 receipt/audit does not bind preserved failed launch")
    detail_path = attempt / "runner" / result["details_path"]
    probe_path = attempt / "runner" / result["probes_path"]
    need(detail_path in paths.values() and probe_path in paths.values() and
         sha(detail_path) == result["details_sha256"] and
         sha(probe_path) == result["probes_sha256"],
         "Posthoc BF16 sidecar descriptors differ")
    details = torch.load(detail_path, map_location="cpu", weights_only=True)
    probes = torch.load(probe_path, map_location="cpu", weights_only=True)
    check_probes(probes)
    need(all(torch.equal(details["probes"][k], probes[k]) for k in probes),
         "Posthoc BF16 detail/probe mismatch")
    return {"native": native_pairs(details), "probes": probes,
            "elapsed_seconds": float(receipt["model_work_elapsed_seconds"]),
            "receipt_sha256": sha(POSTHOC_RECEIPT),
            "result_sha256": sha(result_path), "audit_sha256": sha(POSTHOC_AUDIT)}


def _rademacher(label: str, width: int, count: int) -> torch.Tensor:
    seed = int.from_bytes(hashlib.sha256(bytes.fromhex(PROTOCOL_SHA) +
                                         label.encode("ascii")).digest()[:8], "big") % (2**63)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    signs = torch.randint(0, 2, (width, count), generator=generator, dtype=torch.int8)
    matrix = signs.to(torch.float64).mul_(2).sub_(1)
    q, r = torch.linalg.qr(matrix, mode="reduced")
    q *= torch.sign(torch.diag(r)).unsqueeze(0)
    return q.contiguous()


def check_probes(probes: dict) -> None:
    expected = {"lm": _rademacher("lm_residual_16", 3072, 16),
                "projector_word": _rademacher("projector_word_4", 9216, 4),
                "lm_input_word": _rademacher("lm_input_word_4", 9216, 4)}
    need(isinstance(probes, dict) and set(probes) == set(expected),
         "Fixed probe bank schema differs")
    for name, correct in expected.items():
        equal(probes[name], correct, f"Fixed probe bank {name}")


def check_layout(layout: dict, arm: str) -> None:
    rows = layout.get("parameters")
    need(layout.get("valid") is True and layout.get("arm") == arm and
         layout.get("errors") == [] and layout.get("named_parameter_count") == 762 and
         isinstance(rows, list) and len(rows) == 762,
         "Stage-B parameter layout receipt invalid")
    names = [row.get("name") for row in rows]
    position_name = "model.audio_tower.embed_positions.weight"
    need(len(set(names)) == 762 and position_name in names,
         "Stage-B parameter names missing or duplicated")
    counts = {}
    total = 0
    for row in rows:
        name, device, dtype = row["name"], row["device"], row["dtype"]
        expected_dtype = "torch.float32" if arm == "fp32" or name == position_name else "torch.bfloat16"
        need(dtype == expected_dtype and device in ("mps", "mps:0") and
             isinstance(row.get("shape"), list) and
             all(type(v) is int and v > 0 for v in row["shape"]) and
             math.prod(row["shape"]) == row.get("numel"),
             "Stage-B parameter dtype/device/shape differs")
        if name == position_name:
            need(row["shape"] == [1500, 1280], "Stage-B position table shape differs")
        counts[f"{dtype}/mps"] = counts.get(f"{dtype}/mps", 0) + 1
        total += row["numel"]
    need(total == layout.get("parameter_count") and
         counts == layout.get("dtype_device_counts") and
         layout.get("expected_fp32_exception") == (position_name if arm == "bf16" else None),
         "Stage-B parameter totals differ")


def close(a: float, b: float, name: str) -> None:
    need(type(a) in (int, float) and type(b) in (int, float) and
         math.isfinite(a) and math.isfinite(b) and
         math.isclose(a, b, rel_tol=2e-5, abs_tol=2e-5), f"{name} differs")


def tensor(value, shape, dtype, name):
    need(isinstance(value, torch.Tensor) and tuple(value.shape) == tuple(shape) and
         value.dtype == dtype and bool(torch.isfinite(value).all()),
         f"{name} shape, dtype, or finiteness differs")
    return value


def equal(a, b, name):
    need(isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor) and
         a.shape == b.shape and a.dtype == b.dtype and torch.equal(a, b),
         f"{name} bitwise parity differs")


def check_lm(trace: dict, dtype: torch.dtype, *, mode: str) -> None:
    need(isinstance(trace, dict) and trace.get("mode") == mode and
         set(trace.get("lm_terminal", {})) == set(LMS), "LM trace mode/stages differ")
    for stage in LMS:
        tensor(trace["lm_terminal"][stage], (3072,), dtype, f"{mode} {stage}")
    tensor(trace.get("first_answer_logits"), (131072,), dtype, f"{mode} logits")
    need(isinstance(trace.get("input_sha256"), str) and
         HEX64.fullmatch(trace["input_sha256"]) is not None, "LM input hash invalid")


def check_lm_parity(reference: dict, observed: dict, name: str) -> None:
    for stage in LMS:
        equal(reference["lm_terminal"][stage], observed["lm_terminal"][stage], f"{name} {stage}")
    equal(reference["first_answer_logits"], observed["first_answer_logits"], f"{name} logits")


def native_pairs(details: dict) -> tuple[dict, ...]:
    observations = details.get("observations")
    need(isinstance(observations, dict) and set(observations) == set(SITES),
         "Stage-A observations missing")
    out = []
    for site in SITES:
        rows = observations[site]
        need(isinstance(rows, list) and len(rows) == 18, "Stage-A site grid incomplete")
        first = [row for row in rows if row.get("cycle") == 0]
        need(tuple(row.get("alpha") for row in first) == ALPHAS,
             "Stage-A cycle-0 order differs")
        for row in first:
            out.append({"site": site, "alpha": row["alpha"],
                        "zero": row["zero"].get("saved", row["zero"]),
                        "edited": row["edited"].get("saved", row["edited"])})
    return tuple(out)


def check_archive_estimate(native: tuple[dict, ...], reported: dict) -> None:
    source_bytes = sum(pair[label]["source_full_post"][pair["site"]].numel() *
                       pair[label]["source_full_post"][pair["site"]].element_size()
                       for pair in native for label in ("zero", "edited"))
    lm_bytes = sum(pair[label]["lm_input_full"].numel() *
                   pair[label]["lm_input_full"].element_size()
                   for pair in native for label in ("zero", "edited"))
    gradients = sum(sum(TARGETS[site].values()) * len(ROWS[site]) *
                    SHAPES[site][-1] * 4 for site in SITES)
    stage_vectors = (24 + 6 + 24 + 24 + 4) * len(LMS) * 3072 * 4
    expected = {"source_pair_bytes": source_bytes, "lm_input_pair_bytes": lm_bytes,
                "selected_vjp_bytes": gradients,
                "stage_vector_upper_bytes": stage_vectors,
                "estimated_tensor_payload_bytes": source_bytes + lm_bytes + gradients + stage_vectors,
                "per_process_output_cap_bytes": BYTE_CAP}
    need(reported == expected and expected["estimated_tensor_payload_bytes"] <= BYTE_CAP * .9,
         "Stage-B archive estimate differs or exceeds reserved headroom")


def _odd_even(rows: list[dict], key: str) -> dict:
    output = {}
    for magnitude in (.00625, .025, .1):
        plus = next(r for r in rows if r["alpha"] == magnitude)
        minus = next(r for r in rows if r["alpha"] == -magnitude)
        p = torch.as_tensor(plus[key], dtype=torch.float64)
        m = torch.as_tensor(minus[key], dtype=torch.float64)
        output[str(magnitude)] = {"odd": ((p - m) / 2).tolist(),
                                  "even": ((p + m) / 2).tolist()}
    return output


def check_direct(records: list, native: tuple[dict, ...], phase: str,
                 probes: dict | None = None) -> dict:
    dtype = torch.bfloat16 if phase == "bf16_direct_lm" else torch.float32
    need(isinstance(records, list) and len(records) == len(native) == 12,
         "Direct-LM pair count differs")
    deltas = []
    for record, pair in zip(records, native):
        need((record.get("site"), record.get("cycle"), record.get("alpha")) ==
             (pair["site"], 0, pair["alpha"]), "Direct-LM signed pair order differs")
        for label in ("zero", "edited"):
            item = record[label]
            archived = pair[label]["lm_input_full"]
            tensor(archived, (1, 396, 3072), torch.bfloat16, "archived LM input")
            equal(item["input"], archived, "direct replay exact archived LM input")
            need(item["input_sha256"] == tensor_sha(archived), "Direct input hash differs")
            observed = item["replayed"]
            check_lm(observed, dtype, mode="direct_lm")
            need(observed["input_sha256"] == tensor_sha(archived.to(dtype)),
                 "Direct model-reported input hash differs")
            if phase == "bf16_direct_lm":
                check_lm_parity(pair[label], observed, "BF16 direct/native")
        dz = record["zero"]["replayed"]["lm_terminal"]
        de = record["edited"]["replayed"]["lm_terminal"]
        stages = {}
        for stage in LMS:
            diff = de[stage].float() - dz[stage].float()
            metrics = {"full_delta_l2": float(torch.linalg.vector_norm(diff.double())),
                       "changed_full_coordinates": int((diff != 0).sum())}
            if probes is not None:
                n = 16 if stage == "lm29" else 4
                metrics["projected_delta"] = (diff @ probes["lm"].float()[:, :n]).tolist()
            stages[stage] = metrics
        deltas.append({"site": pair["site"], "alpha": pair["alpha"], "stages": stages})
    odd_even = {}
    if probes is not None:
        for site in SITES:
            odd_even[site] = {stage: _odd_even(
                [{"alpha": row["alpha"], "measured": row["stages"][stage]["projected_delta"]}
                 for row in deltas if row["site"] == site], "measured") for stage in LMS}
    return {"pairs": 12, "forwards": 24, "deltas": deltas, "odd_even": odd_even}


def source_projection(saved: dict, stage: str, probes: dict) -> torch.Tensor:
    if stage == "projector_word":
        return saved["projector_word"].float() @ probes[stage].float()
    if stage == "lm_input_word":
        return saved["lm_input_word"].float() @ probes[stage].float()
    value = saved["lm_terminal"]["lm29" if stage == "lm29_raw" else stage].float()
    if stage == "lm29_raw":
        return value[:4]
    return value @ probes["lm"].float()[:, :16 if stage == "lm29" else 4]


def check_source_trace(saved: dict, site: str) -> None:
    need(isinstance(saved, dict) and isinstance(saved.get("source_sha256"), str) and
         HEX64.fullmatch(saved["source_sha256"]) is not None,
         "Source trace hash missing")
    tensor(saved.get("projector_word"), (9216,), torch.float32, "projector word")
    tensor(saved.get("lm_input_word"), (9216,), torch.float32, "LM input word")
    need(set(saved.get("lm_terminal", {})) == set(LMS), "Source LM stages differ")
    for stage in LMS:
        tensor(saved["lm_terminal"][stage], (3072,), torch.float32, f"source {stage}")
    tensor(saved.get("first_answer_logits"), (131072,), torch.float32, "source logits")
    positions = saved.get("audio_placeholder_positions")
    need(isinstance(positions, list) and len(positions) == 375 and
         all(type(p) is int and 0 <= p < 396 for p in positions) and
         len(set(positions)) == 375, "Source audio placeholder positions differ")
    equal(saved["projector_word"], saved["lm_input_word"],
          "Source projector to LM word handoff")


def check_source(source: dict, native: tuple[dict, ...], probes: dict) -> dict:
    need(isinstance(source, dict) and set(source) == {"parity", "vjps", "pairs"},
         "Source replay schema differs")
    need(set(source["parity"]) == set(SITES) and
         set(source["vjps"]) == set(SITES) and len(source["pairs"]) == 12,
         "Source parity/VJP/pair matrix incomplete")
    analysis = []
    for site in SITES:
        parity = source["parity"][site]
        need(set(parity) == {"clean", "repeat", "self"}, "Source parity trio incomplete")
        for value in parity.values():
            check_source_trace(value, site)
        for name in ("repeat", "self"):
            p = parity[name]
            need(p["source_sha256"] == parity["clean"]["source_sha256"] and
                 p["audio_placeholder_positions"] == parity["clean"]["audio_placeholder_positions"],
                 "Source parity metadata differs")
            for field in ("projector_word", "lm_input_word", "first_answer_logits"):
                equal(parity["clean"][field], p[field], f"source {site} {name} {field}")
            for stage in LMS:
                equal(parity["clean"]["lm_terminal"][stage], p["lm_terminal"][stage],
                      f"source {site} {name} {stage}")
        gradients = source["vjps"][site]
        need(set(gradients) == set(TARGETS[site]), "Source VJP stages differ")
        width = SHAPES[site][-1]
        for stage, n in TARGETS[site].items():
            tensor(gradients[stage], (n, len(ROWS[site]), width), torch.float32,
                   f"{site}/{stage} VJP")
        first = next(p for p in native if p["site"] == site)
        need(parity["clean"]["source_sha256"] ==
             tensor_sha(first["zero"]["source_full_post"][site]),
             "Source parity input is not archived clean source")
    for record, pair in zip(source["pairs"], native):
        site = pair["site"]
        need((record.get("site"), record.get("cycle"), record.get("alpha")) ==
             (site, 0, pair["alpha"]), "Source replay signed pair order differs")
        source_zero = pair["zero"]["source_full_post"][site]
        source_edit = pair["edited"]["source_full_post"][site]
        tensor(source_zero, SHAPES[site], torch.bfloat16, "archived source zero")
        tensor(source_edit, SHAPES[site], torch.bfloat16, "archived source edit")
        first_clean = next(p for p in native if p["site"] == site)["zero"]["source_full_post"][site]
        equal(source_zero, first_clean, "source clean input across signed steps")
        diff = source_edit.float() - source_zero.float()
        allowed = torch.zeros_like(diff, dtype=torch.bool)
        if site == "encoder31":
            allowed[0, ROWS[site]] = True
            selected = diff[0, ROWS[site]]
        else:
            allowed[ROWS[site]] = True
            selected = diff[ROWS[site]]
        need(bool((diff != 0).any()) and not bool(((diff != 0) & ~allowed).any()),
             "Source exact edit missing or outside frozen rows")
        equal(record["effective_edit"], selected, "Source saved effective edit")
        for label, archived in (("zero", source_zero), ("edited", source_edit)):
            item = record[label]
            equal(item["input"], archived, "Source exact archived input")
            need(item["input_sha256"] == tensor_sha(archived), "Source input hash differs")
            trace = item["trace"]
            check_source_trace(trace, site)
            need(trace["source_sha256"] == tensor_sha(archived),
                 "Source model-reported input hash differs")
            need(trace["audio_placeholder_positions"] ==
                 parity_positions(source["parity"][site]), "Source placeholder positions differ")
        zero, edited = record["zero"]["trace"], record["edited"]["trace"]
        close(record["terminal_delta_l2"],
              float(torch.linalg.vector_norm((edited["lm_terminal"]["lm29"].double() -
                                             zero["lm_terminal"]["lm29"].double()))),
              "Source terminal delta")
        if site == "projector":
            equal(edited["projector_word"] - zero["projector_word"],
                  selected.reshape(-1), "Projector self derivative identity")
        cells = {}
        for stage in TARGETS[site]:
            measured = source_projection(edited, stage, probes) - source_projection(zero, stage, probes)
            predicted = (source["vjps"][site][stage].double() * selected.double()).sum((1, 2))
            need(torch.isfinite(measured).all() and torch.isfinite(predicted).all(),
                 "Source projection/VJP contraction nonfinite")
            residual = measured.double() - predicted
            cells[stage] = {"measured": measured.tolist(), "predicted": predicted.tolist(),
                            "measured_l2": float(torch.linalg.vector_norm(measured.double())),
                            "residual_l2": float(torch.linalg.vector_norm(residual))}
        analysis.append({"site": site, "alpha": pair["alpha"], "stages": cells})
    summaries = {}
    for site in SITES:
        summaries[site] = {}
        for stage in TARGETS[site]:
            paired_zeros = torch.stack([
                source_projection(record["zero"]["trace"], stage, probes).double()
                for record in source["pairs"] if record["site"] == site])
            need(paired_zeros.shape[0] == 6, "Source paired-zero grid incomplete")
            repeat_spread = float(torch.cdist(paired_zeros, paired_zeros).max())
            resolution = max(3 * repeat_spread, 1e-6)
            rows = [cell for cell in analysis if cell["site"] == site]
            for row in rows:
                measured_l2 = row["stages"][stage]["measured_l2"]
                row["stages"][stage]["resolved"] = measured_l2 > resolution
                row["stages"][stage]["relative_l2_error_if_resolved"] = (
                    row["stages"][stage]["residual_l2"] / measured_l2
                    if measured_l2 > resolution else None)
            summaries[site][stage] = {"zero_repeat_spread_l2": repeat_spread,
                                      "resolution_l2": resolution,
                                      "measured_odd_even": _odd_even(
                                          [{"alpha": r["alpha"],
                                            "measured": r["stages"][stage]["measured"]}
                                           for r in rows], "measured")}
    return {"pairs": 12, "pair_forwards": 24, "parity_forwards": 6,
            "vjps": sum(sum(TARGETS[site].values()) for site in SITES),
            "projector_self_identity_pairs": 6, "cells": analysis,
            "stage_summary": summaries}


def parity_positions(parity: dict) -> list[int]:
    return parity["clean"]["audio_placeholder_positions"]


def check_overrides(validations: list, direct: list) -> dict:
    need(isinstance(validations, list) and len(validations) == 4,
         "FP32 full-model override count differs")
    expected = [(site, label) for site in SITES for label in ("zero", "edited")]
    first = {site: next(r for r in direct if r["site"] == site) for site in SITES}
    need([(v.get("site"), v.get("label")) for v in validations] == expected,
         "FP32 override order differs")
    for value in validations:
        target = first[value["site"]][value["label"]]
        need(value["alpha"] == first[value["site"]]["alpha"] and
             value["input_sha256"] == target["input_sha256"],
             "FP32 override replay input hash differs")
        check_lm(value["full"], torch.float32, mode="full_model_lm_override")
        need(value["full"]["input_sha256"] == target["replayed"]["input_sha256"],
             "FP32 full model input hash differs")
        check_lm_parity(target["replayed"], value["full"], "FP32 direct/full override")
    return {"full_model_override_forwards": 4}


def expected_progress(phase: str) -> list[str]:
    phases = ["preflight_complete_before_model_load", "model_loaded"]
    if phase == "fp32_replay":
        for site in SITES:
            phases.extend((f"after_source_{site}_clean_graph",
                           f"after_source_{site}_clean_repeat",
                           f"after_source_{site}_self",
                           f"after_source_{site}_vjps"))
            for alpha in ALPHAS:
                phases.extend((f"after_source_{site}_{alpha}_zero",
                               f"after_source_{site}_{alpha}_edited"))
    for site in SITES:
        for alpha in ALPHAS:
            phases.extend((f"after_direct_lm_{site}_{alpha}_zero",
                           f"after_direct_lm_{site}_{alpha}_edited"))
    if phase == "fp32_replay":
        for site in SITES:
            phases.extend((f"after_full_model_lm_override_{site}_zero",
                           f"after_full_model_lm_override_{site}_edited"))
    return phases + ["all_planned_model_work_complete", "details_saved"]


def check_progress(path: Path, phase: str, elapsed: float, peak: int) -> dict:
    lines = [json.loads(line, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
             for line in path.read_text().splitlines()]
    need([row.get("phase") for row in lines] == expected_progress(phase),
         "Stage-B progress chronology/order differs")
    seconds = [row.get("elapsed_seconds") for row in lines]
    memory = [row.get("mps_driver_bytes") for row in lines]
    need(all(type(v) in (int, float) and math.isfinite(v) and 0 <= v < CAP_SECONDS[phase]
             for v in seconds) and seconds == sorted(seconds) and seconds[-1] <= elapsed,
         "Stage-B progress elapsed counters differ")
    need(all(type(v) is int and 0 <= v < DRIVER_CAP for v in memory) and
         max(memory) <= peak, "Stage-B progress MPS cap/peak differs")
    return {"progress_lines": len(lines), "max_progress_seconds": max(seconds),
            "max_progress_driver_bytes": max(memory)}


def check_attempt(attempt: Path, native: tuple[dict, ...], probes: dict,
                  *, stage_a_binding: dict, prior_model_work_seconds: float) -> dict:
    need(attempt.is_dir() and not any(p.is_symlink() for p in attempt.rglob("*")),
         "Stage-B attempt absent or contains symlink")
    result_path = attempt / "result.json"
    need(result_path.is_file() and not (attempt / "error.json").exists(),
         "Stage-B completed and error receipts conflict or are missing")
    result = read_json(result_path)
    phase = result.get("phase")
    need(phase in PHASES and result.get("status") ==
         "completed_old_fit_handoff_stage_b_replay_only", "Stage-B result status/phase differs")
    need(sha(PROTOCOL) == PROTOCOL_SHA and sha(DRIVER) == DRIVER_SHA and
         sha(PRIMITIVES) == PRIMITIVES_SHA and result.get("script_sha256") == DRIVER_SHA and
         result.get("protocol_sha256") == PROTOCOL_SHA,
         "Stage-B reviewed source/protocol bytes differ")
    for field, file in (("meta_sha256", "meta.json"),
                        ("model_layout_sha256", "model_layout.json"),
                        ("progress_sha256", "progress.jsonl")):
        need(sha(attempt / file) == result.get(field), f"Stage-B {file} hash differs")
    digest = result.get("details_sha256")
    need(isinstance(digest, str) and HEX64.fullmatch(digest) is not None and
         result.get("details_path") == f"blobs/{digest}.pt", "Stage-B sidecar descriptor differs")
    detail_path = attempt / "blobs" / f"{digest}.pt"
    need(detail_path.is_file() and sha(detail_path) == digest and
         result.get("details_bytes") == detail_path.stat().st_size,
         "Stage-B details sidecar hash/size differs")
    total_bytes = sum(p.stat().st_size for p in attempt.rglob("*") if p.is_file())
    need(total_bytes < BYTE_CAP, "Stage-B output byte cap reached")
    meta, layout = read_json(attempt / "meta.json"), read_json(attempt / "model_layout.json")
    need(set(stage_a_binding) == {"manifest", "result", "audit"} and
         all(set(stage_a_binding[k]) == {"bf16", "fp32"} and
             all(isinstance(v, str) and HEX64.fullmatch(v) is not None
                 for v in stage_a_binding[k].values()) for k in stage_a_binding),
         "External Stage-A hash binding absent")
    need(meta.get("status") == "started_old_fit_handoff_stage_b_replay_only" and
         meta.get("phase") == phase and meta.get("protocol_sha256") == PROTOCOL_SHA and
         meta.get("script_sha256") == DRIVER_SHA and
         meta.get("replay_primitives_sha256") == PRIMITIVES_SHA and
         meta.get("planned_counts") == COUNTS[phase] and
         meta.get("max_output_bytes") == BYTE_CAP and
         meta.get("max_mps_driver_bytes") == DRIVER_CAP and
         meta.get("process_cap_seconds") == CAP_SECONDS[phase] and
         meta.get("cumulative_cap_seconds") == TOTAL_SECONDS and
         meta.get("fresh_v2_corpus_opened") is False and
         meta.get("source_rows") == ROWS and meta.get("cycle") == 0 and
         meta.get("alphas") == list(ALPHAS) and
         result.get("planned_counts") == result.get("actual_counts") == COUNTS[phase] and
         result.get("stage_a_manifest_sha256") == meta.get("stage_a_manifest_sha256") and
         result.get("stage_a_result_sha256") == meta.get("stage_a_result_sha256") and
         result.get("stage_a_audit_sha256") == meta.get("stage_a_audit_sha256") and
         meta.get("stage_a_manifest_sha256") == stage_a_binding["manifest"] and
         meta.get("stage_a_result_sha256") == stage_a_binding["result"] and
         meta.get("stage_a_audit_sha256") == stage_a_binding["audit"],
         "Stage-B metadata, count, or source pin differs")
    check_archive_estimate(native, meta.get("archive_estimate"))
    check_probes(probes)
    check_layout(layout, "bf16" if phase == "bf16_direct_lm" else "fp32")
    prior = result.get("prior_model_work_seconds")
    elapsed = result.get("elapsed_seconds")
    peak = result.get("peak_mps_driver_bytes")
    need(type(prior) in (int, float) and type(elapsed) in (int, float) and
         math.isfinite(prior) and math.isfinite(elapsed) and
         0 < elapsed < CAP_SECONDS[phase] and
         prior == prior_model_work_seconds and
         prior + elapsed < TOTAL_SECONDS and
         meta.get("prior_model_work_seconds") == prior and
         type(peak) is int and 0 <= peak < DRIVER_CAP,
         "Stage-B process/cumulative/MPS cap receipt invalid")
    progress = check_progress(attempt / "progress.jsonl", phase, elapsed, peak)
    records = torch.load(detail_path, map_location="cpu", weights_only=True)
    need(isinstance(records, dict) and set(records) ==
         {"source_replay", "direct_lm", "lm_override_validation"},
         "Stage-B detail record schema differs")
    direct = check_direct(records["direct_lm"], native, phase, probes)
    analysis = {"direct_lm": direct, "progress": progress}
    if phase == "bf16_direct_lm":
        need(records["source_replay"] is None and records["lm_override_validation"] is None,
             "BF16 Stage-B contains FP32-only products")
    else:
        analysis["source_replay"] = check_source(records["source_replay"], native, probes)
        analysis["lm_override_validation"] = check_overrides(records["lm_override_validation"],
                                                             records["direct_lm"])
    status = ("audited_completed_old_fit_handoff_stage_b_bf16_direct_lm_only"
              if phase == "bf16_direct_lm" else
              "audited_completed_old_fit_handoff_stage_b_fp32_replay_only")
    return {"status": status,
            "phase": phase, "result_sha256": sha(result_path),
            "details_sha256": digest, "auditor_sha256": sha(Path(__file__)),
            "elapsed_seconds": elapsed, "actual_counts": COUNTS[phase],
            "analysis": analysis,
            "scope": "saved_tensors_and_receipts_only_no_forward_or_backward_execution_attestation"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("attempt", type=Path)
    parser.add_argument("--stage-a-bf16-details", type=Path, required=True)
    parser.add_argument("--stage-a-probes", type=Path, required=True)
    parser.add_argument("--prior-model-work-seconds", type=float, required=True)
    args = parser.parse_args()
    # This CLI is intentionally not a live result authorizer until the
    # external Stage-A manifests and Stage-B launcher chronology are wired.
    raise SystemExit("Stage-B external Stage-A/launcher chain verifier is not yet implemented")


if __name__ == "__main__":
    main()
