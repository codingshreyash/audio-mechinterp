"""Gated one-shot 328-call Whisper non-speech causal worker.

Import and `admit` do not deserialize candidate weights. `run` requires a
later independent executable GO bound to the exact committed source.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from . import causal_plan_v1 as plan
from . import descriptive_worker_v1 as old

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ATTEMPT = HERE / "runs/causal_whisper_v1/attempt_001"
ADMISSION = HERE / "CAUSAL_WHISPER_ADMISSION_V1.json"
GO = HERE / "CAUSAL_WHISPER_EXECUTABLE_GO_V1.json"
DESCRIPTIVE_RECEIPT = HERE / "runs/descriptive_source_preflight_v2/preflight.json"
DESCRIPTIVE_GO = HERE / "INDEPENDENT_DESCRIPTIVE_SOURCE_GO_V2.json"
DESCRIPTIVE_V2_RECORDS = HERE / "runs/descriptive_whisper_multilingual_20261119_v2/records.jsonl"
DESCRIPTIVE_V2_RECORDS_SHA = "588165986eadcfc7df04f6a8869de78ccd656f5b9be578fe41b3e1395abeec42"
AMENDED_DESIGN_GO = HERE / "WHISPER_NONSPEECH_CAUSAL_V2_V3_INDEPENDENT_DESIGN_REVIEW_V1.md"
AMENDED_DESIGN_GO_SHA = "9d1046a467c8b9d20ab06d58d363a3f11d49316e6136b3dc20ae087b68346aa2"
MAX_SECONDS = 1800
MAX_RSS_BYTES = 4 * 1024**3
MAX_EVIDENCE_BYTES = plan.MAX_EVIDENCE_BYTES
EVIDENCE_RESERVE_BYTES = plan.EVIDENCE_RESERVE_BYTES
SOURCE_NAMES = (
    "causal_worker_v1.py", "causal_parent_v1.py", "causal_audit_v1.py",
    "test_causal_worker_v1.py", "test_causal_parent_v1.py",
    "CAUSAL_EXECUTABLE_HANDOFF_V1.md", "causal_plan_v1.py",
    "test_causal_plan_v1.py", "WHISPER_NONSPEECH_CAUSAL_INTERVENTION_PROSPECTIVE_V1.md",
    "WHISPER_NONSPEECH_CAUSAL_DESIGN_AMENDMENT_V1.md",
    "WHISPER_NONSPEECH_CAUSAL_DESIGN_AMENDMENT_V2.md",
    "WHISPER_NONSPEECH_CAUSAL_DESIGN_AMENDMENT_V3.md",
    "WHISPER_NONSPEECH_CAUSAL_DESIGN_INDEPENDENT_REVIEW_V1.md",
    "WHISPER_NONSPEECH_CAUSAL_V2_V3_INDEPENDENT_DESIGN_REVIEW_V1.md",
    "descriptive_worker_v1.py", "descriptive_pins_v1.json",
    "whisper_tiny_metadata_pin.json", "spec.json", "render.py",
)
HOST_SOURCE = HERE.parent / "wav2vec2_fit_noise_norm_causal_host_v1.py"
HOST_SOURCE_SHA = "b59f3998dcd36340d9fa0a9b5ceaa85179295899943dcabb1c6325f6635ce21f"


def require(ok: Any, message: str) -> None:
    if not ok:
        raise ValueError(message)


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_file(path: Path) -> str:
    return old.sha_file(path)


def write_once(path: Path, value: dict) -> None:
    require(path.parent.is_dir() and not path.exists() and not path.is_symlink(),
            f"exclusive output file required: {path.name}")
    with path.open("x", encoding="utf-8") as stream:
        stream.write(canonical(value) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def append_record(path: Path, value: dict) -> None:
    require(path.is_file() and not path.is_symlink(), "missing ledger")
    with path.open("a", encoding="utf-8") as stream:
        stream.write(canonical(value) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def source_identity() -> dict[str, str]:
    result = {}
    for name in SOURCE_NAMES:
        path = HERE / name
        require(path.is_file(), f"missing study source: {name}")
        result[name] = sha_file(path)
    require(HOST_SOURCE.is_file() and sha_file(HOST_SOURCE) == HOST_SOURCE_SHA,
            "audited host source changed or absent")
    result["../wav2vec2_fit_noise_norm_causal_host_v1.py"] = sha_file(HOST_SOURCE)
    return result


def committed_sources() -> dict[str, str]:
    result = source_identity()
    for relative in result:
        path = (HERE / relative).resolve()
        rel = str(path.relative_to(ROOT))
        for flag in ((), ("--cached",)):
            require(subprocess.run(("git", "diff", *flag, "--quiet", "--", rel),
                                   cwd=ROOT, check=False).returncode == 0,
                    f"dirty study source: {relative}")
        commit = subprocess.check_output(("git", "log", "-1", "--format=%H", "--", rel),
                                         cwd=ROOT, text=True).strip()
        require(len(commit) == 40, f"uncommitted study source: {relative}")
    return result


def preflight_receipt() -> dict:
    """No-weight frozen source, asset and 328-schedule admission."""
    old.require_independent_go(DESCRIPTIVE_RECEIPT, DESCRIPTIVE_GO)
    require(sha_file(AMENDED_DESIGN_GO) == AMENDED_DESIGN_GO_SHA,
            "independently reviewed amended causal design changed")
    require(sha_file(DESCRIPTIVE_V2_RECORDS) == DESCRIPTIVE_V2_RECORDS_SHA,
            "audited descriptive evidence changed")
    old_receipt = old.preflight_receipt()
    rows = plan.schedule()
    require(len(rows) == 328 and plan.MAX_EVIDENCE_BYTES == 2*1024**3,
            "prospective schedule or evidence cap changed")
    payload = canonical(rows).encode()
    require(sha_bytes(payload) ==
            "a4a77079c9381758720712f7b5420186e0b08a18c8f3fbbf1a99beb4926d087b",
            "prospective 328-call schedule hash changed")
    require(plan.worst_case_core_evidence_bytes()
            + plan.METADATA_ALLOWANCE_BYTES + EVIDENCE_RESERVE_BYTES
            <= MAX_EVIDENCE_BYTES, "no-weight evidence bound exceeded")
    return {"schema": "nonspeech_whisper_causal_admission_v1",
            "status": "source_only_no_model_load", "sources": committed_sources(),
            "descriptive_go_sha256": sha_file(DESCRIPTIVE_GO),
            "amended_design_go_sha256": AMENDED_DESIGN_GO_SHA,
            "descriptive_valid_records_sha256": DESCRIPTIVE_V2_RECORDS_SHA,
            "snapshot": old_receipt["models"]["whisper_multilingual"],
            "installed": old_receipt["installed"],
            "manifest_sha256": old.MANIFEST_SHA256,
            "schedule_sha256": sha_bytes(payload), "planned_calls": len(rows),
            "attempt_path": str(ATTEMPT.resolve()),
            "limits": {"wall_seconds": MAX_SECONDS, "rss_bytes": MAX_RSS_BYTES,
                       "evidence_bytes": MAX_EVIDENCE_BYTES,
                       "evidence_reserve_bytes": EVIDENCE_RESERVE_BYTES}}


def validate_go(admission_path: Path, go_path: Path) -> dict:
    require(admission_path.resolve() == ADMISSION.resolve()
            and go_path.resolve() == GO.resolve(),
            "fixed admission and executable GO paths required")
    require(admission_path.is_file() and not admission_path.is_symlink()
            and go_path.is_file() and not go_path.is_symlink(),
            "missing admission or independent executable GO")
    admission = json.loads(admission_path.read_text())
    require(admission == preflight_receipt(), "admission does not bind current source")
    go = json.loads(go_path.read_text())
    require(go.get("schema") == "nonspeech_whisper_causal_executable_go_v1"
            and go.get("status") == "GO"
            and go.get("admission_sha256") == sha_file(admission_path)
            and go.get("sources") == admission["sources"],
            "independent executable GO not bound to source/admission")
    for receipt_path in (admission_path, go_path):
        rel = str(receipt_path.resolve().relative_to(ROOT))
        require(subprocess.run(("git", "ls-files", "--error-unmatch", "--", rel),
                               cwd=ROOT, check=False, capture_output=True).returncode == 0,
                f"untracked receipt: {receipt_path.name}")
        for flag in ((), ("--cached",)):
            require(subprocess.run(("git", "diff", *flag, "--quiet", "--", rel),
                                   cwd=ROOT, check=False).returncode == 0,
                    f"dirty receipt: {receipt_path.name}")
    rel_go = str(go_path.resolve().relative_to(ROOT))
    review_commit = subprocess.check_output(("git", "log", "-1", "--format=%H",
                                             "--", rel_go), cwd=ROOT, text=True).strip()
    require(len(review_commit) == 40, "GO receipt uncommitted")
    for path in [admission_path, *(HERE / name for name in SOURCE_NAMES), HOST_SOURCE]:
        rel = str(path.resolve().relative_to(ROOT))
        source_commit = subprocess.check_output(("git", "log", "-1", "--format=%H",
                                                 "--", rel), cwd=ROOT, text=True).strip()
        require(source_commit != review_commit and subprocess.run(
            ("git", "merge-base", "--is-ancestor", source_commit, review_commit),
            cwd=ROOT, check=False).returncode == 0,
            f"GO must postdate committed evidence: {path.name}")
    return admission


def call_name(row: dict) -> str:
    return f"{row['stage']}_{row['case_id']}_{row['arm']}"


def output_bytes(directory: Path) -> int:
    return sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())


class Budget:
    def __init__(self, output: Path) -> None:
        self.output = output
        self.start = time.monotonic()
        self.attempted = 0

    def check(self, *, before_forward: bool = False, proposed_bytes: int = 0) -> None:
        require(time.monotonic() - self.start <= MAX_SECONDS, "worker wall cap exceeded")
        require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss <= MAX_RSS_BYTES,
                "worker RSS cap exceeded")
        used = output_bytes(self.output)
        require(used + proposed_bytes + EVIDENCE_RESERVE_BYTES <= MAX_EVIDENCE_BYTES,
                "worker evidence reserve exceeded")
        if before_forward:
            require(self.attempted < len(plan.schedule()), "forward count exceeded")
            self.attempted += 1

    def append(self, path: Path, value: dict, *, failure_ledger: bool = False) -> None:
        proposed = len(canonical(value).encode()) + 1
        if failure_ledger:
            require(output_bytes(self.output) + proposed <= MAX_EVIDENCE_BYTES,
                    "failure ledger exceeds hard evidence cap")
        else:
            self.check(proposed_bytes=proposed)
        append_record(path, value)
        require(output_bytes(self.output) <= MAX_EVIDENCE_BYTES,
                "ledger write exceeded hard evidence cap")


def save_npy(output: Path, relative: str, value: Any, budget: Budget) -> dict:
    import numpy as np
    array = np.ascontiguousarray(value)
    require(array.dtype == np.float32 and np.isfinite(array).all(),
            "saved tensor must be finite float32")
    path = output / relative
    require(path.resolve().is_relative_to(output.resolve())
            and not path.exists() and not path.is_symlink(), "invalid exclusive sidecar")
    budget.check(proposed_bytes=array.nbytes+1024)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.save(stream, array, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    budget.check()
    return {"path": relative, "file_sha256": sha_file(path),
            "tensor_sha256": sha_bytes(array.tobytes()),
            "shape": list(array.shape), "dtype": str(array.dtype),
            "bytes": path.stat().st_size}


def random_patch(native: Any, donor: Any, seed: int) -> tuple[Any, float, float]:
    """Exact float64 normalization then float32 addition frozen in amendment V2."""
    import numpy as np
    native = np.asarray(native, dtype=np.float32)
    donor = np.asarray(donor, dtype=np.float32)
    require(native.shape == donor.shape == (1, 1500, 384), "random state shape changed")
    target = float(np.linalg.norm((donor-native).astype(np.float64).ravel()))
    if target == 0:
        edit = np.zeros_like(native)
    else:
        noise = np.random.default_rng(seed).standard_normal(native.shape).astype(np.float32)
        norm = float(np.linalg.norm(noise.astype(np.float64).ravel()))
        require(norm > 0, "zero random normal draw")
        edit = (noise.astype(np.float64) / norm * target).astype(np.float32)
    patched = (native + edit).astype(np.float32)
    realized = float(np.linalg.norm((patched-native).astype(np.float64).ravel()))
    return patched, target, realized


class PhaseHooks:
    """Capture both encoder sites and edit only the scheduled site/phase."""
    def __init__(self, model: Any, row: dict, donor_states: dict[str, dict[str, Any]]):
        self.model, self.row, self.donor_states = model, row, donor_states
        self.phase = "transcribe"
        self.original_detect = None
        self.handles: list[Any] = []
        self.native: dict[str, dict[str, Any]] = {"detect": {}, "transcribe": {}}
        self.hook: dict[str, dict[str, dict]] = {"detect": {}, "transcribe": {}}
        self.counts: dict[str, dict[str, int]] = {"detect": {}, "transcribe": {}}
        self.detection: dict | None = None

    def _selected(self, site: str, phase: str) -> bool:
        arm = self.row["arm"]
        if self.row["site"] != site or arm in ("no_hook", "read_only"):
            return False
        if arm.endswith("_detect"):
            return phase == "detect"
        if arm.endswith("_transcribe"):
            return phase == "transcribe"
        return arm.endswith("_both")

    def _edit(self, native: Any, site: str, phase: str) -> tuple[Any, list[int], int | None]:
        import numpy as np
        if not self._selected(site, phase):
            return native, [0, 0], None
        donor = self.donor_states[site][phase]
        require(native.shape == donor.shape == (1, 1500, 384),
                "phase donor tensor shape changed")
        arm = self.row["arm"]
        if arm.startswith("self_"):
            return native.copy(), [0, 1500], None
        if arm.startswith("donor_"):
            return donor.copy(), [0, 1500], None
        if arm.startswith("random_"):
            seed = int(self.row["seeds"][phase])
            edited, _target, _realized = random_patch(native, donor, seed)
            return edited, [0, 1500], seed
        active = int(self.row["active_rows"])
        if arm == "active_both":
            start, end = 0, active
        elif arm == "displaced_both":
            start, end = active, 2*active
        else:
            raise ValueError(f"unknown intervention arm: {arm}")
        edited = native.copy()
        edited[:, start:end, :] = donor[:, start:end, :]
        return edited, [start, end], None

    def _site(self, site: str, tensor: Any) -> Any | None:
        import numpy as np
        import torch
        require(torch.is_tensor(tensor) and tensor.device.type == "cpu"
                and tensor.dtype == torch.float32 and tuple(tensor.shape) == (1, 1500, 384)
                and bool(torch.isfinite(tensor).all()),
                f"{site} tensor topology changed")
        phase = self.phase
        counts = self.counts[phase]
        counts[site] = counts.get(site, 0) + 1
        require(counts[site] == 1, f"repeated {phase}/{site} encoder hook")
        native = np.ascontiguousarray(tensor.detach().numpy().copy())
        self.native[phase][site] = native
        donor = self.donor_states.get(site, {}).get(phase)
        donor_delta = (float(np.linalg.norm((donor-native).astype(np.float64).ravel()))
                       if donor is not None else 0.0)
        edited, support, seed = self._edit(native, site, phase)
        edit_l2 = float(np.linalg.norm((edited-native).astype(np.float64).ravel()))
        self.hook[phase][site] = {
            "hook_calls": 1, "native_sha256": sha_bytes(native.tobytes()),
            "edited_sha256": sha_bytes(np.ascontiguousarray(edited).tobytes()),
            "edit_l2": edit_l2, "donor_delta_l2": donor_delta,
            "support_rows": support, "seed": seed,
        }
        if self._selected(site, phase):
            return torch.from_numpy(np.ascontiguousarray(edited)).to(tensor.device)
        return None

    def install(self) -> None:
        encoder = self.model.model.encoder

        def early(_module: Any, args: tuple[Any, ...]) -> tuple[Any, ...] | None:
            updated = self._site("pre_block0", args[0])
            if updated is None:
                return None
            return (updated, *args[1:])

        def final(_module: Any, _args: tuple[Any, ...], result: Any) -> Any | None:
            return self._site("final_encoder", result)

        self.handles.append(encoder.layers[0].register_forward_pre_hook(early))
        self.handles.append(encoder.layer_norm.register_forward_hook(final))
        self.original_detect = self.model.detect_language

        def detect(*args: Any, **kwargs: Any) -> Any:
            self.phase = "detect"
            try:
                result = self.original_detect(*args, **kwargs)
                require(self.detection is not None, "language detection logit capture absent")
                self.detection["chosen_language_ids"] = [int(x) for x in result.tolist()]
                return result
            finally:
                self.phase = "transcribe"

        self.model.detect_language = detect

        def language(_module: Any, _args: Any, result: Any) -> None:
            if self.phase != "detect":
                return
            import torch
            vec = result.logits[0, -1].detach().to("cpu", dtype=torch.float32)
            config = self.model.generation_config
            values = {name: float(vec[token]) for name, token in config.lang_to_id.items()}
            require(self.detection is None, "repeated language detection forward")
            self.detection = {"raw_logits_sha256": sha_bytes(vec.numpy().tobytes()),
                              "language_logits": values, "candidate_count": len(values)}

        self.handles.append(self.model.register_forward_hook(language))

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        if self.original_detect is not None:
            self.model.detect_language = self.original_detect

    def validated(self) -> None:
        expected = {"pre_block0": 1, "final_encoder": 1}
        require(self.counts == {"detect": expected, "transcribe": expected},
                "two-phase hook topology changed")
        require(self.detection is not None
                and len(self.detection.get("chosen_language_ids", [])) == 1,
                "automatic language detection count changed")


def generate_one(row: dict, case: dict, processor: Any, model: Any,
                 donor_states: dict[str, dict[str, Any]],
                 output_dir: Path, budget: Budget, call_index: int) -> tuple[dict, dict]:
    """One complete generate call; no-hook arms install no model hooks."""
    import numpy as np
    import torch
    pcm = case["pcm"]
    audio = pcm.astype(np.float32) / np.float32(32768.0)
    processed = processor(audio, sampling_rate=16000, return_tensors="pt",
                          return_attention_mask=True)
    features = processed.input_features
    require(features.dtype == torch.float32 and tuple(features.shape) == (1, 80, 3000),
            "Whisper mel geometry changed")
    geometry = old.whisper_geometry(processed.attention_mask, len(pcm), 3000)
    require(geometry["nominal_active_encoder_rows"] == row["active_rows"],
            "frozen early-time interval changed")
    require(model.generation_config.is_multilingual is True
            and getattr(model.generation_config, "language", None) is None,
            "automatic-language config changed")
    hooks = None if row["arm"] == "no_hook" else PhaseHooks(model, row, donor_states)
    try:
        if hooks is not None:
            hooks.install()
        with torch.inference_mode():
            result = model.generate(
                input_features=features, task="transcribe", language=None,
                attention_mask=processed.attention_mask,
                return_timestamps=False, force_unique_generate_call=True,
                condition_on_prev_tokens=False, do_sample=False, num_beams=1,
                max_new_tokens=plan.MAX_NEW_TOKENS, output_scores=True,
                output_logits=True, output_hidden_states=True,
                return_dict_in_generate=True)
        if hooks is not None:
            hooks.validated()
        scored = old.score_whisper(result, processor.tokenizer,
                                   hooks.detection if hooks is not None else None)
        require(scored["prefix_ids"][:1] == [model.generation_config.decoder_start_token_id]
                and scored["prefix_ids"][2:] == [
                    model.generation_config.task_to_id["transcribe"],
                    model.generation_config.no_timestamps_token_id],
                "resolved task/prefix changed")
        require(scored["termination"] in ("eos", "length_censored", "incomplete_unknown"),
                "generation termination classification changed")
        if hooks is not None:
            require(scored["prefix_ids"][1] ==
                    hooks.detection["chosen_language_ids"][0],
                    "detection/prefix language mismatch")
        raw = scored.pop("first_raw_logits")
        decoder = scored.pop("first_decoder_step")
        require(len(raw["values"]) == plan.VOCAB_SIZE,
                "first raw logit vector width changed")
        stem = f"{call_index:03d}_{call_name(row)}"
        first_raw = save_npy(output_dir, f"raw/{stem}_first.npy",
                             np.asarray(raw.pop("values"), dtype=np.float32), budget)
        require(first_raw["tensor_sha256"] == raw["sha256"],
                "saved first raw logits differ from generation")
        first_decoder = save_npy(output_dir, f"decoder/{stem}_first.npy",
                                 np.asarray(decoder.pop("values"), dtype=np.float32), budget)
        require(first_decoder["tensor_sha256"] == decoder["sha256"],
                "saved first decoder state differs from generation")
        full_raw = []
        if row["save_full_raw_logits_each_step"]:
            require(len(result.logits) == scored["actual_generation_steps"],
                    "full raw generation count changed")
            for step, value in enumerate(result.logits):
                array = value[0].detach().to("cpu", dtype=torch.float32).numpy()
                require(array.shape == (plan.VOCAB_SIZE,),
                        "full raw step width changed")
                ref = save_npy(output_dir, f"raw/{stem}_step_{step:03d}.npy",
                               array, budget)
                require(ref["tensor_sha256"] == scored["score_rows"][step][
                    "raw_logits_sha256"], "full raw step differs from generated row")
                full_raw.append(ref)
        return ({"output": scored, "hook": hooks.hook if hooks is not None else {},
                 "hook_counts": hooks.counts if hooks is not None else {},
                 "native": hooks.native if hooks is not None else {},
                 "sidecars": {"first_raw": first_raw,
                              "full_raw": full_raw,
                              "first_decoder_state": first_decoder},
                 "input_geometry": geometry,
                 "first_raw_meta": raw, "first_decoder_meta": decoder},
                {"raw_logits": tuple(value[0].detach().to("cpu", dtype=torch.float32).numpy().copy()
                                     for value in result.logits)})
    finally:
        if hooks is not None:
            hooks.close()


def token_parity(output: dict, baseline: dict | None) -> bool | None:
    if baseline is None:
        return None
    return output["sequence_ids"] == baseline["sequence_ids"] \
        and output["actual_generation_steps"] == baseline["actual_generation_steps"] \
        and output["termination"] == baseline["termination"]


def raw_maxabs(actual: tuple[Any, ...], expected: tuple[Any, ...]) -> float | None:
    import numpy as np
    if len(actual) != len(expected):
        return None
    if any(a.shape != b.shape for a, b in zip(actual, expected, strict=True)):
        return None
    return max(float(np.max(np.abs(a.astype(np.float64)-b.astype(np.float64))))
               for a, b in zip(actual, expected, strict=True))


def save_native_states(output_dir: Path, case_id: str,
                       captured: dict[str, dict[str, Any]],
                       budget: Budget) -> dict[str, dict[str, dict]]:
    refs: dict[str, dict[str, dict]] = {"pre_block0": {}, "final_encoder": {}}
    for site in ("pre_block0", "final_encoder"):
        for phase in ("detect", "transcribe"):
            value = captured[phase][site]
            refs[site][phase] = save_npy(
                output_dir, f"native/{case_id}_{site}_{phase}.npy", value, budget)
    return refs


def run_model(output_dir: Path, admission_path: Path, go_path: Path) -> dict:
    """Run only behind separate executable GO; never retry an existing attempt."""
    import numpy as np
    import torch
    admission = validate_go(admission_path, go_path)
    require(output_dir.resolve() == (ATTEMPT / "worker_output").resolve()
            and output_dir.parent.is_dir() and not output_dir.exists(),
            "fixed exclusive worker output path required")
    schedule = plan.schedule()
    cases = {case["case_id"]: case for case in old.trusted_cases()}
    require(set(cases) == set(plan.frozen_cases()), "audited case population changed")
    v2_outputs = {}
    with DESCRIPTIVE_V2_RECORDS.open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            if record.get("type") == "case":
                require(record["case_id"] not in v2_outputs,
                        "duplicate valid descriptive case")
                v2_outputs[record["case_id"]] = record["output"]
    require(set(v2_outputs) == set(cases), "valid descriptive case census changed")
    output_dir.mkdir()
    records = output_dir / "records.jsonl"
    progress = output_dir / "call_progress.jsonl"
    write_once(records, {"type": "start", "schema": "nonspeech_causal_records_v1",
                         "planned_calls": len(schedule), "admission_sha256": sha_file(admission_path),
                         "go_sha256": sha_file(go_path),
                         "schedule_sha256": admission["schedule_sha256"]})
    with progress.open("x", encoding="utf-8") as stream:
        stream.flush()
        os.fsync(stream.fileno())
    budget = Budget(output_dir)
    states: dict[str, dict[str, dict[str, Any]]] = {}
    baselines: dict[str, dict] = {}
    donor_raw: dict[str, tuple[Any, ...]] = {}
    model = None
    failed_index: int | None = None
    try:
        torch.set_num_threads(4)
        processor, model = old._snapshot_load("whisper_multilingual",
                                              admission["snapshot"])
        require(model.training is False and all(
            parameter.device.type == "cpu" and parameter.dtype == torch.float32
            for parameter in model.parameters()), "model not CPU FP32 eval")
        for index, row in enumerate(schedule, start=1):
            failed_index = index
            name = call_name(row)
            prior_baseline = baselines.get(row["case_id"])
            budget.append(progress, {"type": "attempt", "call_index": index,
                                     "call_name": name})
            budget.check(before_forward=True)
            donor_id = row["donor_id"]
            donor_states = states.get(donor_id, {})
            start = time.monotonic()
            observed, extras = generate_one(row, cases[row["case_id"]], processor,
                                            model, donor_states, output_dir, budget, index)
            output = observed["output"]
            native_refs = {}
            if row["arm"] == "read_only" and (
                    row["stage"] == "donor" or row["stage"] == "A"):
                require(row["case_id"] not in states,
                        "duplicate native source capture")
                states[row["case_id"]] = {site: {phase: observed["native"][phase][site]
                                                 for phase in ("detect", "transcribe")}
                                          for site in ("pre_block0", "final_encoder")}
                native_refs = save_native_states(output_dir, row["case_id"],
                                                 observed["native"], budget)
            if row["stage"] == "B" and row["arm"] == "read_only":
                require(all(sha_bytes(observed["native"][phase][site].tobytes()) ==
                            sha_bytes(states[row["case_id"]][site][phase].tobytes())
                            for site in ("pre_block0", "final_encoder")
                            for phase in ("detect", "transcribe")),
                        "Stage B read-only native capture differs from Stage A")
            if row["arm"] == "no_hook" and row["case_id"] not in baselines:
                baselines[row["case_id"]] = output
            if row["stage"] == "donor" and row["arm"] == "no_hook":
                donor_raw[row["case_id"]] = extras["raw_logits"]
            baseline = baselines.get(row["case_id"])
            donor = baselines.get(donor_id)
            parity: dict[str, Any] = {
                "recipient_tokens_match": token_parity(output, baseline),
                "donor_tokens_match": token_parity(output, donor),
                "descriptive_v2_tokens_match": token_parity(
                    output, v2_outputs[row["case_id"]]),
            }
            if row["arm"] == "no_hook" and prior_baseline is not None:
                parity["repeated_no_hook_match"] = token_parity(output, prior_baseline)
            if row["arm"] == "donor_both":
                parity["donor_raw_maxabs"] = raw_maxabs(
                    extras["raw_logits"], donor_raw[donor_id])
                parity["donor_raw_within_1e-5"] = (
                    parity["donor_raw_maxabs"] is not None
                    and parity["donor_raw_maxabs"] <= 1e-5)
            # Parity failures invalidate interpretation but remain complete rows
            # in the prospective 328-call denominator.
            hook = observed["hook"]
            if hook:
                for phase in ("detect", "transcribe"):
                    for site in ("pre_block0", "final_encoder"):
                        info = hook[phase][site]
                        if row["case_id"] in states and not (
                                row["stage"] == "B" and site == "final_encoder"
                                and row["arm"] not in ("read_only", "no_hook")):
                            require(info["native_sha256"] == sha_bytes(
                                states[row["case_id"]][site][phase].tobytes()),
                                "native state changed across calls")
            budget.append(records, {"type": "forward", "call_index": index,
                                    "call_name": name, "plan_row": row,
                                    "output": output, "hook": hook,
                                    "hook_counts": observed["hook_counts"],
                                    "native_state_refs": native_refs,
                                    "sidecars": observed["sidecars"],
                                    "first_raw_meta": observed["first_raw_meta"],
                                    "first_decoder_meta": observed["first_decoder_meta"],
                                    "input_geometry": observed["input_geometry"],
                                    "parity": parity,
                                    "elapsed_seconds": time.monotonic()-start})
            budget.append(progress, {"type": "complete", "call_index": index,
                                     "call_name": name})
            failed_index = None
            budget.check()
        budget.append(records, {"type": "complete", "planned_calls": len(schedule),
                                "attempted_calls": budget.attempted,
                                "evidence_bytes": output_bytes(output_dir),
                                "peak_rss_bytes": resource.getrusage(
                                    resource.RUSAGE_SELF).ru_maxrss})
        return {"status": "worker_complete", "forwards": len(schedule),
                "evidence_bytes": output_bytes(output_dir)}
    except BaseException as error:
        censored_start = (failed_index if failed_index is not None else budget.attempted) + 1
        budget.append(records, {"type": "failure", "error_type": type(error).__name__,
                                "error": str(error),
                                "failed_attempted_call_name": (
                                    call_name(schedule[failed_index-1])
                                    if failed_index is not None else None),
                                "censored_call_names": [call_name(row)
                                                        for row in schedule[censored_start-1:]],
                                "attempted_calls": budget.attempted},
                      failure_ledger=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    admit = sub.add_parser("admit")
    admit.add_argument("--output", type=Path, required=True)
    run = sub.add_parser("run")
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--admission", type=Path, required=True)
    run.add_argument("--go", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "admit":
        require(args.output.resolve() == ADMISSION.resolve(),
                "fixed admission path required")
        receipt = preflight_receipt()
        write_once(args.output, receipt)
        print(canonical(receipt))
    else:
        print(canonical(run_model(args.output, args.admission, args.go)))


if __name__ == "__main__":
    main()
