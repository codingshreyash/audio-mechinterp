"""Gated, one-shot 436-call Whisper padding-channel worker.

Import, source admission, and fake tests never deserialize candidate weights.
Only `run` after a later independently committed executable GO may load them.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import subprocess
import time
from pathlib import Path
from typing import Any

from . import descriptive_worker_v1 as descriptive
from . import padding_channel_plan_v4 as plan
from . import padding_channel_core_v1 as core

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ATTEMPT = HERE / "runs/padding_channel_v1/attempt_001"
ADMISSION = HERE / "PADDING_CHANNEL_ADMISSION_V1.json"
GO = HERE / "PADDING_CHANNEL_EXECUTABLE_GO_V1.json"
DESIGN_GO = HERE / "WHISPER_PADDING_CHANNEL_INDEPENDENT_DESIGN_REREVIEW_V4.md"
DESIGN_GO_SHA256 = "ec0220b7d93a986e10522315ada0b2994a397975f5e545e1a51aa25d4d74a00f"
HOST_SOURCE = HERE.parent / "wav2vec2_fit_noise_norm_causal_host_v1.py"
HOST_SOURCE_SHA256 = "b59f3998dcd36340d9fa0a9b5ceaa85179295899943dcabb1c6325f6635ce21f"
MAX_SECONDS = plan.WALL_SECONDS
MAX_RSS_BYTES = plan.RSS_LIMIT_BYTES
MAX_EVIDENCE_BYTES = plan.EVIDENCE_LIMIT_BYTES
EVIDENCE_RESERVE_BYTES = plan.EVIDENCE_RESERVE_BYTES
SOURCE_NAMES = (
    "padding_channel_worker_v1.py", "padding_channel_parent_v1.py",
    "padding_channel_audit_v1.py", "padding_channel_core_v1.py",
    "padding_channel_plan_v1.py", "padding_channel_plan_v4.py",
    "test_padding_channel_plan_v4.py", "test_padding_channel_worker_v1.py",
    "test_padding_channel_parent_v1.py", "test_padding_channel_audit_v1.py",
    "test_padding_channel_core_v1.py", "test_padding_channel_plan_v1.py",
    "PADDING_CHANNEL_EXECUTABLE_HANDOFF_V1.md",
    "WHISPER_PADDING_CHANNEL_PROSPECTIVE_V1.md",
    "WHISPER_PADDING_CHANNEL_PROSPECTIVE_V2.md",
    "WHISPER_PADDING_CHANNEL_PROSPECTIVE_V3.md",
    "WHISPER_PADDING_CHANNEL_PROSPECTIVE_V4.md",
    "WHISPER_PADDING_CHANNEL_INDEPENDENT_DESIGN_REREVIEW_V3.md",
    "WHISPER_PADDING_CHANNEL_INDEPENDENT_DESIGN_REREVIEW_V4.md",
    "descriptive_worker_v1.py", "causal_parent_v2.py", "causal_worker_v2.py",
    "causal_plan_v1.py", "descriptive_pins_v1.json",
    "whisper_tiny_metadata_pin.json", "spec.json", "render.py",
)


def require(ok: Any, message: str) -> None:
    if not ok:
        raise ValueError(message)


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha_file(path: Path) -> str:
    return descriptive.sha_file(path)


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def write_once(path: Path, value: dict) -> None:
    require(path.parent.is_dir() and not path.exists() and not path.is_symlink(),
            f"exclusive output file required: {path}")
    with path.open("x", encoding="utf-8") as stream:
        stream.write(canonical(value) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def append_record(path: Path, value: dict) -> None:
    require(path.is_file() and not path.is_symlink(), "ledger missing")
    with path.open("a", encoding="utf-8") as stream:
        stream.write(canonical(value) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def source_identity() -> dict[str, str]:
    result = {}
    for name in SOURCE_NAMES:
        path = HERE / name
        require(path.is_file(), f"missing source: {name}")
        result[name] = sha_file(path)
    require(sha_file(HOST_SOURCE) == HOST_SOURCE_SHA256,
            "host-gate source changed")
    result["../wav2vec2_fit_noise_norm_causal_host_v1.py"] = HOST_SOURCE_SHA256
    return result


def committed_sources() -> dict[str, str]:
    result = source_identity()
    for relative in result:
        path = (HERE / relative).resolve()
        rel = str(path.relative_to(ROOT))
        for flag in ((), ("--cached",)):
            require(subprocess.run(("git", "diff", *flag, "--quiet", "--", rel),
                                   cwd=ROOT, check=False).returncode == 0,
                    f"dirty source: {relative}")
        commit = subprocess.check_output(("git", "log", "-1", "--format=%H", "--", rel),
                                         cwd=ROOT, text=True).strip()
        require(len(commit) == 40, f"uncommitted source: {relative}")
    return result


def preflight_receipt() -> dict:
    """Hash sources/assets and schedule; no checkpoint deserialization."""
    require(sha_file(DESIGN_GO) == DESIGN_GO_SHA256, "design GO changed")
    plan.verify_installed_source_bytes()
    require(plan.schedule_sha256() ==
            "d939618689e989a2dfe249cc6d4f0d3b32a532c184991680b77f11b4af6a0ace"
            and len(plan.schedule()) == 436 and plan.resource_budget_fits(),
            "frozen schedule or evidence bound changed")
    installed = descriptive.verify_installed_sources()
    snapshot = descriptive.verify_local_assets("whisper_multilingual")
    return {"schema": "whisper_padding_admission_v1",
            "status": "source_only_no_model_load", "sources": committed_sources(),
            "design_go_sha256": DESIGN_GO_SHA256,
            "manifest_sha256": plan.MANIFEST_SHA256,
            "schedule_sha256": plan.schedule_sha256(), "planned_calls": 436,
            "snapshot": snapshot, "installed": installed,
            "attempt_path": str(ATTEMPT.resolve()),
            "limits": {"wall_seconds": MAX_SECONDS, "rss_bytes": MAX_RSS_BYTES,
                       "evidence_bytes": MAX_EVIDENCE_BYTES,
                       "reserve_bytes": EVIDENCE_RESERVE_BYTES}}


def validate_go(admission_path: Path, go_path: Path) -> dict:
    require(admission_path.resolve() == ADMISSION.resolve()
            and go_path.resolve() == GO.resolve(), "fixed admission/GO paths required")
    require(admission_path.is_file() and go_path.is_file()
            and not admission_path.is_symlink() and not go_path.is_symlink(),
            "admission or independent GO missing")
    admission = json.loads(admission_path.read_text())
    require(admission == preflight_receipt(), "admission differs from current source")
    go = json.loads(go_path.read_text())
    require(go.get("schema") == "whisper_padding_executable_go_v1"
            and go.get("status") == "GO"
            and go.get("admission_sha256") == sha_file(admission_path)
            and go.get("sources") == admission["sources"],
            "independent executable GO not bound to source")
    for path in (admission_path, go_path):
        rel = str(path.resolve().relative_to(ROOT))
        require(subprocess.run(("git", "ls-files", "--error-unmatch", "--", rel),
                               cwd=ROOT, capture_output=True, check=False).returncode == 0,
                "uncommitted admission/GO")
        for flag in ((), ("--cached",)):
            require(subprocess.run(("git", "diff", *flag, "--quiet", "--", rel),
                                   cwd=ROOT, check=False).returncode == 0,
                    "dirty admission/GO")
    review_commit = subprocess.check_output(("git", "log", "-1", "--format=%H", "--",
                                             str(go_path.relative_to(ROOT))),
                                            cwd=ROOT, text=True).strip()
    require(len(review_commit) == 40, "GO commit missing")
    for path in [admission_path, DESIGN_GO, HOST_SOURCE,
                 *(HERE / name for name in SOURCE_NAMES)]:
        rel = str(path.resolve().relative_to(ROOT))
        source_commit = subprocess.check_output(("git", "log", "-1", "--format=%H", "--",
                                                 rel), cwd=ROOT, text=True).strip()
        require(source_commit != review_commit and subprocess.run(
            ("git", "merge-base", "--is-ancestor", source_commit, review_commit),
            cwd=ROOT, check=False).returncode == 0,
            f"GO does not postdate frozen source: {path.name}")
    return admission


def call_name(row: dict) -> str:
    return f"{row['site']}_{row['case_id']}_{row['arm']}"


def output_bytes(directory: Path) -> int:
    return sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())


class Budget:
    def __init__(self, output: Path) -> None:
        self.output = output
        self.start = time.monotonic()
        self.attempted = 0

    def check(self, *, before_forward: bool = False, proposed_bytes: int = 0) -> None:
        require(time.monotonic() - self.start <= MAX_SECONDS, "wall cap exceeded")
        require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss <= MAX_RSS_BYTES,
                "worker RSS cap exceeded")
        require(output_bytes(self.output) + proposed_bytes + EVIDENCE_RESERVE_BYTES
                <= MAX_EVIDENCE_BYTES, "evidence reserve exceeded")
        if before_forward:
            require(self.attempted < len(plan.schedule()), "forward count exceeded")
            self.attempted += 1

    def append(self, path: Path, value: dict, *, failure_ledger: bool = False) -> None:
        size = len(canonical(value).encode()) + 1
        if failure_ledger:
            require(output_bytes(self.output) + size <= MAX_EVIDENCE_BYTES,
                    "failure ledger exceeds evidence cap")
        else:
            self.check(proposed_bytes=size)
        append_record(path, value)
        require(output_bytes(self.output) <= MAX_EVIDENCE_BYTES,
                "evidence cap exceeded after write")


def save_array(output: Path, relative: str, array: Any, budget: Budget,
               *, processed: bool = False) -> dict:
    import numpy as np
    value = np.ascontiguousarray(array)
    require(value.dtype == np.float32, "sidecar dtype changed")
    if processed:
        require(not np.isnan(value).any() and not np.isposinf(value).any()
                and bool(np.isfinite(value).any()), "invalid processed score")
    else:
        require(bool(np.isfinite(value).all()), "nonfinite raw/state sidecar")
    path = output / relative
    require(path.resolve().is_relative_to(output.resolve())
            and not path.exists() and not path.is_symlink(), "invalid sidecar path")
    budget.check(proposed_bytes=value.nbytes+1024)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.save(stream, value, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    budget.check()
    return {"path": relative, "file_sha256": sha_file(path),
            "tensor_sha256": core.sha_array(value), "shape": list(value.shape),
            "dtype": str(value.dtype), "bytes": path.stat().st_size}


def validate_first_step_arrays(raw: Any, processed: Any, decoder: Any) -> None:
    import numpy as np
    require(raw.dtype == processed.dtype == decoder.dtype == np.float32
            and raw.shape == processed.shape == (plan.VOCAB_SIZE,)
            and decoder.shape == (1, 4, 384), "first-step evidence shape/dtype changed")
    require(bool(np.isfinite(raw).all()) and bool(np.isfinite(decoder).all()),
            "raw logit or decoder state nonfinite")
    require(not np.isnan(processed).any() and not np.isposinf(processed).any()
            and bool(np.isfinite(processed).any()), "processed score NaN/+inf/all suppressed")


class PhaseHooks:
    """Capture both sites in both phases; edit only the scheduled site/phase."""
    def __init__(self, model: Any, row: dict,
                 states: dict[str, dict[str, dict[str, Any]]]):
        self.model, self.row, self.states = model, row, states
        self.phase = "transcribe"
        self.original_detect = None
        self.handles: list[Any] = []
        self.native: dict[str, dict[str, Any]] = {p: {} for p in plan.PHASES}
        self.hook: dict[str, dict[str, dict]] = {p: {} for p in plan.PHASES}
        self.counts: dict[str, dict[str, int]] = {p: {} for p in plan.PHASES}
        self.detection: dict | None = None

    def _site(self, site: str, tensor: Any) -> Any | None:
        import numpy as np
        import torch
        require(torch.is_tensor(tensor) and tensor.device.type == "cpu"
                and tensor.dtype == torch.float32
                and tuple(tensor.shape) == core.expected_shape(site)
                and bool(torch.isfinite(tensor).all()), f"{site} hook geometry changed")
        phase = self.phase
        self.counts[phase][site] = self.counts[phase].get(site, 0) + 1
        require(self.counts[phase][site] == 1, f"repeated {phase}/{site} hook")
        native = np.ascontiguousarray(tensor.detach().numpy().copy())
        self.native[phase][site] = native
        edited = native.copy()
        meta = {"support": [0, 0], "seed": None, "target_l2": 0.0,
                "realized_l2": 0.0, "cosine": None}
        if core.selected(self.row, site, phase):
            case, donor = self.row["case_id"], self.row["donor_id"]
            require(case in self.states and donor in self.states,
                    "edit source states missing")
            recipient_state = self.states[case][site][phase]
            donor_state = self.states[donor][site][phase]
            edited, meta = core.edit(native, recipient_state, donor_state,
                                     self.row, site, phase)
        self.hook[phase][site] = {"calls": 1, "native_sha256": core.sha_array(native),
                                  "edited_sha256": core.sha_array(edited), **meta}
        if core.selected(self.row, site, phase):
            return torch.from_numpy(np.ascontiguousarray(edited)).to(tensor.device)
        return None

    def install(self) -> None:
        encoder = self.model.model.encoder

        def mel(_module: Any, args: tuple[Any, ...]) -> tuple[Any, ...] | None:
            edited = self._site("log_mel", args[0])
            return None if edited is None else (edited, *args[1:])

        def early(_module: Any, args: tuple[Any, ...]) -> tuple[Any, ...] | None:
            edited = self._site("pre_block0", args[0])
            return None if edited is None else (edited, *args[1:])

        self.handles.append(encoder.conv1.register_forward_pre_hook(mel))
        self.handles.append(encoder.layers[0].register_forward_pre_hook(early))
        self.original_detect = self.model.detect_language

        def detect(*args: Any, **kwargs: Any) -> Any:
            self.phase = "detect"
            try:
                result = self.original_detect(*args, **kwargs)
                require(self.detection is not None, "language detection absent")
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
            values = {name: float(vec[token]) for name, token in
                      self.model.generation_config.lang_to_id.items()}
            require(self.detection is None, "repeated language detection")
            self.detection = {"raw_logits_sha256": sha_bytes(vec.numpy().tobytes()),
                              "language_logits": values, "candidate_count": len(values)}

        self.handles.append(self.model.register_forward_hook(language))

    def validated(self) -> None:
        expected = {site: 1 for site in plan.SITES}
        require(self.counts == {p: expected for p in plan.PHASES},
                "two-phase/two-site hook count changed")
        require(self.detection is not None
                and len(self.detection.get("chosen_language_ids", [])) == 1,
                "automatic language detection count changed")

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        if self.original_detect is not None:
            self.model.detect_language = self.original_detect


def save_native(output: Path, case_id: str, captured: dict,
                budget: Budget) -> dict:
    refs = {site: {} for site in plan.SITES}
    for site in plan.SITES:
        for phase in plan.PHASES:
            refs[site][phase] = save_array(
                output, f"native/{case_id}_{site}_{phase}.npy",
                captured[phase][site], budget)
    return refs


def generate_one(row: dict, case: dict, processor: Any, model: Any,
                 states: dict, output_dir: Path, budget: Budget,
                 call_index: int) -> dict:
    import numpy as np
    import torch
    audio = case["pcm"].astype(np.float32) / np.float32(32768.0)
    processed = processor(audio, sampling_rate=16000, return_tensors="pt",
                          return_attention_mask=True)
    features, mask = processed.input_features, processed.attention_mask
    require(tuple(features.shape) == (1, 80, 3000)
            and features.dtype == torch.float32, "processor mel geometry changed")
    geometry = descriptive.whisper_geometry(mask, len(case["pcm"]), 3000)
    require(geometry["active_mel_frames"] == plan.geometry(case["case_id"], "log_mel")["active"]
            and geometry["nominal_active_encoder_rows"]
            == plan.geometry(case["case_id"], "pre_block0")["active"],
            "processor mask geometry changed")
    require(model.generation_config.is_multilingual is True
            and getattr(model.generation_config, "language", None) is None,
            "automatic-language generation config changed")
    hooks = None if row["hook_policy"] == "none" else PhaseHooks(model, row, states)
    try:
        if hooks is not None:
            hooks.install()
        with torch.inference_mode():
            result = model.generate(
                input_features=features, attention_mask=mask,
                task="transcribe", language=None, return_timestamps=False,
                force_unique_generate_call=True, condition_on_prev_tokens=False,
                do_sample=False, num_beams=1, max_new_tokens=plan.MAX_NEW_TOKENS,
                output_scores=True, output_logits=True, output_hidden_states=True,
                return_dict_in_generate=True)
        if hooks is not None:
            hooks.validated()
        output = descriptive.score_whisper(result, processor.tokenizer,
                                           hooks.detection if hooks is not None else None)
        require(len(output["prefix_ids"]) == 4
                and output["prefix_ids"][0]
                == model.generation_config.decoder_start_token_id
                and output["prefix_ids"][2:] == [
                    model.generation_config.task_to_id["transcribe"],
                    model.generation_config.no_timestamps_token_id],
                "resolved generation prefix changed")
        if hooks is not None:
            require(output["prefix_ids"][1]
                    == hooks.detection["chosen_language_ids"][0],
                    "detected language differs from prefix")
        first_raw_meta = output.pop("first_raw_logits")
        first_decoder_meta = output.pop("first_decoder_step")
        stem = f"{call_index:03d}_{call_name(row)}"
        first_raw = np.asarray(first_raw_meta.pop("values"), dtype=np.float32)
        first_decoder = np.asarray(first_decoder_meta.pop("values"), dtype=np.float32)
        first_scores = result.scores[0][0].detach().to("cpu", dtype=torch.float32).numpy()
        validate_first_step_arrays(first_raw, first_scores, first_decoder)
        selected_scores = np.asarray([step["selected_logit"]
                                      for step in output["score_rows"]], dtype=np.float32)
        require(bool(np.isfinite(selected_scores).all()),
                "selected token has nonfinite processed score")
        refs = {
            "first_raw": save_array(output_dir, f"raw/{stem}_first.npy", first_raw, budget),
            "first_processed": save_array(output_dir, f"processed/{stem}_first.npy",
                                          first_scores, budget, processed=True),
            "first_decoder": save_array(output_dir, f"decoder/{stem}_first.npy",
                                        first_decoder, budget),
            "selected_scores": save_array(output_dir, f"processed/{stem}_selected.npy",
                                          selected_scores, budget, processed=True),
            "full_raw": [],
        }
        require(refs["first_raw"]["tensor_sha256"] == first_raw_meta["sha256"]
                and refs["first_decoder"]["tensor_sha256"] == first_decoder_meta["sha256"]
                and refs["first_processed"]["tensor_sha256"]
                == output["score_rows"][0]["processed_logits_sha256"],
                "first-step sidecar differs from scorer")
        if row["arm"] in ("no_hook", "DD"):
            for step, tensor in enumerate(result.logits):
                raw = tensor[0].detach().to("cpu", dtype=torch.float32).numpy()
                ref = save_array(output_dir, f"raw/{stem}_step_{step:03d}.npy", raw,
                                 budget)
                require(ref["tensor_sha256"] == output["score_rows"][step][
                    "raw_logits_sha256"], "full raw step differs from scorer")
                refs["full_raw"].append(ref)
        return {"output": output, "sidecars": refs,
                "hook": hooks.hook if hooks is not None else {},
                "hook_counts": hooks.counts if hooks is not None else {},
                "native": hooks.native if hooks is not None else {},
                "input_geometry": geometry,
                "input_mask": mask.detach().to("cpu").numpy().astype(np.int32).tolist(),
                "first_raw_meta": first_raw_meta,
                "first_decoder_meta": first_decoder_meta}
    finally:
        if hooks is not None:
            hooks.close()


def same_sequence(left: dict, right: dict) -> bool:
    return (left["sequence_ids"] == right["sequence_ids"]
            and left["actual_generation_steps"] == right["actual_generation_steps"]
            and left["termination"] == right["termination"])


def first_parity(left: dict, right: dict, output_dir: Path) -> dict:
    import numpy as np
    from . import padding_channel_audit_v1 as audit
    a, b = left["sidecars"], right["sidecars"]
    actual = {}
    for name in ("first_raw", "first_processed", "first_decoder"):
        x, y = audit.saved_array(output_dir, a[name]), audit.saved_array(output_dir, b[name])
        actual[name] = (core.processed_score_maxabs(x, y)
                        if name == "first_processed" else core.maxabs_finite(x, y))
    return {"tokens_match": same_sequence(left["output"], right["output"]),
            "language_match": left["output"]["prefix_ids"][1]
            == right["output"]["prefix_ids"][1],
            "maxabs": actual,
            "within_1e-5": all(value <= 1e-5 for value in actual.values())}


def full_raw_parity(left: dict, right: dict, output_dir: Path) -> dict:
    from . import padding_channel_audit_v1 as audit
    first = left["sidecars"]["full_raw"]
    second = right["sidecars"]["full_raw"]
    if len(first) != len(second):
        return {"steps_match": False, "maxabs": None, "within_1e-5": False}
    differences = [core.maxabs_finite(audit.saved_array(output_dir, a),
                                     audit.saved_array(output_dir, b))
                   for a, b in zip(first, second, strict=True)]
    measured = max(differences) if differences else None
    return {"steps_match": True, "maxabs": measured,
            "within_1e-5": measured is not None and measured <= 1e-5}


def run_model(output_dir: Path, admission_path: Path, go_path: Path) -> dict:
    """Only path that may load candidate weights; fixed GO validated first."""
    import numpy as np
    import torch
    admission = validate_go(admission_path, go_path)
    require(output_dir.resolve() == (ATTEMPT / "worker_output").resolve()
            and output_dir.parent.is_dir() and not output_dir.exists(),
            "fixed exclusive worker output path required")
    rows = plan.schedule()
    cases = {case["case_id"]: case for case in descriptive.trusted_cases()}
    require(set(cases) == set(plan.population()), "case census changed")
    output_dir.mkdir()
    records = output_dir / "records.jsonl"
    progress = output_dir / "call_progress.jsonl"
    write_once(records, {"type": "start", "schema": "whisper_padding_records_v1",
                         "planned_calls": len(rows), "schedule_sha256": plan.schedule_sha256(),
                         "admission_sha256": sha_file(admission_path),
                         "go_sha256": sha_file(go_path)})
    with progress.open("x", encoding="utf-8") as stream:
        stream.flush()
        os.fsync(stream.fileno())
    budget = Budget(output_dir)
    states: dict[str, dict[str, dict[str, Any]]] = {}
    native_refs_by_case: dict[str, dict] = {}
    donor_baselines: dict[str, dict] = {}
    recipient_baselines: dict[tuple[str, str], dict] = {}
    mask_by_donor: dict[str, list[int]] = {}
    failed_index: int | None = None
    try:
        torch.set_num_threads(4)
        processor, model = descriptive._snapshot_load("whisper_multilingual",
                                                       admission["snapshot"])
        require(model.training is False and all(
            parameter.device.type == "cpu" and parameter.dtype == torch.float32
            for parameter in model.parameters()), "model not CPU FP32 eval")
        for index, row in enumerate(rows, 1):
            failed_index = index
            name = call_name(row)
            budget.append(progress, {"type": "attempt", "call_index": index,
                                     "call_name": name})
            budget.check(before_forward=True)
            started = time.monotonic()
            observed = generate_one(row, cases[row["case_id"]], processor, model,
                                    states, output_dir, budget, index)
            output = observed["output"]
            donor_id = row["donor_id"]
            mask = observed["input_mask"][0]
            if row["arm"] == "no_hook" and row["case_id"] == donor_id:
                require(donor_id not in donor_baselines,
                        "duplicate donor baseline")
                donor_baselines[donor_id] = observed
                mask_by_donor[donor_id] = mask
            else:
                require(mask == mask_by_donor[donor_id],
                        "recipient/donor processor masks differ")
            native_refs = {}
            if row["arm"] in ("read_only_donor_capture", "read_only"):
                case_id = row["case_id"]
                captured = {site: {phase: observed["native"][phase][site]
                                   for phase in plan.PHASES} for site in plan.SITES}
                if case_id not in states:
                    states[case_id] = captured
                    native_refs = save_native(output_dir, case_id,
                                              observed["native"], budget)
                    native_refs_by_case[case_id] = native_refs
                else:
                    require(all(np.array_equal(captured[site][phase],
                                               states[case_id][site][phase])
                                for site in plan.SITES for phase in plan.PHASES),
                            "duplicate read-only native capture changed")
            parity = {}
            if row["arm"] == "read_only_donor_capture":
                parity["donor_no_hook"] = first_parity(
                    observed, donor_baselines[donor_id], output_dir)
            elif row["arm"] == "RR":
                key = (row["site"], row["case_id"])
                require(key not in recipient_baselines, "duplicate RR baseline")
                recipient_baselines[key] = observed
                other_site = next(site for site in plan.SITES if site != row["site"])
                other = recipient_baselines.get((other_site, row["case_id"]))
                if other is not None:
                    parity["cross_site_no_hook"] = first_parity(observed, other,
                                                                output_dir)
            elif row["arm"] in ("read_only", "self"):
                parity["recipient_no_hook"] = first_parity(
                    observed, recipient_baselines[(row["site"], row["case_id"])],
                    output_dir)
            elif row["arm"] == "DD":
                donor = donor_baselines[donor_id]
                parity["donor_no_hook"] = first_parity(observed, donor, output_dir)
                parity["donor_full_raw"] = full_raw_parity(observed, donor,
                                                            output_dir)
            budget.append(records, {"type": "forward", "call_index": index,
                                    "call_name": name, "plan_row": row,
                                    "output": output, "sidecars": observed["sidecars"],
                                    "first_raw_meta": observed["first_raw_meta"],
                                    "first_decoder_meta": observed["first_decoder_meta"],
                                    "hook": observed["hook"],
                                    "hook_counts": observed["hook_counts"],
                                    "native_state_refs": native_refs,
                                    "input_geometry": observed["input_geometry"],
                                    "parity": parity,
                                    "elapsed_seconds": time.monotonic()-started})
            budget.append(progress, {"type": "complete", "call_index": index,
                                     "call_name": name})
            failed_index = None
            budget.check()
        require(len(native_refs_by_case) == 20
                and len(recipient_baselines) == 36
                and len(donor_baselines) == 2,
                "final capture/baseline census changed")
        budget.append(records, {"type": "complete", "planned_calls": 436,
                                "attempted_calls": budget.attempted,
                                "evidence_bytes": output_bytes(output_dir)})
        return {"status": "worker_complete", "forwards": 436,
                "evidence_bytes": output_bytes(output_dir)}
    except BaseException as error:
        start = failed_index + 1 if failed_index is not None else budget.attempted + 1
        budget.append(records, {"type": "failure", "error_type": type(error).__name__,
                                "error": str(error),
                                "failed_attempted_call_name": (call_name(rows[failed_index-1])
                                                              if failed_index else None),
                                "censored_call_names": [call_name(row)
                                                        for row in rows[start-1:]],
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
        require(args.output.resolve() == ADMISSION.resolve()
                and not args.output.exists(), "fixed exclusive admission path")
        write_once(args.output, preflight_receipt())
        print(canonical({"status": "source_only_no_model_load",
                         "admission": str(args.output)}))
    else:
        result = run_model(args.output, args.admission, args.go)
        print(canonical(result))


if __name__ == "__main__":
    main()
