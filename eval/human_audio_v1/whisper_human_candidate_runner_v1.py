"""Candidate one-shot development runner; no inference without a separate GO.

The admission command is source-only. The run command checks the independently
committed GO before importing target model classes, loading weights, or creating
output. Source validation can import libraries to verify pinned versions.
Sealed-test inputs are deliberately unsupported in this version.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import time
import wave
from pathlib import Path
from typing import Any

from . import whisper_human_candidate_gate_v1 as gate
from . import whisper_human_preflight_v1 as frozen

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def need(ok: bool, message: str) -> None:
    gate.need(ok, message)


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def write_once(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        stream.write(canonical(value) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def append(path: Path, value: Any) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(canonical(value) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def admission_cli(third: Path, a: Path, b: Path, c: Path | None,
                  receipt: Path, output: Path) -> dict:
    need(output.resolve() == gate.ADMISSION.resolve(), "fixed admission path required")
    need(not output.exists(), "admission already exists")
    value = gate.admission(third=third, a=a, b=b, c=c, receipt=receipt)
    write_once(output, value)
    return {"status": value["status"], "path": str(output),
            "sha256": gate.sha(output)}


class Budget:
    def __init__(self, attempt: Path) -> None:
        self.attempt = attempt
        self.started = time.monotonic()
        self.calls = 0

    def check(self, *, forward: bool = False, extra: int = 0) -> None:
        need(time.monotonic() - self.started <= gate.LIMITS["wall_seconds"],
             "wall cap exceeded")
        need(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss <=
             gate.LIMITS["worker_rss_bytes"], "RSS cap exceeded")
        size = sum(p.stat().st_size for p in self.attempt.rglob("*") if p.is_file())
        need(size + extra + 4096 < gate.LIMITS["evidence_bytes"],
             "evidence cap exceeded")
        if forward:
            need(self.calls < gate.LIMITS["max_model_calls"], "model-call cap exceeded")
            self.calls += 1

    def npy(self, stem: str, value: Any, *, allow_neginf: bool = False) -> dict:
        import numpy as np
        array = np.ascontiguousarray(value)
        valid_values = (bool(np.isfinite(array).all()) if not allow_neginf
                        else bool(np.isfinite(array).any()) and
                        not bool(np.isnan(array).any()) and
                        not bool(np.isposinf(array).any()))
        need(array.dtype in (np.float32, np.int64, np.int32, np.uint8)
             and valid_values, "sidecar dtype or finite check failed")
        self.check(extra=array.nbytes + 1024)
        relative = f"sidecars/{stem}.npy"
        path = self.attempt / relative
        need(not path.exists(), "sidecar already exists")
        with path.open("xb") as stream:
            np.save(stream, array, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        self.check()
        return {"path": relative, "sha256": gate.sha(path),
                "shape": list(array.shape), "dtype": str(array.dtype)}


def _pcm(row: dict) -> Any:
    import numpy as np
    path = ROOT / row["wav"]
    need(gate.sha(path) == row["wav_sha256"], "source WAV bytes changed")
    with wave.open(str(path), "rb") as clip:
        need((clip.getnchannels(), clip.getsampwidth(), clip.getframerate(),
              clip.getnframes()) == (1, 2, 16000, row["frames"]),
             "source WAV geometry changed")
        pcm = np.frombuffer(clip.readframes(row["frames"]), dtype="<i2").copy()
    return pcm


def _case_id(row: dict) -> str:
    return f"{row['utterance_id']}|{row['condition']}"


def _feature_path(processor: Any) -> None:
    feature = processor.feature_extractor
    need(feature.sampling_rate == 16000 and feature.hop_length == 160
         and feature.n_fft == 400 and feature.nb_max_frames == 3000,
         "pinned mel/time mapping changed")


class Hooks:
    """Record both encoder phases; edit only generation at one frozen site."""
    def __init__(self, model: Any, site: str | None = None, arm: str | None = None,
                 donor: Any = None, boundary: list[int] | None = None,
                 speech: list[int] | None = None) -> None:
        self.model, self.site, self.arm = model, site, arm
        self.donor, self.boundary, self.speech = donor, boundary, speech
        self.phase = "transcribe"
        self.native: dict[str, dict[str, Any]] = {"detect": {}, "transcribe": {}}
        self.edited: dict[str, dict[str, Any]] = {"detect": {}, "transcribe": {}}
        self.counts: dict[str, dict[str, int]] = {"detect": {}, "transcribe": {}}
        self.edit_l2: float | None = None
        self.boundary_delta_l2: float | None = None
        self.detection: dict | None = None
        self.handles = []
        self.original_detect = None

    def _capture(self, site: str, tensor: Any) -> Any:
        import numpy as np
        import torch
        need(torch.is_tensor(tensor) and tensor.device.type == "cpu"
             and tensor.dtype == torch.float32 and tuple(tensor.shape) == (1, 1500, 384)
             and bool(torch.isfinite(tensor).all()), "encoder state topology changed")
        phase = self.phase
        self.counts[phase][site] = self.counts[phase].get(site, 0) + 1
        need(self.counts[phase][site] == 1, "encoder hook repeated")
        native = np.ascontiguousarray(tensor.detach().numpy().copy())
        edited = native.copy()
        self.native[phase][site] = native
        if phase == "transcribe" and site == self.site:
            need(self.donor is not None and self.donor.shape == native.shape,
                 "donor state unavailable")
            if self.arm == "boundary":
                edited[:, self.boundary, :] = self.donor[:, self.boundary, :]
            elif self.arm == "self":
                pass
            elif self.arm == "source_donor":
                edited[:, self.speech, :] = self.donor[:, self.speech, :]
            elif self.arm == "displaced_delta":
                delta = self.donor[:, self.boundary, :].astype(np.float64) - \
                    native[:, self.boundary, :].astype(np.float64)
                target_norm = float(np.linalg.norm(delta.ravel()))
                self.boundary_delta_l2 = target_norm
                if target_norm:
                    shifted = native[:, self.speech, :].astype(np.float64) + delta
                    edited[:, self.speech, :] = shifted.astype(np.float32)
            elif self.arm == "full_donor":
                edited = self.donor.copy()
            else:
                raise ValueError("unknown causal arm")
            self.edit_l2 = float(np.linalg.norm(
                (edited-native).astype(np.float64).ravel()))
            if self.arm == "displaced_delta":
                need(abs(self.edit_l2 - self.boundary_delta_l2) <=
                     1e-4 * max(self.boundary_delta_l2, 1.0),
                     "displaced delta norm differs from boundary edit")
        self.edited[phase][site] = edited
        return torch.from_numpy(edited) if phase == "transcribe" and site == self.site else None

    def install(self) -> None:
        enc = self.model.model.encoder

        def pre(_module: Any, args: tuple[Any, ...]) -> tuple[Any, ...] | None:
            result = self._capture("post_conv_pos", args[0])
            return (result, *args[1:]) if result is not None else None

        def post(_module: Any, _args: Any, result: Any) -> Any:
            return self._capture("post_block0", result)

        self.handles.append(enc.layers[0].register_forward_pre_hook(pre))
        self.handles.append(enc.layers[0].register_forward_hook(post))
        self.original_detect = self.model.detect_language

        def detect(*args: Any, **kwargs: Any) -> Any:
            self.phase = "detect"
            try:
                result = self.original_detect(*args, **kwargs)
                need(self.detection is not None, "language detection logits missing")
                self.detection["chosen_language_ids"] = [int(x) for x in result.tolist()]
                return result
            finally:
                self.phase = "transcribe"

        self.model.detect_language = detect

        def language(_module: Any, _args: Any, result: Any) -> None:
            if self.phase == "detect":
                import torch
                vec = result.logits[0, -1].detach().to("cpu", dtype=torch.float32)
                need(self.detection is None, "language detection repeated")
                self.detection = {"raw_logits_sha256": gate.sha_bytes(vec.numpy().tobytes())}

        self.handles.append(self.model.register_forward_hook(language))

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        if self.original_detect is not None:
            self.model.detect_language = self.original_detect

    def validated(self) -> None:
        need(self.counts == {"detect": {"post_conv_pos": 1, "post_block0": 1},
                             "transcribe": {"post_conv_pos": 1, "post_block0": 1}},
             "two-phase encoder hook topology changed")
        need(self.detection is not None
             and len(self.detection.get("chosen_language_ids", [])) == 1,
             "detected language missing")


def _one(row: dict, processor: Any, model: Any, budget: Budget, stem: str,
         *, site: str | None = None, arm: str | None = None, donor: Any = None,
         boundary: list[int] | None = None, speech: list[int] | None = None) -> tuple[dict, dict]:
    import numpy as np
    import torch
    from models.second_architecture.nonspeech_probe_v1 import descriptive_worker_v1 as old
    from . import blinded_listening_v1 as blind

    pcm = _pcm(row)
    processed = processor(pcm.astype(np.float32) / np.float32(32768.0),
                          sampling_rate=16000, return_tensors="pt",
                          return_attention_mask=True)
    need(processed.input_features.dtype == torch.float32
         and tuple(processed.input_features.shape) == (1, 80, 3000),
         "mel geometry changed")
    geometry = old.whisper_geometry(processed.attention_mask, len(pcm))
    hooks = Hooks(model, site, arm, donor, boundary, speech)
    try:
        hooks.install()
        budget.check(forward=True)
        with torch.inference_mode():
            result = model.generate(
                input_features=processed.input_features, attention_mask=processed.attention_mask,
                task="transcribe", language=None, return_timestamps=False,
                force_unique_generate_call=True, condition_on_prev_tokens=False,
                do_sample=False, num_beams=1, max_new_tokens=128,
                output_scores=True, output_logits=True, output_hidden_states=True,
                return_dict_in_generate=True)
        hooks.validated()
        output = old.score_whisper(result, processor.tokenizer, hooks.detection)
        need(output["prefix_ids"][1] == hooks.detection["chosen_language_ids"][0],
             "detected language/prefix mismatch")
        need(output["prefix_ids"][:1] == [model.generation_config.decoder_start_token_id]
             and output["prefix_ids"][2:] == [
                 model.generation_config.task_to_id["transcribe"],
                 model.generation_config.no_timestamps_token_id],
             "transcription prefix changed")
        raw = np.ascontiguousarray(result.logits[0][0].detach().numpy())
        scores = np.ascontiguousarray(result.scores[0][0].detach().numpy())
        need(raw.shape == scores.shape == (51865,), "first-step vocabulary width changed")
        need(gate.sha_bytes(raw.tobytes()) == output["score_rows"][0]["raw_logits_sha256"]
             and gate.sha_bytes(scores.tobytes()) ==
             output["score_rows"][0]["processed_logits_sha256"],
             "first-step vectors differ from generated scores")
        output.pop("first_decoder_step")
        output.pop("first_raw_logits")
        output["normalized_transcript"] = blind.normalize(output["transcript"])
        output["first_raw"] = budget.npy(stem + "_raw", raw)
        output["first_processed"] = budget.npy(stem + "_processed", scores,
                                                allow_neginf=True)
        sidecars = {"mel": budget.npy(stem + "_mel", processed.input_features[0].numpy()),
                    "mask": budget.npy(stem + "_mask", processed.attention_mask[0].numpy().astype(np.int32))}
        for phase in ("detect", "transcribe"):
            for name in ("post_conv_pos", "post_block0"):
                if phase == "transcribe" or site == name:
                    sidecars[f"{phase}_{name}_native"] = budget.npy(
                        stem + f"_{phase}_{name}_native", hooks.native[phase][name])
                if phase == "transcribe" and site == name:
                    sidecars[f"{phase}_{name}_edited"] = budget.npy(
                        stem + f"_{phase}_{name}_edited", hooks.edited[phase][name])
        record = {"case_id": _case_id(row), "wav": row["wav"],
                  "wav_sha256": row["wav_sha256"], "frames": row["frames"],
                  "split": row["split"], "site": site, "arm": arm,
                  "output": output, "geometry": geometry, "hook_counts": hooks.counts,
                  "edit_l2": hooks.edit_l2,
                  "boundary_delta_l2": hooks.boundary_delta_l2,
                  "sidecars": sidecars}
        return record, hooks.native["transcribe"]
    finally:
        hooks.close()


def _equal_first(a: dict, b: dict) -> bool:
    return (a["output"]["first_raw"]["sha256"] == b["output"]["first_raw"]["sha256"]
            and a["output"]["first_processed"]["sha256"] ==
            b["output"]["first_processed"]["sha256"])


def _complete_path(a: dict, b: dict) -> bool:
    return (a["output"]["termination"] == b["output"]["termination"] == "eos"
            and _same_path(a, b))


def _same_path(a: dict, b: dict) -> bool:
    return (a["output"]["termination"] == b["output"]["termination"]
            and a["output"]["sequence_ids"] == b["output"]["sequence_ids"])


def _pair_rows(plan: dict, pair: dict) -> tuple[dict, dict]:
    rows = [row for row in plan["calls"] if row["utterance_id"] == pair["utterance_id"]]
    zero = next(row for row in rows if row["condition"] == "z32000")
    noise = next(row for row in rows if row["wav_sha256"] == pair["noise_wav_sha256"])
    need(zero["wav_sha256"] == pair["zero_wav_sha256"], "pair source binding changed")
    return zero, noise


def _target(go: dict) -> dict:
    """The only target loading path. Called strictly after validate_go."""
    import numpy as np
    from models.second_architecture.nonspeech_probe_v1 import descriptive_worker_v1 as old
    from models.second_architecture.nonspeech_probe_v1 import whisper_tail_causal_host_v1 as host

    before = host.quick_probe(initial=True)
    need(before["external_connected"] is True, "AC power required")
    plan = frozen.frozen_plan()
    need(plan["counts"]["development_comparisons"] == 4, "split count changed")
    assets = old.verify_local_assets("whisper_multilingual")
    need(assets == go["admission"]["local_assets"], "target asset drift")
    need(old.verify_installed_sources() == go["admission"]["installed"],
         "installed source drift")
    attempt = gate.ATTEMPT
    attempt.parent.mkdir(parents=True, exist_ok=True)
    attempt.mkdir(exist_ok=False)
    (attempt / "sidecars").mkdir()
    write_once(attempt / "launch.json", {"go_sha256": go["go_sha256"],
                                          "go_commit": go["go_commit"],
                                          "admission_sha256": gate.sha(gate.ADMISSION),
                                          "host_before": before})
    ledger = attempt / "records.jsonl"
    ledger.touch(exist_ok=False)
    budget = Budget(attempt)
    processor, model = old._snapshot_load("whisper_multilingual", assets)
    _feature_path(processor)
    calls = [row for row in plan["calls"] if row["split"] == "development"]
    need(len(calls) == 8, "development call census changed")
    records, states = {}, {}
    try:
        for index, row in enumerate(calls):
            key = _case_id(row)
            record, state = _one(row, processor, model, budget, f"d{index:03d}")
            append(ledger, {"type": "descriptive", "index": index, **record})
            records[key], states[key] = record, state
        for index, row in enumerate(row for row in calls if row["condition"] == "base"):
            record, _ = _one(row, processor, model, budget, f"base_repeat{index:02d}")
            append(ledger, {"type": "base_repeat", "index": index,
                            "stable_path": _same_path(records[_case_id(row)], record),
                            "stable_first": _equal_first(records[_case_id(row)], record),
                            **record})
        human = json.loads(Path(go["admission"]["human_review"]["receipt_path"]).read_text())
        screened = {case["case_id"]: case["listening_screen_pass"]
                    for case in human["cases"]}
        need(len(screened) == 16, "human case census changed")
        eligible = 0
        qualified = 0
        for pair_index, pair in enumerate(plan["comparisons"]):
            if pair["split"] != "development":
                continue
            zero, noise = _pair_rows(plan, pair)
            key_zero, key_noise = _case_id(zero), _case_id(noise)
            a, b = records[key_zero], records[key_noise]
            reasons = []
            if not screened[key_zero] or not screened[key_noise]:
                reasons.append("human_screen_failed")
            if a["output"]["termination"] != "eos" or b["output"]["termination"] != "eos":
                reasons.append("unpatched_not_both_eos")
            if a["output"]["prefix_ids"] != b["output"]["prefix_ids"]:
                reasons.append("prefix_changed")
            if a["output"]["language_detection"]["chosen_language_ids"] != \
                    b["output"]["language_detection"]["chosen_language_ids"]:
                reasons.append("language_changed")
            if a["output"]["normalized_transcript"] == b["output"]["normalized_transcript"]:
                reasons.append("complete_answer_equal")
            mel_a = np.load(attempt / a["sidecars"]["mel"]["path"], allow_pickle=False)
            mel_b = np.load(attempt / b["sidecars"]["mel"]["path"], allow_pickle=False)
            difference = np.linalg.norm((mel_b-mel_a).astype(np.float64), axis=0)
            append(ledger, {"type": "mel_difference", "pair_index": pair_index,
                            "all_frame_l2": difference.tolist(),
                            "nonzero_frame_count": int(np.count_nonzero(difference)),
                            "boundary_rows": pair["boundary_rows"]})
            if not reasons:
                for repeat in range(2):
                    for name, row, original in (("zero", zero, a), ("noise", noise, b)):
                        record, _ = _one(row, processor, model, budget,
                                          f"replay_p{pair_index}_{name}_{repeat}")
                        stable = _complete_path(original, record) and _equal_first(original, record)
                        append(ledger, {"type": "eligibility_replay", "pair_index": pair_index,
                                        "repeat": repeat, "kind": name,
                                        "stable": stable, **record})
                        if not stable:
                            reasons.append("replay_changed")
            append(ledger, {"type": "eligibility", "pair_index": pair_index,
                            "zero_case_id": key_zero, "noise_case_id": key_noise,
                            "eligible": not reasons, "reasons": sorted(set(reasons))})
            if reasons:
                continue
            eligible += 1
            for site in ("post_conv_pos", "post_block0"):
                for direction, recipient, donor_row, recipient_ref, donor_ref in (
                    ("zero_to_noise", noise, zero, b, a),
                    ("noise_to_zero", zero, noise, a, b),
                ):
                    results = {}
                    for arm in ("boundary", "self", "source_donor",
                                "displaced_delta", "full_donor"):
                        record, _ = _one(
                            recipient, processor, model, budget,
                            f"patch_p{pair_index}_{site}_{direction}_{arm}",
                            site=site, arm=arm, donor=states[_case_id(donor_row)][site],
                            boundary=pair["boundary_rows"],
                            speech=pair["displaced_source_rows"])
                        need(record["output"]["prefix_ids"] ==
                             recipient_ref["output"]["prefix_ids"],
                             "patch changed recipient prefix")
                        append(ledger, {"type": "causal", "pair_index": pair_index,
                                        "direction": direction, **record})
                        results[arm] = record
                    full_valid = (_complete_path(results["full_donor"], donor_ref)
                                  and results["full_donor"]["output"]["first_raw"]["sha256"]
                                  == donor_ref["output"]["first_raw"]["sha256"])
                    controls_valid = all(_complete_path(results[name], recipient_ref)
                                         for name in ("self", "source_donor", "displaced_delta"))
                    transferred = _complete_path(results["boundary"], donor_ref)
                    local_raw = np.load(attempt / results["boundary"]["output"]["first_raw"]["path"],
                                        allow_pickle=False)
                    donor_raw = np.load(attempt / donor_ref["output"]["first_raw"]["path"],
                                        allow_pickle=False)
                    local_raw_l2 = float(np.linalg.norm(
                        local_raw.astype(np.float64) - donor_raw.astype(np.float64)))
                    cell_qualified = full_valid and controls_valid and transferred
                    qualified += int(cell_qualified)
                    append(ledger, {"type": "causal_cell", "pair_index": pair_index,
                                    "direction": direction, "site": site,
                                    "full_positive_valid": full_valid,
                                    "location_controls_valid": controls_valid,
                                    "boundary_transfer": transferred,
                                    "boundary_first_raw_l2_to_donor": local_raw_l2,
                                    "qualified": cell_qualified})
        budget.check()
        after = host.quick_probe(initial=False)
        need(after["external_connected"] is True, "AC power lost")
        write_once(attempt / "complete.json", {"schema": "whisper_human_dev_candidate_v1",
                                               "status": "completed_pending_independent_audit",
                                               "development_pairs": 4,
                                               "eligible_pairs": eligible,
                                               "qualified_cells": qualified,
                                               "model_calls": budget.calls,
                                               "host_after": after})
        return {"status": "completed_pending_independent_audit", "attempt": str(attempt),
                "eligible_pairs": eligible, "qualified_cells": qualified,
                "model_calls": budget.calls}
    except BaseException as error:
        write_once(attempt / "failure.json", {"status": "invalid_or_partial",
                                              "error_type": type(error).__name__,
                                              "error": str(error), "model_calls": budget.calls})
        raise


def run() -> dict:
    """Bound the single worker live, including a hung forward pass."""
    gate.validate_go()
    from models.second_architecture.nonspeech_probe_v1 import whisper_tail_causal_host_v1 as host

    before = host.quick_probe(initial=True)
    need(before["external_connected"] is True, "AC power required")
    started = time.monotonic()
    last_host = started
    process = subprocess.Popen(
        (sys.executable, "-m", "eval.human_audio_v1.whisper_human_candidate_runner_v1",
         "worker"), cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    reason = "unknown"
    peak_rss = 0
    try:
        while process.poll() is None:
            elapsed = time.monotonic() - started
            if elapsed > gate.LIMITS["wall_seconds"]:
                reason = "wall_limit"
                break
            memory = subprocess.run(("ps", "-o", "rss=", "-p", str(process.pid)),
                                    capture_output=True, text=True, timeout=5)
            if process.poll() is not None:
                break
            need(memory.returncode == 0 and memory.stdout.strip().isdigit(),
                 "worker RSS probe failed")
            peak_rss = max(peak_rss, int(memory.stdout.strip()) * 1024)
            if peak_rss > gate.LIMITS["worker_rss_bytes"]:
                reason = "rss_limit"
                break
            if gate.ATTEMPT.exists():
                size = sum(path.stat().st_size for path in gate.ATTEMPT.rglob("*")
                           if path.is_file())
                if size > gate.LIMITS["evidence_bytes"]:
                    reason = "evidence_limit"
                    break
            if time.monotonic() - last_host >= 10:
                sample = host.quick_probe(initial=False)
                need(sample["external_connected"] is True, "AC power lost")
                last_host = time.monotonic()
            time.sleep(0.25)
        if reason != "unknown":
            process.terminate()
        try:
            out, err = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            out, err = process.communicate(timeout=5)
        if reason != "unknown":
            if gate.ATTEMPT.is_dir():
                write_once(gate.ATTEMPT / "parent_stop.json",
                           {"status": "invalid_or_partial", "reason": reason,
                            "peak_worker_rss_bytes": peak_rss})
            raise RuntimeError(f"bounded worker stopped: {reason}")
        need(process.returncode == 0 and (gate.ATTEMPT / "complete.json").is_file(),
             f"worker failed: {err[-2000:]}")
        after = host.quick_probe(initial=False)
        need(after["external_connected"] is True, "AC power lost")
        result = json.loads(out.strip())
        need(result.get("status") == "completed_pending_independent_audit",
             "worker result invalid")
        result["peak_worker_rss_bytes"] = peak_rss
        return result
    except BaseException:
        if reason == "unknown" and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    admit = sub.add_parser("admission")
    for flag in ("third", "review-a", "review-b", "receipt"):
        admit.add_argument("--" + flag, type=Path, required=True)
    admit.add_argument("--review-c", type=Path)
    admit.add_argument("--output", type=Path, default=gate.ADMISSION)
    sub.add_parser("run")
    sub.add_parser("worker")
    args = parser.parse_args()
    if args.mode == "admission":
        result = admission_cli(args.third, args.review_a, args.review_b,
                               args.review_c, args.receipt, args.output)
    elif args.mode == "run":
        result = run()
    else:
        result = _target(gate.validate_go())
    print(canonical(result))


if __name__ == "__main__":
    main()
