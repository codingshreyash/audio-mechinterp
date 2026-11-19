"""No-weight spoken-input bridge for the frozen Qwen action method.

This candidate can prepare the six development audio prompts only after the
real two-listener QA and its separate independent review. It never loads model
weights, calls Thinker, or opens a locked-case WAV. A later executable review
and GO must bind any model runner that consumes its output.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from models.qwen_omni.adapter import prepare_audio_batch
from models.qwen_omni.time_map.index import build_row_index

from . import human_qa_v2, runner_contract_v1
from .endpoint import load_spec, validate_local_source

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = HERE / "runs/source_render_v2/attempt_001"
REVIEW_SCHEMA = "qwen_action_audio_human_qa_independent_review_v1"
QA_PASS = "human_qa_pass_needs_independent_review"
EXPECTED_BATCH_KEYS = {"input_ids", "attention_mask", "input_features",
                       "feature_attention_mask"}
SNAPSHOT_MANIFEST = HERE / "TYPED_PILOT_CUDA_HF_SNAPSHOT_MANIFEST_V2.json"
PROCESSOR_FILES = ("added_tokens.json", "chat_template.json", "config.json",
                   "merges.txt", "preprocessor_config.json",
                   "special_tokens_map.json", "tokenizer.json",
                   "tokenizer_config.json", "vocab.json")


def need(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def sha(path: Path) -> str:
    need(path.is_file() and not path.is_symlink(), f"missing or linked file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1 << 20), b""):
            digest.update(part)
    return digest.hexdigest()


def read(path: Path) -> dict[str, Any]:
    need(path.is_file() and not path.is_symlink()
         and path.stat().st_size <= 2 * 1024 * 1024,
         f"missing, linked or oversized JSON: {path}")
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result = {}
        for key, item in pairs:
            need(key not in result, "duplicate JSON key")
            result[key] = item
        return result
    value = json.loads(path.read_text(encoding="utf-8"),
                       object_pairs_hook=unique)
    need(isinstance(value, dict), "JSON root must be an object")
    return value


@dataclass(frozen=True)
class QAFiles:
    first_a: Path
    first_b: Path
    attestation: Path
    report: Path
    independent_review: Path
    third: Path | None = None


def verify_processor_snapshot(snapshot: Path) -> dict[str, str]:
    """Hash only processor assets from the pinned HF inventory, no weights."""
    spec = load_spec()
    manifest = read(SNAPSHOT_MANIFEST)
    need(snapshot.is_dir() and snapshot.name == spec["model_revision"]
         and manifest.get("model_id") == spec["model_id"]
         and manifest.get("revision") == spec["model_revision"],
         "processor snapshot identity changed")
    expected = {row["path"]: row for row in manifest["files"]}
    need(all(name in expected for name in PROCESSOR_FILES),
         "processor asset inventory incomplete")
    found = {}
    for name in PROCESSOR_FILES:
        path = snapshot / name
        need(path.is_file() and path.stat().st_size == expected[name]["size"],
             f"processor asset missing or changed: {name}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        need(digest == expected[name]["sha256"],
             f"processor asset hash changed: {name}")
        found[name] = digest
    return found


def source_plan() -> list[dict[str, Any]]:
    """Re-audit all source bytes, then return only D_* Piper audio cases."""
    source = runner_contract_v1.source_preflight()
    need(source["candidate_inference_authorized"] is False
         and source["human_reviews"] == 0, "source preflight changed")
    manifest = read(SOURCE / "manifest.json")
    wanted = runner_contract_v1.planned_case_ids()["development_audio"]
    rows = [row for row in manifest["rows"] if row["split"] == "development"]
    need([row["case_id"] for row in rows] == wanted and len(rows) == 6,
         "development order or census changed")
    spec = load_spec()
    scripts = {row["id"]: row for row in spec["scripts"]}
    result = []
    for row in rows:
        case_id = row["case_id"]
        need(row["script_id"] in scripts
             and scripts[row["script_id"]]["split"] == "development"
             and row["text"] == scripts[row["script_id"]]["text"]
             and row["label"] == scripts[row["script_id"]]["label"]
             and row["voice"] == "piper_en_US_amy_medium"
             and row["final_frames"] == 256000,
             "development source mapping changed")
        path = SOURCE / row["final"]
        need(path.resolve().parent == SOURCE.resolve()
             and path.name == f"{case_id}.final.wav"
             and sha(path) == row["final_sha256"], "source WAV changed")
        with wave.open(str(path), "rb") as wav:
            need((wav.getnchannels(), wav.getsampwidth(), wav.getframerate(),
                  wav.getnframes()) == (1, 2, 16000, 256000),
                 "development audio geometry changed")
        result.append({"case_id": case_id, "script_id": row["script_id"],
                       "label": row["label"], "wav_path": str(path.resolve()),
                       "wav_sha256": row["final_sha256"],
                       "source_frames": row["source_frames"],
                       "final_frames": row["final_frames"]})
    return result


def reviewed_qa(files: QAFiles) -> dict[str, Any]:
    """Recompute real V2 QA and require a separate exact-hash review receipt."""
    for path in (files.first_a, files.first_b, files.attestation, files.report,
                 files.independent_review):
        sha(path)
    if files.third is not None:
        sha(files.third)
    observed = read(files.report)
    recomputed = human_qa_v2.audit_reviews(files.first_a, files.first_b,
                                            files.attestation, files.third)
    need(observed == recomputed and observed.get("status") == QA_PASS
         and observed.get("case_census") == 18
         and observed.get("all_18_initial_pairs_pass") is True
         and observed.get("critical_disagreement_count") == 0
         and observed.get("third_review_received") is False
         and all(row["initial_pair_pass"] is True for row in observed["cases"]),
         "real 18/18 two-listener QA has not passed")
    hashes = {"first_a": sha(files.first_a), "first_b": sha(files.first_b),
              "attestation": sha(files.attestation), "report": sha(files.report),
              "third": sha(files.third) if files.third is not None else None,
              "source_manifest": sha(SOURCE / "manifest.json"),
              "bundle_mapping": sha(human_qa_v2.BUNDLE / "TECHNICAL_MAPPING.json")}
    review = read(files.independent_review)
    need(review == {"schema": REVIEW_SCHEMA, "status": "PASS_REAL_HUMAN_QA",
                    "sha256": hashes, "reviewer_distinct_from_first_pair": True,
                    "original_exports_inspected": True},
         "independent QA review receipt missing or mismatched")
    return {"schema": "qwen_action_audio_qa_admission_candidate_v1",
            "status": "human_qa_reviewed_source_only_no_model_go",
            "sha256": hashes, "development_case_ids":
                [row["case_id"] for row in source_plan()],
            "candidate_model_inference_authorized": False}


def _array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def validate_audio_batch(batch: Any, processor: Any, *, wav_sha256: str,
                         case_id: str) -> dict[str, Any]:
    """Bind the actual processor tensors to audio placeholders, before weights."""
    need(set(batch.keys()) == EXPECTED_BATCH_KEYS,
         "unexpected audio processor batch fields")
    ids, attention, features, mask = (_array(batch[name]) for name in
                                      ("input_ids", "attention_mask",
                                       "input_features", "feature_attention_mask"))
    need(ids.ndim == attention.ndim == 2 and ids.shape == attention.shape
         and ids.shape[0] == 1 and ids.shape[1] > 0
         and np.issubdtype(ids.dtype, np.integer)
         and np.all(ids >= 0) and np.all(attention == 1),
         "prompt IDs or attention mask changed")
    need(features.ndim == 3 and features.shape[0:2] == (1, 128)
         and features.shape[2] > 0 and features.dtype == np.float32
         and np.isfinite(features).all(), "audio features invalid")
    need(mask.ndim == 2 and mask.shape == (1, features.shape[2])
         and np.all((mask == 0) | (mask == 1))
         and np.all(mask[0, :int(mask.sum())] == 1)
         and np.all(mask[0, int(mask.sum()):] == 0),
         "feature mask invalid or non-contiguous")
    positions = np.flatnonzero(ids[0] == int(processor.audio_token_id)).tolist()
    need(positions and positions == list(range(positions[0], positions[-1] + 1)),
         "missing or split audio placeholders")
    extractor = processor.feature_extractor
    spec = load_spec()
    validate_local_source(spec)
    config = read(ROOT / "models/qwen_omni/config.snapshot.json")
    n_window = config["thinker_config"]["audio_config"]["n_window"]
    need(extractor.sampling_rate == 16000 and extractor.hop_length == 160
         and n_window == 100, "pinned audio time geometry changed")
    row_map = build_row_index(mel_frames=int(mask.sum()),
                              audio_token_positions=positions,
                              n_window=n_window,
                              hop_length=extractor.hop_length,
                              sample_rate=extractor.sampling_rate)
    return {"schema": "qwen_action_audio_processor_record_v1",
            "case_id": case_id, "wav_sha256": wav_sha256,
            "prompt_ids": ids[0].astype(int).tolist(),
            "prompt_last_position": int(ids.shape[1] - 1),
            "batch_shapes": {name: list(_array(batch[name]).shape)
                             for name in sorted(EXPECTED_BATCH_KEYS)},
            "tensor_sha256": {"input_ids": hashlib.sha256(ids.tobytes()).hexdigest(),
                              "attention_mask": hashlib.sha256(attention.tobytes()).hexdigest(),
                              "input_features": hashlib.sha256(features.tobytes()).hexdigest(),
                              "feature_attention_mask": hashlib.sha256(mask.tobytes()).hexdigest()},
            "audio_placeholder_positions": positions,
            "encoder_row_count": row_map["encoder_row_count"],
            "projected_row_count": row_map["projected_row_count"],
            "mel_frames": row_map["mel_frames"],
            "timing_warning": row_map["timing_warning"]}


def prepare_development_case(case_id: str, files: QAFiles,
                             snapshot: Path | None = None) -> tuple[Any, dict[str, Any]]:
    """Load only the pinned processor and prepare one QA-cleared D_* batch."""
    admission = reviewed_qa(files)
    need(snapshot is not None, "pinned processor snapshot required")
    processor_hashes = verify_processor_snapshot(snapshot)
    from transformers import Qwen2_5OmniProcessor
    processor = Qwen2_5OmniProcessor.from_pretrained(str(snapshot),
                                                      local_files_only=True)
    need(type(processor).__name__ == "Qwen2_5OmniProcessor"
         and int(processor.tokenizer.eos_token_id) == 151645,
         "pinned Qwen processor/tokenizer required")
    cases = {row["case_id"]: row for row in source_plan()}
    need(case_id in cases, "only six development audio cases are accepted")
    row = cases[case_id]
    batch, digest = prepare_audio_batch(
        processor, row["wav_path"], system_prompt=load_spec()["system_prompt"])
    need(digest == row["wav_sha256"], "audio changed during processor call")
    record = validate_audio_batch(batch, processor, wav_sha256=digest,
                                  case_id=case_id)
    record["qa_report_sha256"] = admission["sha256"]["report"]
    record["independent_review_sha256"] = sha(files.independent_review)
    record["processor_file_sha256"] = processor_hashes
    record["candidate_model_inference_authorized"] = False
    return batch, record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-plan", action="store_true")
    parser.add_argument("--prepare-development")
    parser.add_argument("--review-a", type=Path)
    parser.add_argument("--review-b", type=Path)
    parser.add_argument("--attestation", type=Path)
    parser.add_argument("--qa-report", type=Path)
    parser.add_argument("--independent-review", type=Path)
    parser.add_argument("--snapshot", type=Path)
    args = parser.parse_args()
    need(args.source_plan != bool(args.prepare_development),
         "choose exactly one source-only action")
    if args.source_plan:
        need(not any((args.review_a, args.review_b, args.attestation,
                      args.qa_report, args.independent_review, args.snapshot)),
             "source plan takes no human paths")
        print(json.dumps({"status": "source_only_human_qa_pending",
                          "development_cases": source_plan()}, sort_keys=True))
        return
    need(all((args.review_a, args.review_b, args.attestation,
              args.qa_report, args.independent_review)),
         "real QA exports, attestation, report and independent review required")
    files = QAFiles(args.review_a, args.review_b, args.attestation,
                    args.qa_report, args.independent_review)
    snapshot = args.snapshot or (Path.home() /
                ".cache/huggingface/hub/models--Qwen--Qwen2.5-Omni-3B"
                / "snapshots" / load_spec()["model_revision"])
    _, record = prepare_development_case(args.prepare_development, files,
                                         snapshot)
    print(json.dumps(record, sort_keys=True))


if __name__ == "__main__":
    main()
