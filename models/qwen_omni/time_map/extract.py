"""Capture pinned Qwen2.5-Omni audio-encoder rows from frozen fit WAVs.

Save one NumPy matrix per selected layer and per clip. Row metadata names the
exact chunk, packed encoder row, pooled pair, and audio-placeholder position.
No model attention or generated transcript is used as a word aligner.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
from pathlib import Path

import numpy as np


THIS_DIR = Path(__file__).resolve().parent
MODEL_DIR = THIS_DIR.parent
PROJECT_DIR = MODEL_DIR.parent.parent
os.environ.setdefault("HF_HOME", str(MODEL_DIR / ".hf_cache"))
sys.path.insert(0, str(MODEL_DIR))

from adapter import (  # noqa: E402
    MODEL_ID,
    MODEL_REVISION,
    TRANSFORMERS_REVISION,
    ActivationCapture,
    load_pinned_thinker,
    prepare_audio_batch,
    resolve_hook_points,
)
from index import build_row_index  # noqa: E402


LAYERS = (0, 7, 15, 23, 31)
SYSTEM_PROMPT = "Answer the spoken request briefly."


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assert_pinned_transformers() -> str:
    direct_url = importlib.metadata.distribution("transformers").read_text("direct_url.json")
    if not direct_url:
        raise RuntimeError("Transformers install has no source commit metadata")
    commit = json.loads(direct_url)["vcs_info"]["commit_id"]
    if commit != TRANSFORMERS_REVISION:
        raise RuntimeError(f"Transformers commit mismatch: {commit}")
    return commit


def read_fit_records(manifest_path: Path, *, limit: int | None, engines: tuple[str, ...]) -> tuple[dict, list[tuple[dict, str, Path]]]:
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("selection", {}).get("split") != "fit":
        raise ValueError("Extraction accepts a fit-only render manifest")
    rows = manifest["records"]
    if any(row["source_item"].get("split") != "fit" for row in rows):
        raise ValueError("A non-fit source item is present")
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be positive")
        rows = rows[:limit]
    selected = []
    for row in rows:
        for engine in engines:
            info = row["outputs"][engine]["processed"]
            path = (manifest_path.parent / info["path"]).resolve()
            if not path.is_relative_to(manifest_path.parent):
                raise ValueError("Audio path escapes render directory")
            if sha256_file(path) != info["file_sha256"]:
                raise ValueError(f"Rendered WAV hash mismatch: {path}")
            if info["sample_rate"] != 16000 or info["channels"] != 1:
                raise ValueError("Expected 16 kHz mono processed WAV")
            selected.append((row, engine, path))
    return manifest, selected


def save_matrix(path: Path, value, *, expected_rows: int, expected_width: int) -> dict:
    if len(value.shape) != 2 or tuple(value.shape) != (expected_rows, expected_width):
        raise ValueError(f"Activation shape mismatch for {path.name}: {tuple(value.shape)}")
    saved = value.float().numpy()
    if not np.isfinite(saved).all():
        raise ValueError(f"Nonfinite activations in {path.name}")
    np.save(path, saved, allow_pickle=False)
    return {
        "path": path.name,
        "sha256": sha256_file(path),
        "source_dtype": str(value.dtype),
        "saved_dtype": str(saved.dtype),
        "shape": list(saved.shape),
        "row_access": "numpy.load(path, mmap_mode='r')[row_index]",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("render_manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("mps", "cpu"), default="cpu")
    parser.add_argument("--engines", nargs="+", choices=("piper", "kokoro"), default=("piper", "kokoro"))
    parser.add_argument("--limit", type=int, help="First N fit records for a bounded smoke")
    args = parser.parse_args()

    import torch
    import transformers

    installed_commit = assert_pinned_transformers()
    pins = json.loads((MODEL_DIR / "pins.json").read_text())
    if sha256_file(MODEL_DIR / "config.snapshot.json") != pins["config_snapshot_sha256"]:
        raise ValueError("Pinned model config snapshot hash mismatch")
    manifest_path = args.render_manifest.resolve()
    render_manifest, selected = read_fit_records(
        manifest_path, limit=args.limit, engines=tuple(dict.fromkeys(args.engines))
    )
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    dtype = torch.bfloat16 if args.device == "mps" else torch.float32
    model, processor = load_pinned_thinker(
        allow_download=False, dtype=dtype, device_map=args.device, low_cpu_mem_usage=True
    )
    model.eval()
    if any(parameter.device.type != args.device for parameter in model.parameters()):
        raise RuntimeError("Model weights are not all on requested device")
    points = resolve_hook_points(model, audio_layers=LAYERS, text_layers=())
    audio_config = model.config.audio_config
    extractor = processor.feature_extractor
    result = {
        "evidence": "source-pinned, observational audio-tensor extraction; rows are not word times",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_pins_sha256": sha256_file(MODEL_DIR / "pins.json"),
        "transformers_source_commit": installed_commit,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "device": args.device,
        "dtype": str(dtype),
        "selected_audio_layers": list(LAYERS),
        "connector_hook": "thinker.model.audio_tower.proj",
        "system_prompt": SYSTEM_PROMPT,
        "render_manifest_path": str(manifest_path),
        "render_manifest_sha256": sha256_file(manifest_path),
        "render_source_manifest_sha256": render_manifest["input_manifest_sha256"],
        "selection": {"engines": list(dict.fromkeys(args.engines)), "first_n_fit_records": args.limit, "selected_clip_count": len(selected)},
        "extractor_script_sha256": sha256_file(Path(__file__)),
        "index_script_sha256": sha256_file(THIS_DIR / "index.py"),
        "processor": {
            "class": type(processor).__name__,
            "feature_extractor_class": type(extractor).__name__,
            "sample_rate": extractor.sampling_rate,
            "hop_length": extractor.hop_length,
            "n_fft": extractor.n_fft,
            "n_window": audio_config.n_window,
            "audio_token_id": processor.audio_token_id,
        },
        "timing_warning": "Rows are packed tensor positions, not word timestamps. Post-attention rows may depend on their entire 200-mel-frame chunk. Nominal times are based only on 160-sample mel hops; use an independent aligner before lexical timing claims.",
        "records": [],
    }
    for index, (row, engine, audio_path) in enumerate(selected):
        batch, audio_sha = prepare_audio_batch(processor, audio_path, system_prompt=SYSTEM_PROMPT)
        if audio_sha != row["outputs"][engine]["processed"]["file_sha256"]:
            raise ValueError("Processor audio bytes differ from render manifest")
        if set(batch.keys()) != {"input_ids", "attention_mask", "input_features", "feature_attention_mask"}:
            raise ValueError(f"Unexpected processor batch fields: {list(batch.keys())}")
        mel_frames = int(batch["feature_attention_mask"].sum())
        placeholder_positions = (batch["input_ids"][0] == processor.audio_token_id).nonzero().flatten().tolist()
        row_map = build_row_index(
            mel_frames=mel_frames,
            audio_token_positions=placeholder_positions,
            n_window=audio_config.n_window,
            hop_length=extractor.hop_length,
            sample_rate=extractor.sampling_rate,
        )
        mel_hash = hashlib.sha256(batch["input_features"].contiguous().numpy().tobytes()).hexdigest()
        moved = batch.to(model.device, dtype=model.dtype)
        with torch.no_grad(), ActivationCapture(points, detach=True) as capture:
            logits = model(**moved, use_cache=False, return_dict=True).logits
            if args.device == "mps":
                torch.mps.synchronize()
        if any(len(values) != 1 for values in capture.activations.values()):
            raise ValueError("Each hooked module must run once per clip")
        stem = f"{index:02d}_{hashlib.sha256((row['item_id'] + engine + audio_sha).encode()).hexdigest()[:16]}"
        clip_dir = output_dir / stem
        clip_dir.mkdir()
        matrices = {}
        for name, values in capture.activations.items():
            expected_rows = row_map["projected_row_count"] if name == "audio.to_language" else row_map["encoder_row_count"]
            expected_width = audio_config.output_dim if name == "audio.to_language" else audio_config.d_model
            path = clip_dir / f"{name}.npy"
            matrices[name] = save_matrix(path, values[0], expected_rows=expected_rows, expected_width=expected_width)
        row_map_path = clip_dir / "row_map.json"
        row_map_path.write_text(json.dumps(row_map, indent=2) + "\n")
        clip_metadata = {
            "item_id": row["item_id"],
            "engine": engine,
            "source_line_sha256": row["source_line_sha256"],
            "audio_path": str(audio_path),
            "audio_sha256": audio_sha,
            "input_ids": batch["input_ids"][0].tolist(),
            "processor_batch_shapes": {key: list(value.shape) for key, value in batch.items()},
            "processor_mel_tensor_sha256": mel_hash,
            "audio_placeholder_positions": placeholder_positions,
            "row_map_path": row_map_path.name,
            "row_map_sha256": sha256_file(row_map_path),
            "matrices": matrices,
            "logits_shape": list(logits.shape),
            "first_reply_token_id": int(logits[0, -1].argmax()),
        }
        meta_path = clip_dir / "metadata.json"
        meta_path.write_text(json.dumps(clip_metadata, indent=2) + "\n")
        result["records"].append({
            "item_id": row["item_id"], "engine": engine,
            "metadata_path": str(meta_path.relative_to(output_dir)),
            "metadata_sha256": sha256_file(meta_path),
            "row_map_path": str(row_map_path.relative_to(output_dir)),
            "row_map_sha256": sha256_file(row_map_path),
        })
        print(f"captured {index + 1}/{len(selected)} {row['item_id']} {engine}: {row_map['encoder_row_count']} encoder rows, {row_map['projected_row_count']} projected rows", flush=True)
    result_path = output_dir / "extraction_manifest.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {result_path}")


if __name__ == "__main__":
    main()
