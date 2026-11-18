"""Render an unchanged JSONL item list with pinned, distinct CPU TTS engines.

Each input line needs ``item_id`` and ``text``. All other fields are copied into
the output manifest. This program does not choose, translate, or reword items.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import struct
import sys
import wave
from pathlib import Path

import numpy as np
import scipy
import soundfile as sf
from huggingface_hub import hf_hub_download
from scipy.signal import resample_poly


ROOT = Path(__file__).resolve().parent
PINS_FILE = ROOT / "pins.json"
CACHE_DIR = ROOT / ".hf_cache" / "hub"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pinned_file(config: dict, filename: str, expected_sha256: str, *, download: bool) -> Path:
    path = Path(
        hf_hub_download(
            repo_id=config["model_repo"],
            filename=filename,
            revision=config["model_revision"],
            cache_dir=str(CACHE_DIR),
            local_files_only=not download,
        )
    )
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(f"Model file SHA-256 mismatch for {filename}: {actual}")
    return path


def item_identity_and_text(item: dict, line_number: int) -> tuple[str, str]:
    if "audio_input" in item:
        item_id = item.get("id")
        audio_input = item.get("audio_input")
        text = audio_input.get("script") if isinstance(audio_input, dict) else None
        typed_input = item.get("typed_input")
        typed_prompt = typed_input.get("prompt") if isinstance(typed_input, dict) else None
        if not isinstance(typed_prompt, str) or not isinstance(text, str) or typed_prompt.rstrip() != text.rstrip():
            raise ValueError(f"Line {line_number} has different spoken words and typed prompt")
    else:
        item_id, text = item.get("item_id"), item.get("text")
    if not isinstance(item_id, str) or not item_id:
        raise ValueError(f"Line {line_number} needs a nonempty string id/item_id")
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"Line {line_number} needs a nonempty audio script/text")
    return item_id, text


def load_items(path: Path) -> tuple[list[dict], str]:
    raw = path.read_bytes()
    rows: list[dict] = []
    seen: set[str] = set()
    for line_number, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            continue
        item = json.loads(line)
        if not isinstance(item, dict):
            raise ValueError(f"Line {line_number} must be a JSON object")
        item_id, text = item_identity_and_text(item, line_number)
        if item_id in seen:
            raise ValueError(f"Duplicate item_id {item_id!r}")
        seen.add(item_id)
        rows.append({"source": item, "item_id": item_id, "text": text, "line_number": line_number, "source_line_sha256": hashlib.sha256(line).hexdigest()})
    if not rows:
        raise ValueError("Input JSONL contains no items")
    return rows, hashlib.sha256(raw).hexdigest()


def audio_record(path: Path) -> dict:
    samples, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if samples.shape[1] != 1:
        raise ValueError(f"Expected mono audio: {path}")
    if not np.isfinite(samples).all():
        raise ValueError(f"Nonfinite audio: {path}")
    mono = samples[:, 0]
    if mono.size == 0:
        raise ValueError(f"Empty audio: {path}")
    peak = float(np.max(np.abs(mono)))
    rms = float(np.sqrt(np.mean(mono.astype(np.float64) ** 2)))
    if peak < 1e-5 or rms < 1e-6:
        raise ValueError(f"Near-silent audio: {path}")
    info = sf.info(path)
    return {
        "path": str(path),
        "file_sha256": sha256_file(path),
        "pcm_float32_sha256": hashlib.sha256(mono.astype("<f4").tobytes()).hexdigest(),
        "sample_rate": sample_rate,
        "frames": int(mono.size),
        "duration_seconds": float(mono.size / sample_rate),
        "channels": 1,
        "subtype": info.subtype,
        "peak": peak,
        "rms": rms,
    }


def write_float32_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    """Write a minimal IEEE-float WAV without libsndfile's timestamped PEAK chunk."""
    mono = np.asarray(samples, dtype="<f4").reshape(-1)
    if not np.isfinite(mono).all():
        raise ValueError("Kokoro returned nonfinite samples")
    data = mono.tobytes()
    if len(data) > 0xFFFFFFFF - 36:
        raise ValueError("WAV exceeds RIFF 32-bit size")
    header = (
        struct.pack("<4sI4s", b"RIFF", 36 + len(data), b"WAVE")
        + struct.pack("<4sIHHIIHH", b"fmt ", 16, 3, 1, sample_rate, sample_rate * 4, 4, 32)
        + struct.pack("<4sI", b"data", len(data))
    )
    path.write_bytes(header + data)


def to_16k(raw_path: Path, output_path: Path, config: dict) -> dict:
    samples, source_rate = sf.read(raw_path, dtype="float64", always_2d=True)
    if samples.shape[1] != 1:
        raise ValueError(f"Expected mono raw audio: {raw_path}")
    target_rate = config["target_sample_rate"]
    gcd = math.gcd(source_rate, target_rate)
    up, down = target_rate // gcd, source_rate // gcd
    window = tuple(config["resampler_window"])
    rendered = resample_poly(
        samples[:, 0], up, down, window=window,
        padtype=config["resampler_padtype"], cval=config["resampler_cval"],
    )
    clip_count = int(np.count_nonzero((rendered < -1.0) | (rendered >= 1.0)))
    rendered = np.clip(rendered, -1.0, 1.0 - 1 / 32768)
    sf.write(output_path, rendered, target_rate, subtype="PCM_16", format="WAV")
    return {**audio_record(output_path), "resample_up": up, "resample_down": down, "pre_quantization_clip_count": clip_count}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("items", type=Path, help="Frozen JSONL with item_id and exact text")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-download", action="store_true", help="Fetch only the pinned model files")
    parser.add_argument("--split", help="Render only records with this existing split label")
    parser.add_argument("--limit", type=int, help="Render the first N selected records for a bounded smoke run")
    args = parser.parse_args()
    rows, input_hash = load_items(args.items)
    all_item_count = len(rows)
    if args.split is not None:
        rows = [row for row in rows if row["source"].get("split") == args.split]
    if args.limit is not None:
        if args.limit < 1:
            parser.error("--limit must be positive")
        rows = rows[: args.limit]
    if not rows:
        raise ValueError("No records selected for rendering")
    pins = json.loads(PINS_FILE.read_text())
    from piper import PiperVoice
    from piper.config import SynthesisConfig
    from kokoro_onnx import Kokoro
    import kokoro_onnx
    import onnxruntime
    import piper

    piper_config = pins["piper"]
    piper_model = pinned_file(piper_config, piper_config["model_file"], piper_config["model_sha256"], download=args.allow_download)
    piper_json = pinned_file(piper_config, piper_config["config_file"], piper_config["config_sha256"], download=args.allow_download)
    kokoro_config = pins["kokoro"]
    kokoro_model = pinned_file(kokoro_config, kokoro_config["model_file"], kokoro_config["model_sha256"], download=args.allow_download)
    kokoro_voices = pinned_file(kokoro_config, kokoro_config["voices_file"], kokoro_config["voices_sha256"], download=args.allow_download)
    voice = PiperVoice.load(piper_model, config_path=piper_json, use_cuda=False)
    kokoro = Kokoro(str(kokoro_model), str(kokoro_voices))
    if kokoro_config["voice"] not in kokoro.get_voices():
        raise ValueError("Pinned Kokoro voice absent from voices.bin")
    if "CPUExecutionProvider" not in kokoro.sess.get_providers():
        raise RuntimeError("Kokoro ONNX session lacks CPUExecutionProvider")
    if "CPUExecutionProvider" not in voice.session.get_providers():
        raise RuntimeError("Piper ONNX session lacks CPUExecutionProvider")

    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    (output_dir / "raw" / "piper").mkdir(parents=True, exist_ok=True)
    (output_dir / "raw" / "kokoro").mkdir(parents=True, exist_ok=True)
    (output_dir / "processed" / "piper").mkdir(parents=True, exist_ok=True)
    (output_dir / "processed" / "kokoro").mkdir(parents=True, exist_ok=True)

    manifest = {
        "evidence": "two-engine synthetic English audio rendering; no ASR or human intelligibility validation",
        "input_manifest_path": str(args.items.resolve()),
        "input_manifest_sha256": input_hash,
        "input_item_count": all_item_count,
        "selection": {"split": args.split, "limit": args.limit, "selected_count": len(rows)},
        "pins_sha256": sha256_file(PINS_FILE),
        "script_sha256": sha256_file(Path(__file__)),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "versions": {
            "piper_tts": __import__("importlib.metadata", fromlist=["version"]).version("piper-tts"),
            "kokoro_onnx": kokoro_onnx.__version__ if hasattr(kokoro_onnx, "__version__") else __import__("importlib.metadata", fromlist=["version"]).version("kokoro-onnx"),
            "onnxruntime": onnxruntime.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "soundfile": sf.__version__,
        },
        "onnx_providers": {
            "piper": voice.session.get_providers(),
            "kokoro": kokoro.sess.get_providers(),
        },
        "pins": pins,
        "records": [],
    }
    piper_syn = SynthesisConfig(**piper_config["synthesis"])
    for index, row in enumerate(rows):
        item = row["source"]
        text = row["text"]
        stem = f"{index:04d}_{hashlib.sha256((row['item_id'] + chr(0) + text).encode('utf-8')).hexdigest()[:16]}"
        outputs = {}
        for engine in ("piper", "kokoro"):
            raw_path = output_dir / "raw" / engine / f"{stem}.wav"
            processed_path = output_dir / "processed" / engine / f"{stem}.wav"
            if engine == "piper":
                with wave.open(str(raw_path), "wb") as wav_file:
                    voice.synthesize_wav(text, wav_file, syn_config=piper_syn)
            else:
                samples, sample_rate = kokoro.create(text, voice=kokoro_config["voice"], **kokoro_config["synthesis"])
                write_float32_wav(raw_path, samples, sample_rate)
            raw_info = audio_record(raw_path)
            processed_info = to_16k(raw_path, processed_path, pins["preprocessing"])
            outputs[engine] = {
                "raw": {**raw_info, "path": str(raw_path.relative_to(output_dir))},
                "processed": {**processed_info, "path": str(processed_path.relative_to(output_dir))},
                "voice": pins[engine]["voice"],
                "model_revision": pins[engine]["model_revision"],
            }
        manifest["records"].append({
            "source_item": item,
            "item_id": row["item_id"],
            "source_line_number": row["line_number"],
            "source_line_sha256": row["source_line_sha256"],
            "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "typed_audio_bytes_equal": item.get("typed_input", {}).get("prompt", text) == text,
            "outputs": outputs,
        })
        print(f"rendered {index + 1}/{len(rows)}: {row['item_id']}", flush=True)
    output = output_dir / "render_manifest.json"
    output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
