"""Independent, read-only CPU ASR screen of every frozen corpus_v1 clip.

Run --preflight first. The protocol and JSON schema are frozen alongside this
script. ASR is a proxy for intelligibility, not a human listening judgment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
MANIFEST = HERE / "rendered" / "manifest.json"
OUTPUT = HERE / "asr_audit.json"
EXPECTED_MANIFEST_SHA256 = "adf8cd02d24a43ef576bcb993dc54bd2527440a62798e12fc788def3c063971f"
MODEL_ID = "openai/whisper-tiny.en"
MODEL_REVISION = "87c7102498dcde7456f24cfd30239ca606ed9063"
MAX_CLIPS = 384


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def words(text: str) -> list[str]:
    return re.findall(r"[^\W_]+(?:['’][^\W_]+)?", text.casefold())


def edit_distance(ref: list[str], hyp: list[str]) -> int:
    prior = list(range(len(hyp) + 1))
    for i, a in enumerate(ref, 1):
        row = [i]
        for j, b in enumerate(hyp, 1):
            row.append(min(row[-1] + 1, prior[j] + 1, prior[j - 1] + (a != b)))
        prior = row
    return prior[-1]


def load_items() -> list[dict]:
    if sha256_file(MANIFEST) != EXPECTED_MANIFEST_SHA256:
        raise ValueError("Frozen manifest SHA-256 mismatch")
    manifest = json.loads(MANIFEST.read_text())
    if len(manifest["pairs"]) != 192 or manifest["expected"]["wav_clips"] != MAX_CLIPS:
        raise ValueError("Frozen corpus count mismatch")
    items = []
    seen_paths = set()
    for pair in manifest["pairs"]:
        for word in pair["words"]:
            info = pair["outputs"][word]
            path = (ROOT / info["path"]).resolve()
            if not path.is_relative_to((HERE / "rendered" / "clips").resolve()):
                raise ValueError(f"Clip path escaped frozen corpus: {path}")
            if str(path) in seen_paths or not path.is_file():
                raise ValueError(f"Duplicate or missing clip: {path}")
            seen_paths.add(str(path))
            if sha256_file(path) != info["file_sha256"]:
                raise ValueError(f"Frozen WAV SHA-256 mismatch: {path}")
            if info["duration_seconds"] > 30:
                raise ValueError(f"Clip exceeds one Whisper window: {path}")
            items.append({
                "pair_id": pair["pair_id"], "split": pair["split"],
                "map_fit": pair["map_fit"], "engine": pair["engine"],
                "voice": pair["voice"], "phrasing_id": pair["phrasing_id"],
                "word": word, "other_word": next(w for w in pair["words"] if w != word),
                "reference": info["spoken_script"], "audio_path": info["path"],
                "audio_sha256": info["file_sha256"],
                "pcm_int16_sha256": info["pcm_int16_sha256"],
                "frames": info["frames"],
            })
    if len(items) != MAX_CLIPS:
        raise ValueError(f"Expected {MAX_CLIPS} clips; got {len(items)}")
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true", help="Validate all frozen WAV hashes; do not load ASR")
    args = parser.parse_args()
    items = load_items()
    if args.preflight:
        print(f"preflight OK: {len(items)} clips; manifest {EXPECTED_MANIFEST_SHA256}")
        return
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")

    import numpy as np
    import soundfile as sf
    import torch
    import transformers
    from huggingface_hub import hf_hub_download
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    torch.set_num_threads(4)
    model_path = Path(hf_hub_download(MODEL_ID, "model.safetensors", revision=MODEL_REVISION, local_files_only=True))
    model_sha = sha256_file(model_path)
    processor = WhisperProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION, local_files_only=True)
    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, local_files_only=True, use_safetensors=True
    ).to("cpu").eval()
    if any(p.device.type != "cpu" for p in model.parameters()):
        raise RuntimeError("ASR model is not entirely on CPU")
    records = []
    aggregate = defaultdict(Counter)
    for idx, item in enumerate(items, 1):
        path = ROOT / item["audio_path"]
        samples, rate = sf.read(path, dtype="int16", always_2d=True)
        if rate != 16000 or samples.shape != (item["frames"], 1):
            raise ValueError(f"Frozen WAV shape/rate mismatch: {path}")
        if hashlib.sha256(samples[:, 0].tobytes()).hexdigest() != item["pcm_int16_sha256"]:
            raise ValueError(f"Frozen PCM SHA-256 mismatch: {path}")
        audio = samples[:, 0].astype(np.float32) / 32768.0
        with torch.inference_mode():
            batch = processor(audio, sampling_rate=rate, return_tensors="pt", return_attention_mask=True)
            ids = model.generate(**batch, do_sample=False, num_beams=1, max_new_tokens=128)
        transcript = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
        reference_words, transcript_words = words(item["reference"]), words(transcript)
        word_errors = edit_distance(reference_words, transcript_words)
        exact_normalized = reference_words == transcript_words
        exact_raw = item["reference"].strip() == transcript
        target_count = transcript_words.count(item["word"].casefold())
        other_count = transcript_words.count(item["other_word"].casefold())
        record = {
            **item, "transcript": transcript, "reference_words": reference_words,
            "transcript_words": transcript_words, "raw_exact": exact_raw,
            "normalized_exact": exact_normalized, "word_errors": word_errors,
            "reference_word_count": len(reference_words),
            "wer": word_errors / len(reference_words),
            "target_word_count": target_count, "other_word_count": other_count,
            "target_word_exactly_once": target_count == 1,
        }
        records.append(record)
        for key in ("overall", item["split"], f"{item['split']}:{item['engine']}"):
            agg = aggregate[key]
            agg["clips"] += 1
            agg["raw_exact"] += int(exact_raw)
            agg["normalized_exact"] += int(exact_normalized)
            agg["target_word_exactly_once"] += int(target_count == 1)
            agg["other_word_present"] += int(other_count > 0)
            agg["word_errors"] += word_errors
            agg["reference_words"] += len(reference_words)
        print(f"{idx}/{len(items)} {item['split']} {item['engine']} {item['pair_id']} {item['word']}: {transcript!r}", flush=True)
    summary = {key: {**dict(c), "wer": c["word_errors"] / c["reference_words"]} for key, c in sorted(aggregate.items())}
    result = {
        "schema_version": "corpus_v1_asr_audit_v1",
        "evidence": "single independent ASR checkpoint quality screen; not human perceptual proof; no clip exclusion",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "corpus_manifest": str(MANIFEST.relative_to(ROOT)),
        "corpus_manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "audit_script_sha256": sha256_file(Path(__file__)),
        "asr_model_id": MODEL_ID, "asr_model_revision": MODEL_REVISION,
        "asr_model_safetensors_sha256": model_sha,
        "device": "cpu", "torch_threads": 4,
        "torch_version": torch.__version__, "transformers_version": transformers.__version__,
        "generation": {"do_sample": False, "num_beams": 1, "max_new_tokens": 128, "attention_mask": True},
        "normalization": "casefold; Unicode letter/number words with internal apostrophes; punctuation ignored",
        "records": records, "summary": summary,
    }
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {OUTPUT}; sha256={sha256_file(OUTPUT)}")


if __name__ == "__main__":
    main()
