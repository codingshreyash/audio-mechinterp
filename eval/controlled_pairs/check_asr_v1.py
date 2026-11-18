"""Pinned CPU ASR quality screen for the four assembled color questions."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "eval" / "tts"))
from audit_asr import MODEL_ID, MODEL_REVISION, waveform_checks  # noqa: E402


def main() -> None:
    base = ROOT / "eval" / "tts" / "outputs" / "controlled_pair_v1"
    path = base / "controlled_pair_manifest.json"
    manifest = json.loads(path.read_text())
    torch.set_num_threads(4)
    processor = WhisperProcessor.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, local_files_only=True
    )
    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, local_files_only=True
    ).to("cpu").eval()
    result = {
        "evidence": "single pinned ASR-model quality screen, not human listening",
        "source_manifest": str(path.relative_to(ROOT)),
        "asr_model_id": MODEL_ID,
        "asr_model_revision": MODEL_REVISION,
        "transcripts": {},
    }
    for engine, group in manifest["engines"].items():
        result["transcripts"][engine] = {}
        for word, info in group["outputs"].items():
            audio = ROOT / info["path"]
            samples, rate, quality = waveform_checks(audio, info["file_sha256"])
            with torch.inference_mode():
                batch = processor(samples, sampling_rate=rate, return_tensors="pt",
                                  return_attention_mask=True)
                ids = model.generate(**batch, do_sample=False, num_beams=1,
                                     max_new_tokens=128)
            text = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
            result["transcripts"][engine][word] = {
                "audio_sha256": info["file_sha256"],
                "transcript": text,
                "quality_flags": quality["audio_quality_flags"],
            }
            print(engine, word, repr(text), quality["audio_quality_flags"])
    (base / "asr_screen.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
