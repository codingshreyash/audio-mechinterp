"""Create an unsigned, sample-only VoiceChat V4 host freeze without model inference.

An independent reviewer must rehash the live host and sign a separate review
before the sample runner may load the checkpoint. This helper never holds keys.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from models.second_architecture.voicechat_calendar_fixture_v1 import public_sample_smoke_v1 as sample


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nemo-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--nano-hf-home", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-review", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    args = parser.parse_args()
    for value in (args.nemo_dir, args.checkpoint_dir, args.nano_hf_home,
                  args.output_dir, args.source_review, args.freeze):
        if not value.is_absolute() or value != value.resolve():
            raise ValueError(f"path must be absolute and resolved: {value}")
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise ValueError("sample output path already consumed")
    if args.freeze.exists() or args.freeze.is_symlink():
        raise ValueError("freeze path already consumed")
    review_sha, _ = sample.hash_file(args.source_review, cap=sample.MAX_EXECUTABLE_FREEZE_BYTES)
    review = json.loads(args.source_review.read_text())
    expected = sample.expected_freeze_fields(
        args.nemo_dir, args.checkpoint_dir, args.nano_hf_home,
        args.output_dir, "a100-80gb")
    if (review.get("schema") != "voicechat_public_sample_source_review_v1"
            or review.get("decision") != "GO"
            or review.get("source_sha256") != expected["source_sha256"]):
        raise ValueError("source review does not bind exact V4 runner sources")
    freeze = {
        "schema": "voicechat_public_sample_smoke_freeze_v1",
        "status": "externally_frozen_sample_only",
        "run_id": args.output_dir.name,
        "expected": expected,
        "caps": {"wall_seconds": 1800, "vram_mib": 75_000,
                 "total_evidence_bytes": 1 << 30},
        "source_review": {"path": str(args.source_review), "sha256": review_sha},
    }
    raw = sample.canonical(freeze)
    if len(raw) > sample.MAX_EXECUTABLE_FREEZE_BYTES:
        raise ValueError(f"freeze exceeds reviewed cap: {len(raw)} bytes")
    args.freeze.parent.mkdir(parents=True, exist_ok=True)
    with args.freeze.open("xb") as stream:
        stream.write(raw)
    print(json.dumps({"freeze_path": str(args.freeze), "freeze_bytes": len(raw),
                      "freeze_sha256": hashlib.sha256(raw).hexdigest(),
                      "source_review_sha256": review_sha,
                      "native_manifest_sha256": expected["native_ffmpeg_runtime"]["source"]["manifest_sha256"],
                      "package_archives_count": expected["native_ffmpeg_runtime"]["source"]["package_archives_count"],
                      "lib_tree_entries": expected["native_ffmpeg_runtime"]["recursive_lib_tree"]["entry_count"]},
                     sort_keys=True))


if __name__ == "__main__":
    main()
