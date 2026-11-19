"""Write an unsigned H200 public-sample freeze; never import or run VoiceChat.

An independent reviewer must verify the prepared host and sign this exact
freeze before the reviewed sample launcher can start the official example.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from models.second_architecture.voicechat_calendar_fixture_v1 import public_sample_smoke_v1 as sample


SOURCE_GO_SHA256 = "5429bda5913edc035e109b71d9d7c67d1a4ee61a48ef04a67dec4fce8edb5080"
GPU_POLICY = "h200-141gb"


def build(args: argparse.Namespace) -> dict[str, object]:
    paths = (args.nemo_dir, args.checkpoint_dir, args.nano_hf_home,
             args.output_dir, args.source_review, args.freeze)
    if any(not path.is_absolute() or path != path.resolve() for path in paths):
        raise ValueError("all paths must be absolute and resolved")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{3,63}", args.output_dir.name):
        raise ValueError("sample run ID invalid")
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise ValueError("sample output path already consumed")
    if args.freeze.exists() or args.freeze.is_symlink():
        raise ValueError("freeze path already consumed")
    if args.output_dir == args.freeze.parent or args.output_dir in args.freeze.parents:
        raise ValueError("freeze must live outside the unused output directory")

    review_sha, _ = sample.hash_file(args.source_review, cap=sample.MAX_EXECUTABLE_FREEZE_BYTES)
    if review_sha != SOURCE_GO_SHA256:
        raise ValueError("not the independently reviewed V5 H200 30 GiB source GO")
    review = json.loads(args.source_review.read_text())
    expected = sample.expected_freeze_fields(
        args.nemo_dir, args.checkpoint_dir, args.nano_hf_home,
        args.output_dir, GPU_POLICY)
    if (set(review) != {"schema", "decision", "source_sha256", "reviewer_id", "reviewed_utc"}
            or review["schema"] != "voicechat_public_sample_source_review_v1"
            or review["decision"] != "GO"
            or review["source_sha256"] != expected["source_sha256"]
            or not isinstance(review["reviewer_id"], str)
            or not review["reviewer_id"].strip()
            or not sample._utc_record(review["reviewed_utc"])):
        raise ValueError("source GO does not bind the current eight-role V5 source")

    # Hash the weight and check actual H200/free disk before creating any file.
    runtime = sample.check_h200_sample_runtime(args.checkpoint_dir)
    if runtime["gpu_policy"] != GPU_POLICY:
        raise ValueError("H200 runtime policy changed")
    freeze = {
        "schema": "voicechat_public_sample_smoke_freeze_v1",
        "status": "externally_frozen_sample_only",
        "run_id": args.output_dir.name,
        "expected": expected,
        "caps": {"wall_seconds": 1_800, "vram_mib": 130_000,
                 "total_evidence_bytes": 1 << 30},
        "source_review": {"path": str(args.source_review), "sha256": review_sha},
    }
    raw = sample.canonical(freeze)
    if len(raw) > sample.MAX_EXECUTABLE_FREEZE_BYTES:
        raise ValueError(f"freeze exceeds reviewed cap: {len(raw)} bytes")
    args.freeze.parent.mkdir(parents=True, exist_ok=True)
    with args.freeze.open("xb") as stream:
        stream.write(raw)
    return {"freeze_path": str(args.freeze), "freeze_bytes": len(raw),
            "freeze_sha256": hashlib.sha256(raw).hexdigest(),
            "source_review_sha256": review_sha, "gpu_policy": GPU_POLICY,
            "disk_free_gib_at_build": runtime["disk_free_gib"],
            "native_manifest_sha256": expected["native_ffmpeg_runtime"]["source"]["manifest_sha256"],
            "package_archives_count": expected["native_ffmpeg_runtime"]["source"]["package_archives_count"],
            "lib_tree_entries": expected["native_ffmpeg_runtime"]["recursive_lib_tree"]["entry_count"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nemo-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--nano-hf-home", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-review", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    print(json.dumps(build(parser.parse_args()), sort_keys=True))


if __name__ == "__main__":
    main()
