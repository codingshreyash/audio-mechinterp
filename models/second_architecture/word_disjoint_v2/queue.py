"""Build a blinded, opaque-ID listening bundle for the 32 audited WAVs.

The public bundle contains only anonymous WAV names and queue order. The
private case map retains reference text and must not be shown to listeners.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DEFAULT_OUTPUT = HERE / "rendered_v2"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build(manifest: dict, binding: dict, manifest_sha: str,
          binding_sha: str) -> tuple[dict, dict]:
    if (binding.get("status") != "cpu_wav_technical_audit_passed_listening_pending"
            or binding.get("manifest_sha256") != manifest_sha
            or binding.get("matched_pairs") != 16
            or binding.get("wavs") != 32
            or binding.get("all_outside_word_pcm_equal") is not True
            or binding.get("candidate_model_inference_authorized") is not False
            or manifest.get("matched_pairs") != 16 or manifest.get("wavs") != 32):
        raise ValueError("audited 32-WAV binding absent or changed")
    cases = []
    ids = set()
    for row in manifest["pairs"]:
        for word in row["words"]:
            wav = row["outputs"][word]
            ident = hashlib.sha256(("word-disjoint-listening-v2|" + manifest_sha +
                                    "|" + wav["file_sha256"]).encode()).hexdigest()[:20]
            if ident in ids:
                raise ValueError("opaque ID collision or duplicate WAV hash")
            ids.add(ident)
            cases.append({"opaque_id": ident, "pair_id": row["pair_id"],
                          "engine": row["engine"], "word": word,
                          "reference_script": wav["spoken_script"],
                          "source_path": wav["path"],
                          "wav_sha256": wav["file_sha256"],
                          "pcm_sha256": wav["pcm_int16_sha256"],
                          "frames": wav["frames"]})
    if len(cases) != 32:
        raise ValueError("incomplete listening population")
    by_id = {case["opaque_id"]: case for case in cases}
    remaining = sorted(by_id, key=lambda ident: hashlib.sha256(
        ("word-disjoint-shuffle-v2|" + manifest_sha + "|" + ident).encode()).hexdigest())
    queue = []
    while remaining:
        previous = (by_id[queue[-1]]["pair_id"], by_id[queue[-1]]["engine"]) if queue else None
        chosen = next((i for i, ident in enumerate(remaining)
                       if (by_id[ident]["pair_id"], by_id[ident]["engine"]) != previous), None)
        if chosen is None:
            raise ValueError("cannot separate paired clips in listening queue")
        queue.append(remaining.pop(chosen))
    if len(queue) != 32:
        raise ValueError("incomplete listening order")
    provenance = {"manifest_sha256": manifest_sha,
                  "technical_binding_sha256": binding_sha,
                  "auditor_sha256": binding["auditor_sha256"],
                  "queue_builder_sha256": sha(Path(__file__))}
    private = {"status": "private_references_no_listening_results",
               "scope": "Do not share with blinded reviewers",
               "provenance": provenance, "cases": cases}
    public = {"status": "blinded_listening_queue_no_results",
              "scope": "Listen to every opaque WAV; do not view private references",
              "provenance": provenance, "planned_clips": 32,
              "opaque_ids_in_order": queue,
              "review_form_fields": ["opaque_id", "reviewer_id", "verbatim_transcript",
                                     "middle_word", "clear_or_unclear", "confidence"]}
    return private, public


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--write", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if not output.is_relative_to(ROOT) or not output.is_dir():
        raise ValueError("render directory missing or outside repository")
    manifest_path = output / "manifest.json"
    binding_path = output / "TECHNICAL_BINDING.json"
    manifest = json.loads(manifest_path.read_text())
    binding = json.loads(binding_path.read_text())
    from . import audit
    audit.validate_provenance(manifest)
    audit.verify_saved_report(manifest, binding, output, sha(manifest_path))
    private, public = build(manifest, binding, sha(manifest_path), sha(binding_path))
    if args.validate_only:
        print(json.dumps({"status": "validated_blinded_queue_no_audio_copied",
                          "clips": 32, "manifest_sha256": sha(manifest_path)}, indent=2))
        return
    private_path = output / "QA_PRIVATE_CASES.json"
    public_dir = output / "listening"
    if private_path.exists() or public_dir.exists():
        raise FileExistsError("refusing to replace listening bundle")
    public_dir.mkdir()
    for case in private["cases"]:
        source = (ROOT / case["source_path"]).resolve()
        if not source.is_relative_to(output) or sha(source) != case["wav_sha256"]:
            raise ValueError("source WAV missing or changed")
        destination = public_dir / (case["opaque_id"] + ".wav")
        with destination.open("xb") as stream:
            with source.open("rb") as original:
                shutil.copyfileobj(original, stream)
        if sha(destination) != case["wav_sha256"]:
            raise ValueError("anonymous listening WAV changed")
    with private_path.open("x") as stream:
        json.dump(private, stream, indent=2)
        stream.write("\n")
    queue_path = public_dir / "QUEUE.json"
    with queue_path.open("x") as stream:
        json.dump(public, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"status": "blinded_audio_bundle_ready_no_listening_results",
                      "queue": str(queue_path), "private_case_map": str(private_path),
                      "clips": 32}, indent=2))


if __name__ == "__main__":
    main()
