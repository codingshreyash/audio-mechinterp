"""CPU-only prospective Voxtral masks for the technically bound 480-WAV v2 corpus.

No processor/model/ASR is loaded. The 1500/375 row geometry is nominal until a
separate processor and model-hook shape check passes before any model answer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import wave
from collections import Counter
from pathlib import Path

import voxtral_fit_v2_causal_masks as old


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CORPUS = ROOT / "eval/audio_jlens/corpus_v2"
MANIFEST = CORPUS / "rendered_v2/manifest.json"
BINDING = CORPUS / "rendered_v2/TECHNICAL_BINDING.json"
CANDIDATES = CORPUS / "CANDIDATES_UNFROZEN.json"
VALIDATION = CORPUS / "CANDIDATE_VALIDATION_UNFROZEN.json"
RENDER_FREEZE = CORPUS / "RENDER_FREEZE_V1.json"
DESIGN = HERE / "PROSPECTIVE_VOXTRAL_AUDIO_JACOBIAN_LENS_V1.md"
OLD_MASK_BUILDER = HERE / "voxtral_fit_v2_causal_masks.py"
FREEZE = HERE / "VOXTRAL_V2_JLENS_MASKS_FREEZE_V1.json"
OUTPUT = HERE / "VOXTRAL_V2_JLENS_MASKS_PROSPECTIVE_V1.json"
PIN = {
    "manifest": "15395c7d132eb3f1c1a6b2b7d555146faa83be4ee2b77fba4f481cf70f26761d",
    "binding": "858f8f68fce65ee60e328b1536b34608ced6b3e84d725f229460c6376e14af93",
    "candidates": "ac2f9598e4459191569662aa34803c5a72e57ff686d415e60935a4f716a11a93",
    "validation": "67e3eec5e14b30a61efbbce449547ec2d5083a87d12a84f0aee8d0e715a59d56",
    "render_freeze": "ca90162648f41c56997452a30c9c95791530929c231306ea7529dc2268071410",
    "design": "d103791610184dd1c3b4beac2ac5cb0bb74d9cd03b55ac6bfaac43c755a293fa",
    "old_builder": "ab8bdcf0374794c32b904cbdee37cba48c87296cef2970fbc0af896e0d21c2fc",
}
SPLITS = ("fit_pool", "development", "locked_test")
ENGINES = ("piper", "kokoro")
SITES = ("encoder31", "projector")
CLIP_ROOT = (CORPUS / "rendered_v2/clips").resolve()


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def read_inputs() -> tuple[dict, dict, dict]:
    paths = {"manifest": MANIFEST, "binding": BINDING, "candidates": CANDIDATES,
             "validation": VALIDATION, "render_freeze": RENDER_FREEZE,
             "design": DESIGN, "old_builder": OLD_MASK_BUILDER}
    for key, path in paths.items():
        require(path.is_file() and sha(path) == PIN[key], f"pinned v2 source drift: {key}")
    manifest = json.loads(MANIFEST.read_text())
    binding = json.loads(BINDING.read_text())
    spec = json.loads(CANDIDATES.read_text())
    validation = json.loads(VALIDATION.read_text())
    require(binding["status"] == "cpu_wav_technical_audit_passed_intelligibility_pending_no_model_output"
            and binding["manifest_sha256"] == PIN["manifest"]
            and binding["matched_records"] == 240 and binding["wavs"] == 480
            and binding["model_inference_authorized"] is False,
            "v2 technical WAV binding changed")
    require(manifest["status"] == "cpu_stimuli_only_pending_independent_wav_and_intelligibility_audit"
            and manifest["matched_records"] == 240 and manifest["wavs"] == 480
            and manifest["primary_pairs"] == 60 and manifest["reserve_pairs_used"] == 0,
            "v2 corpus population/status changed")
    require(validation["candidate_pairs"] == 60 and validation["reserve_pairs"] == 12
            and validation["rendered_pair_records_planned"] == 240
            and validation["wavs_planned"] == 480,
            "v2 candidate assignment changed")
    require(set(validation["assigned_pairs"]) == set(SPLITS), "unexpected v2 split")
    return manifest, validation, spec


def planned_rows(validation: dict, spec: dict) -> dict[tuple[str, str, str], dict]:
    planned: dict[tuple[str, str, str], dict] = {}
    counts = Counter()
    roles = Counter()
    pair_words: dict[str, tuple[str, ...]] = {}
    for split in SPLITS:
        templates = spec["templates_by_split"][split]
        require(len(templates) == 2 and len({t["id"] for t in templates}) == 2,
                f"v2 {split} template grammar changed")
        pairs = validation["assigned_pairs"][split]
        require(len(pairs) == {"fit_pool": 36, "development": 12, "locked_test": 12}[split],
                f"v2 {split} pair count changed")
        for pair in pairs:
            pair_id, words = pair["pair_id"], pair["words"]
            require(pair_id not in pair_words and len(words) == 2 and words[0] != words[1],
                    "duplicate v2 pair or word")
            pair_words[pair_id] = tuple(words)
            category = pair["category"]
            require(category in ("animal", "color", "nature", "object") and
                    tuple(words) in [tuple(p) for p in spec["pairs_by_category"][category]],
                    "pair not in primary candidate population")
            counts[(split, category)] += 1
            if split == "fit_pool":
                require(pair["fit_role"] in ("basis", "map"), "FIT role changed")
                roles[pair["fit_role"]] += 1
            for engine in ENGINES:
                for template in templates:
                    key = (pair_id, engine, template["id"])
                    require(key not in planned, "duplicate v2 matched record")
                    planned[key] = {"split": split, "pair": pair,
                                    "engine": engine, "template": template}
    require(len(planned) == 240 and len(pair_words) == 60 and
            roles == {"basis": 16, "map": 20} and
            all(counts[(split, category)] == expected
                for split, expected in (("fit_pool", 9), ("development", 3), ("locked_test", 3))
                for category in ("animal", "color", "nature", "object")),
            "v2 split/category/FIT-role balance changed")
    all_words = [w for words in pair_words.values() for w in words]
    require(len(set(all_words)) == 120, "v2 word reused across pairs")
    return planned


def read_clip(item: dict) -> tuple[bytes, int]:
    relative = item["path"]
    require(isinstance(relative, str) and not Path(relative).is_absolute(),
            "absolute/invalid v2 clip path")
    path = (ROOT / relative).resolve()
    require(path.is_relative_to(CLIP_ROOT) and path.is_file() and
            sha(path) == item["file_sha256"], "v2 clip path/file hash drift")
    with wave.open(str(path), "rb") as wav:
        frames = wav.getnframes()
        require((wav.getnchannels(), wav.getframerate(), wav.getsampwidth(),
                 wav.getcomptype()) == (1, 16000, 2, "NONE") and
                frames == item["frames"], "v2 WAV geometry drift")
        pcm = wav.readframes(frames)
    require(hashlib.sha256(pcm).hexdigest() == item["pcm_int16_sha256"],
            "v2 PCM SHA drift")
    return pcm, frames


def require_pair_pcm(a: bytes, b: bytes, start: int, end: int) -> None:
    require(len(a) == len(b) and 0 <= start < end <= len(a) // 2,
            "invalid v2 paired PCM geometry")
    require(a[:2 * start] == b[:2 * start] and a[2 * end:] == b[2 * end:]
            and a[2 * start:2 * end] != b[2 * start:2 * end],
            "v2 pair differs outside word or not inside word")


def nominal_geometry(start: int, end: int, frames: int) -> tuple[dict, list[int], dict]:
    """Reuse exact old rules; never infer a changed processor shape from WAVs."""
    require(0 <= start < end <= frames, "v2 word slot outside clip")
    require(frames <= 2999 * old.MEL_HOP - old.MEL_HALF_WINDOW,
            "v2 clip exceeds old 3000-mel-frame padded-fill geometry")
    intervals = old.bounds(start, end)
    require(intervals["pre"][0] >= 0 and intervals["post"][1] <= frames,
            "v2 pre/post interval outside clip")
    mel = [i for i in range(3000)
           if i * old.MEL_HOP - old.MEL_HALF_WINDOW < end
           and i * old.MEL_HOP + old.MEL_HALF_WINDOW > start]
    require(bool(mel) and 2999 not in mel, "v2 word-overlap mel frame invalid")
    site_maps = {}
    for site in SITES:
        bins = {name: old.center_rows(lo, hi, site) for name, (lo, hi) in intervals.items()}
        require(len(bins) == 6 and all(bins.values()),
                f"v2 {site} source bin empty/changed")
        combined = [index for rows in bins.values() for index in rows]
        require(len(combined) == len(set(combined)), f"v2 {site} bins overlap")
        whole = sorted(index for q in range(4) for index in bins[f"word_{q}"])
        padding = old.padding_rows(frames, len(whole), site)
        require(len(padding) == len(whole) and not set(padding) & set(combined),
                f"v2 {site} padding support invalid")
        site_maps[site] = {"bins": bins, "whole_word_rows": whole,
                           "padding_control_rows": padding}
    return {name: list(bounds) for name, bounds in intervals.items()}, mel, site_maps


def make_manifest(manifest: dict, validation: dict, spec: dict) -> dict:
    expected = planned_rows(validation, spec)
    require(len(manifest["pairs"]) == 240, "v2 manifest record count changed")
    actual = {(r["pair_id"], r["engine"], r["phrasing_id"]): r
              for r in manifest["pairs"]}
    require(len(actual) == 240 and set(actual) == set(expected),
            "v2 matched record identity changed")
    records = []
    for key in expected:
        row, plan = actual[key], expected[key]
        pair = plan["pair"]
        require(row["split"] == plan["split"] and row["category"] == pair["category"]
                and row["fit_role"] == pair.get("fit_role")
                and row["words"] == pair["words"]
                and row["outside_slot_pcm_identical"] is True
                and row["inside_slot_pcm_differs"] is True
                and set(row["outputs"]) == set(pair["words"]),
                "v2 manifest row identity or paired claims changed")
        start, end = row["word_slot_start_sample"], row["word_slot_end_sample_exclusive"]
        clips, pcms = [], []
        for word in pair["words"]:
            item = row["outputs"][word]
            pcm, frames = read_clip(item)
            clips.append({"word": word, "wav_path": item["path"],
                          "wav_sha256": item["file_sha256"],
                          "pcm_int16_sha256": item["pcm_int16_sha256"],
                          "frames": frames})
            pcms.append(pcm)
        require(clips[0]["frames"] == clips[1]["frames"],
                "v2 matched word clips unequal length")
        require_pair_pcm(pcms[0], pcms[1], start, end)
        intervals, mel, site_maps = nominal_geometry(start, end, clips[0]["frames"])
        records.append({"split": plan["split"], "fit_role": pair.get("fit_role"),
                        "pair_id": pair["pair_id"], "category": pair["category"],
                        "engine": plan["engine"], "phrasing_id": plan["template"]["id"],
                        "words": pair["words"], "clips": clips,
                        "word_slot_start_sample": start,
                        "word_slot_end_sample_exclusive": end,
                        "intervals_half_open_samples": intervals,
                        "word_mel_frames_direct_overlap": mel,
                        "blank_fill_mel_frame": 2999, "site_maps": site_maps})
    return {"schema_version": "voxtral_v2_jlens_masks_prospective_v1",
            "status": "cpu_only_nominal_geometry; no processor/model/intelligibility result",
            "source_sha256": PIN, "builder_sha256": sha(Path(__file__)),
            "site_tensor_shapes_assumed_not_verified": {
                "encoder31": [1, 1500, 1280], "projector": [375, 3072]},
            "row_centers_samples": {"encoder31": "320*j",
                                    "projector": "160*(8*r+3)"},
            "records": records}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--write-manifest", action="store_true")
    args = parser.parse_args()
    try:
        source, validation, spec = read_inputs()
        output = make_manifest(source, validation, spec)
    except Exception as exc:
        print(json.dumps({"status": "failed_prospective_geometry_or_source_check",
                          "error_type": type(exc).__name__, "error": str(exc),
                          "planned_records": 240, "model_output_opened": False}, indent=2))
        raise
    if args.validate_only:
        counts = Counter(r["split"] for r in output["records"])
        print(json.dumps({"status": "validated_cpu_nominal_masks_no_model_output",
                          "records": len(output["records"]),
                          "wavs_rehashed": sum(len(r["clips"]) for r in output["records"]),
                          "split_records": dict(counts),
                          "fit_basis_records": sum(r["fit_role"] == "basis" for r in output["records"]),
                          "fit_map_records": sum(r["fit_role"] == "map" for r in output["records"]),
                          "builder_sha256": sha(Path(__file__)),
                          "processor_shape_verified": False,
                          "intelligibility_verified": False}, indent=2))
        return
    require(FREEZE.is_file(), "independent mask-builder review/freeze absent")
    freeze = json.loads(FREEZE.read_text())
    require(freeze.get("status") == "independently_reviewed_voxtral_v2_masks_frozen"
            and freeze.get("builder_sha256") == sha(Path(__file__))
            and freeze.get("source_sha256") == PIN,
            "mask-builder freeze receipt changed")
    require(not OUTPUT.exists(), "prospective mask manifest exists")
    with OUTPUT.open("x") as stream:
        json.dump(output, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"status": output["status"], "output": str(OUTPUT),
                      "records": len(output["records"]), "sha256": sha(OUTPUT)}, indent=2))


if __name__ == "__main__":
    main()
