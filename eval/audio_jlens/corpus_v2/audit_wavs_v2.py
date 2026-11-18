"""Independent CPU readback audit for the prospective 480-WAV v2 corpus.

No TTS, ASR, or Qwen model is loaded. Passing writes a technical receipt only;
intelligibility and permission to run an answer model remain separate gates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CANDIDATES = HERE / "CANDIDATES_UNFROZEN.json"
VALIDATION = HERE / "CANDIDATE_VALIDATION_UNFROZEN.json"
PROTOCOL = HERE / "RENDER_PROTOCOL_PROSPECTIVE_V1.md"
FREEZE = HERE / "RENDER_FREEZE_V1.json"
RENDERER = HERE / "render_v2.py"
SOURCE_PLAN = ROOT / "eval/heldout_flow/plan_v1.json"
ASSEMBLY = ROOT / "eval/heldout_flow/render_v1.py"
TTS_RENDER = ROOT / "eval/tts/render.py"
SPLITS = ("fit_pool", "development", "locked_test")
VOICES = ("piper", "kokoro")


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def local_path(relative: str, output: Path, subdir: str) -> Path:
    require(isinstance(relative, str) and relative and not Path(relative).is_absolute(),
            "invalid absolute or empty corpus path")
    path = (ROOT / relative).resolve()
    require(path.is_relative_to(output / subdir) and path.is_file(),
            f"missing or outside-output corpus file: {relative}")
    return path


def expected_population(spec: dict, validation: dict) -> dict[tuple[str, str, str], dict]:
    require(validation["candidate_pairs"] == 60 and validation["reserve_pairs"] == 12
            and validation["rendered_pair_records_planned"] == 240
            and validation["wavs_planned"] == 480, "validated selection counts changed")
    primary = {w for pairs in spec["pairs_by_category"].values() for pair in pairs for w in pair}
    reserve = {w for pairs in spec["ordered_reserve_pairs_by_category"].values()
               for pair in pairs for w in pair}
    require(len(primary) == 120 and len(reserve) == 24 and not primary & reserve,
            "primary or inactive-reserve word count changed")
    out = {}
    seen_pairs = set()
    for split in SPLITS:
        for pair in validation["assigned_pairs"][split]:
            pair_id = pair["pair_id"]
            require(pair_id not in seen_pairs and pair["category"] in spec["pairs_by_category"]
                    and tuple(pair["words"]) in [tuple(p) for p in spec["pairs_by_category"][pair["category"]]],
                    "assigned pair not a unique primary candidate")
            seen_pairs.add(pair_id)
            for engine in VOICES:
                for template in spec["templates_by_split"][split]:
                    key = (pair_id, engine, template["id"])
                    require(key not in out, "duplicate planned record")
                    out[key] = {"split": split, "pair": pair,
                                "engine": engine, "template": template}
    require(len(seen_pairs) == 60 and len(out) == 240, "planned primary population incomplete")
    return out


def audit_fragment(fragment: dict, output: Path, expected_text: str, voice: str) -> None:
    require(fragment["text"] == expected_text and fragment["voice"] == voice
            and fragment["text_sha256"] == hashlib.sha256(expected_text.encode()).hexdigest(),
            "fragment text or voice changed")
    for variant in ("raw", "processed"):
        rec = fragment[variant]
        path = local_path(rec["path"], output, variant)
        require(sha(path) == rec["file_sha256"], "fragment file hash changed")
        info = sf.info(path)
        pcm, rate = sf.read(path, dtype="float32", always_2d=True)
        require(rate == rec["sample_rate"] == info.samplerate
                and info.channels == rec["channels"] == 1
                and info.frames == rec["frames"] == len(pcm)
                and info.subtype == rec["subtype"] and len(pcm) > 0
                and math.isclose(rec["duration_seconds"], len(pcm) / rate,
                                 rel_tol=0, abs_tol=1e-8)
                and np.isfinite(pcm).all(), "fragment WAV geometry/duration invalid")
        mono = pcm[:, 0]
        require(hashlib.sha256(mono.astype("<f4").tobytes()).hexdigest() ==
                rec["pcm_float32_sha256"], "fragment PCM hash changed")
        peak = float(np.max(np.abs(mono)))
        rms = float(np.sqrt(np.mean(mono.astype(np.float64)**2)))
        require(peak >= 1e-5 and rms >= 1e-6
                and math.isclose(peak, rec["peak"], rel_tol=1e-6, abs_tol=1e-7)
                and math.isclose(rms, rec["rms"], rel_tol=1e-6, abs_tol=1e-7),
                "fragment amplitude invalid")
        if variant == "processed":
            require(rate == 16000 and info.subtype == "PCM_16", "processed fragment format changed")


def processed_pcm(fragment: dict, output: Path) -> np.ndarray:
    """Read audited PCM16 bytes for an independent exact assembly check."""
    rec = fragment["processed"]
    path = local_path(rec["path"], output, "processed")
    samples, rate = sf.read(path, dtype="int16", always_2d=True)
    require(rate == 16000 and samples.shape == (rec["frames"], 1),
            "processed fragment PCM geometry changed")
    return samples[:, 0]


def audit_pair(row: dict, planned: dict, fragments: dict, output: Path) -> tuple[set[str], float, int]:
    pair, template, engine = planned["pair"], planned["template"], planned["engine"]
    words = pair["words"]
    require(row["split"] == planned["split"] and row["category"] == pair["category"]
            and row["fit_role"] == pair.get("fit_role")
            and row["engine"] == engine and row["voice"] == fragments[(engine, f"prefix_{template['id']}")]["voice"]
            and row["pair_id"] == pair["pair_id"] and row["phrasing_id"] == template["id"]
            and row["words"] == words and row["sample_rate"] == 16000
            and row["subtype"] == "PCM_16"
            and row["outside_slot_pcm_identical"] is True
            and row["inside_slot_pcm_differs"] is True,
            "manifest pair identity/format/claims changed")
    ids = (f"prefix_{template['id']}",
           *(f"word_{pair['pair_id']}_{word}" for word in words),
           f"suffix_{template['id']}")
    require(set(row["fragment_sha256"]) == set(ids), "row fragment keys changed")
    for id_ in ids:
        src = fragments[(engine, id_)]
        require(row["fragment_sha256"][id_] == {
            "raw": src["raw"]["file_sha256"],
            "processed": src["processed"]["file_sha256"]},
            "row fragment hash changed")
    start, end = row["word_slot_start_sample"], row["word_slot_end_sample_exclusive"]
    prefix_frames = fragments[(engine, f"prefix_{template['id']}")]["processed"]["frames"]
    suffix_frames = fragments[(engine, f"suffix_{template['id']}")]["processed"]["frames"]
    word_frames = [fragments[(engine, f"word_{pair['pair_id']}_{word}")]["processed"]["frames"]
                   for word in words]
    require(start == prefix_frames + 1920 and end == start + max(word_frames)
            and end > start and math.isclose(row["word_slot_start_seconds"], start / 16000, abs_tol=1e-9)
            and math.isclose(row["word_slot_end_seconds"], end / 16000, abs_tol=1e-9),
            "word-slot splice geometry changed")
    expected_frames = end + 1920 + suffix_frames
    require(set(row["outputs"]) == set(words), "missing or added word output")
    prefix_pcm = processed_pcm(fragments[(engine, f"prefix_{template['id']}")], output)
    suffix_pcm = processed_pcm(fragments[(engine, f"suffix_{template['id']}")], output)
    gap = np.zeros(1920, dtype=np.int16)
    wavs, pcm = set(), []
    for word, word_length in zip(words, word_frames, strict=True):
        item = row["outputs"][word]
        path = local_path(item["path"], output, "clips")
        require(path.name == f"{engine}_{pair['pair_id']}_{template['id']}_{word.lower()}.wav"
                and sha(path) == item["file_sha256"], "clip identity/hash changed")
        info = sf.info(path)
        samples, rate = sf.read(path, dtype="int16", always_2d=True)
        require(rate == 16000 and info.subtype == "PCM_16" and info.channels == 1
                and len(samples) == info.frames == item["frames"] == expected_frames
                and item["word_fragment_frames"] == word_length
                and math.isclose(item["duration_seconds"], expected_frames / 16000,
                                 abs_tol=1e-8), "clip format/duration changed")
        samples = samples[:, 0]
        word_pcm = processed_pcm(fragments[(engine, f"word_{pair['pair_id']}_{word}")], output)
        require(len(word_pcm) == word_length, "processed word fragment length changed")
        expected_pcm = np.concatenate((prefix_pcm, gap,
                                       np.pad(word_pcm, (0, end - start - word_length)),
                                       gap, suffix_pcm))
        require(np.array_equal(samples, expected_pcm),
                "clip PCM differs from exact processed-fragment assembly")
        require(hashlib.sha256(samples.astype("<i2").tobytes()).hexdigest() ==
                item["pcm_int16_sha256"], "clip PCM hash changed")
        peak = int(np.max(np.abs(samples.astype(np.int32))))
        require(peak > 0 and peak == item["peak_int16"]
                and item["spoken_script"] == f"{template['prefix']} {word}. {template['suffix']}",
                "clip peak or expected script changed")
        wavs.add(str(path))
        pcm.append(samples)
    a, b = pcm
    require(np.array_equal(a[:start], b[:start])
            and np.array_equal(a[end:], b[end:])
            and not np.array_equal(a[start:end], b[start:end]),
            "pair PCM outside/inside-slot invariant failed")
    return wavs, expected_frames / 16000, min(int(np.max(np.abs(x.astype(np.int32)))) for x in pcm)


def audit(output: Path) -> dict:
    output = output.resolve()
    require(output.is_relative_to(ROOT) and output.is_dir(), "output outside repo or absent")
    require(not (output / "RENDER_FAILED.json").exists(), "render marked incomplete")
    require(FREEZE.is_file(), "freeze receipt missing")
    frozen = json.loads(FREEZE.read_text())
    require(frozen.get("status") == "independently_reviewed_cpu_renderer_frozen"
            and frozen.get("candidate_sha256") == sha(CANDIDATES)
            and frozen.get("candidate_validation_sha256") == sha(VALIDATION)
            and frozen.get("renderer_sha256") == sha(RENDERER)
            and frozen.get("auditor_sha256") == sha(Path(__file__))
            and frozen.get("protocol_sha256") == sha(PROTOCOL),
            "source/input bytes do not match frozen render receipt")
    manifest_path = output / "manifest.json"
    require(manifest_path.is_file(), "no complete render manifest")
    manifest = json.loads(manifest_path.read_text())
    provenance = {"candidate_sha256": CANDIDATES,
                  "candidate_validation_sha256": VALIDATION,
                  "renderer_sha256": RENDERER,
                  "auditor_sha256": Path(__file__),
                  "protocol_sha256": PROTOCOL,
                  "freeze_receipt_sha256": FREEZE,
                  "source_tts_plan_sha256": SOURCE_PLAN,
                  "source_assembly_helper_sha256": ASSEMBLY,
                  "source_tts_renderer_sha256": TTS_RENDER}
    require(all(manifest.get(key) == sha(path) for key, path in provenance.items()),
            "manifest source provenance changed")
    require(manifest["status"] == "cpu_stimuli_only_pending_independent_wav_and_intelligibility_audit"
            and manifest["primary_pairs"] == 60 and manifest["reserve_pairs_used"] == 0
            and manifest["matched_records"] == 240 and manifest["wavs"] == 480,
            "render status/count or reserve rule changed")
    spec = json.loads(CANDIDATES.read_text())
    validation = json.loads(VALIDATION.read_text())
    require(validation["proposal_sha256"] == sha(CANDIDATES), "validation candidate hash changed")
    planned = expected_population(spec, validation)
    plan = json.loads(SOURCE_PLAN.read_text())
    voices = plan["render"]["voices"]
    expected_fragments = {}
    for engine in VOICES:
        for split in SPLITS:
            for template in spec["templates_by_split"][split]:
                expected_fragments[(engine, f"prefix_{template['id']}")] = template["prefix"]
                expected_fragments[(engine, f"suffix_{template['id']}")] = template["suffix"]
        for split in SPLITS:
            for pair in validation["assigned_pairs"][split]:
                for word in pair["words"]:
                    expected_fragments[(engine, f"word_{pair['pair_id']}_{word}")] = word
    require(len(expected_fragments) == 264 and len(manifest["fragments"]) == 264,
            "fragment population count changed")
    fragments = {}
    for fragment in manifest["fragments"]:
        key = (fragment["engine"], fragment["fragment_id"])
        require(key in expected_fragments and key not in fragments, "extra/duplicate fragment")
        audit_fragment(fragment, output, expected_fragments[key], voices[key[0]]["voice"])
        fragments[key] = fragment
    require(set(fragments) == set(expected_fragments), "missing fragment")
    require(len(manifest["pairs"]) == 240, "matched record count changed")
    found, clip_paths, durations, peaks = set(), set(), [], []
    for row in manifest["pairs"]:
        key = (row["pair_id"], row["engine"], row["phrasing_id"])
        require(key in planned and key not in found, "extra/duplicate matched record")
        wavs, duration, peak = audit_pair(row, planned[key], fragments, output)
        require(not clip_paths & wavs, "clip path reused")
        clip_paths |= wavs
        durations.append(duration)
        peaks.append(peak)
        found.add(key)
    require(found == set(planned) and len(clip_paths) == 480, "missing matched record or WAV")
    discovered = {str(path.resolve()) for path in (output / "clips").glob("*.wav")}
    require(discovered == clip_paths, "unexpected or missing clip files")
    for kind in ("raw", "processed"):
        discovered = {str(path.resolve()) for path in (output / kind).glob("*/*.wav")}
        expected = {str(local_path(fragment[kind]["path"], output, kind))
                    for fragment in fragments.values()}
        require(discovered == expected, f"unexpected or missing {kind} fragment")
    require(all(math.isfinite(d) and d > 0 for d in durations), "invalid clip duration")
    return {"status": "cpu_wav_technical_audit_passed_intelligibility_pending_no_model_output",
            "manifest_sha256": sha(manifest_path),
            "freeze_receipt_sha256": sha(FREEZE),
            "auditor_sha256": sha(Path(__file__)),
            "matched_records": len(found), "wavs": len(clip_paths),
            "raw_fragments": 264, "processed_fragments": 264,
            "min_duration_seconds": min(durations),
            "max_duration_seconds": max(durations),
            "min_peak_int16": min(peaks),
            "intelligibility": "pending_independent_asr_and_listening_review",
            "model_inference_authorized": False,
            "limits": "PCM and provenance integrity only; no ASR, intelligibility, Qwen answer, or security claim"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--write-binding", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.output_dir)
    if args.write_binding:
        binding = args.output_dir.resolve() / "TECHNICAL_BINDING.json"
        with binding.open("x") as stream:
            json.dump(result, stream, indent=2)
            stream.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
