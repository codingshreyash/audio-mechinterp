"""Prospective CPU-only v2 TTS renderer; --render requires an independently frozen receipt.

No audio is rendered by --validate-only. No answer/ASR/model inference is used.
All primary pairs are mandatory; reserves are inactive. A partial render cannot
produce TECHNICAL_BINDING.json, which only the independent auditor writes.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

import numpy as np
import soundfile as sf


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CANDIDATES = HERE / "CANDIDATES_UNFROZEN.json"
VALIDATION = HERE / "CANDIDATE_VALIDATION_UNFROZEN.json"
PROTOCOL = HERE / "RENDER_PROTOCOL_PROSPECTIVE_V1.md"
FREEZE = HERE / "RENDER_FREEZE_V1.json"
AUDITOR = HERE / "audit_wavs_v2.py"
SOURCE_PLAN = ROOT / "eval/heldout_flow/plan_v1.json"
HELPER = ROOT / "eval/heldout_flow/render_v1.py"
TTS_RENDER = ROOT / "eval/tts/render.py"
OUTPUT = HERE / "rendered_v2"
EXPECTED = {
    "candidate_sha256": "ac2f9598e4459191569662aa34803c5a72e57ff686d415e60935a4f716a11a93",
    "candidate_validation_sha256": "67e3eec5e14b30a61efbbce449547ec2d5083a87d12a84f0aee8d0e715a59d56",
    "candidate_validator_sha256": "73ccd48bf622445778332edd8e9e63d245afb6cb83d3419c0686655ebf7d9d5b",
    "source_tts_plan_sha256": "13cea31e2e4843d70a6ab26b060812f43691147435c1e10ea9ebaaa0b24efc78",
    "source_assembly_helper_sha256": "bbbd7931dd93121d32b32a9c7898d0a14a3e884f7a421a01d4ecdfd3bd3d063c",
    "source_tts_renderer_sha256": "c3f4f9f71d9c7db8db3af414dbde72ebe7e0a9bc1a0c354ed0d86b8dbe3d9783",
}
SPLITS = ("fit_pool", "development", "locked_test")
VOICES = ("piper", "kokoro")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def load_population() -> tuple[dict, dict, list[dict]]:
    import validate_candidates_unfrozen as selection

    paths = {
        "candidate_sha256": CANDIDATES,
        "candidate_validation_sha256": VALIDATION,
        "candidate_validator_sha256": HERE / "validate_candidates_unfrozen.py",
        "source_tts_plan_sha256": SOURCE_PLAN,
        "source_assembly_helper_sha256": HELPER,
        "source_tts_renderer_sha256": TTS_RENDER,
    }
    for key, path in paths.items():
        require(sha(path) == EXPECTED[key], f"input changed: {key}")
    spec = json.loads(CANDIDATES.read_text())
    report = json.loads(VALIDATION.read_text())
    require(report["status"] == "unfrozen_cpu_candidate_check_passed_no_wavs_or_model_answers"
            and report["proposal_sha256"] == EXPECTED["candidate_sha256"]
            and report["validator_sha256"] == EXPECTED["candidate_validator_sha256"],
            "candidate validation no longer binds selected words")
    old, old_hashes = selection.historical_words()  # word metadata only, no old locked audio
    words, assigned = selection.validate_structure(spec, old)
    require(len(words) == 144 and report["assigned_pairs"] == assigned
            and report["prior_metadata_sha256"] == old_hashes
            and report["candidate_pairs"] == 60 and report["reserve_pairs"] == 12
            and report["rendered_pair_records_planned"] == 240
            and report["wavs_planned"] == 480,
            "candidate validation population or split changed")
    ordered = []
    for split in SPLITS:
        for pair in assigned[split]:
            for engine in VOICES:
                for template in spec["templates_by_split"][split]:
                    ordered.append({"split": split, "pair": pair,
                                    "engine": engine, "template": template})
    keys = [(r["pair"]["pair_id"], r["engine"], r["template"]["id"]) for r in ordered]
    require(len(ordered) == 240 and len(set(keys)) == 240,
            "planned matched-pair count/identity changed")
    return spec, report, ordered


def preflight_tts(spec: dict) -> tuple[dict, dict, dict]:
    """Check local model-file/package pins without loading a voice engine."""
    sys.path.insert(0, str(HELPER.parent))
    import render_v1 as helper

    plan = json.loads(SOURCE_PLAN.read_text())
    voices = plan["render"]["voices"]
    require(set(voices) == set(VOICES) and spec["render_intent"]["voices"] == list(VOICES),
            "voice population changed")
    for engine, config in voices.items():
        package, version = config["package"].split("==")
        require(importlib.metadata.version(package) == version,
                f"TTS package version changed: {engine}")
    for key, hash_key in (("tts_render_source", "tts_render_source_sha256"),
                          ("tts_requirements", "tts_requirements_sha256"),
                          ("tts_base_pins", "tts_base_pins_sha256")):
        path = (SOURCE_PLAN.parent / plan["render"][key]).resolve()
        require(sha(path) == plan["render"][hash_key], f"TTS plan input changed: {key}")
    pins = json.loads((SOURCE_PLAN.parent / plan["render"]["tts_base_pins"]).read_text())
    preprocessing = pins["preprocessing"]
    intent = spec["render_intent"]
    require(preprocessing["target_sample_rate"] == intent["sample_rate"] == 16000
            and intent["channels"] == 1 and intent["subtype"] == "PCM_16"
            and intent["gap_samples"] == 1920, "audio preprocessing grammar changed")
    for engine, config in voices.items():
        keys = (("model_file", "model_sha256"), ("config_file", "config_sha256")) \
            if engine == "piper" else (("model_file", "model_sha256"),
                                       ("voices_file", "voices_sha256"))
        for key, hash_key in keys:
            helper.model_file(config, key, hash_key, download=False)
    return voices, preprocessing, {engine: config["package"] for engine, config in voices.items()}


def freeze_receipt() -> dict:
    require(FREEZE.is_file(), "render freeze receipt absent; independent source review required")
    receipt = json.loads(FREEZE.read_text())
    require(receipt.get("status") == "independently_reviewed_cpu_renderer_frozen"
            and receipt.get("candidate_sha256") == sha(CANDIDATES)
            and receipt.get("candidate_validation_sha256") == sha(VALIDATION)
            and receipt.get("renderer_sha256") == sha(Path(__file__))
            and receipt.get("auditor_sha256") == sha(AUDITOR)
            and receipt.get("protocol_sha256") == sha(PROTOCOL),
            "render freeze receipt does not bind exact source/input bytes")
    return receipt


def fragment_texts(spec: dict, ordered: list[dict]) -> dict[str, str]:
    texts: dict[str, str] = {}
    for split in SPLITS:
        for template in spec["templates_by_split"][split]:
            texts[f"prefix_{template['id']}"] = template["prefix"]
            texts[f"suffix_{template['id']}"] = template["suffix"]
    for item in ordered:
        pair = item["pair"]
        for word in pair["words"]:
            texts[f"word_{pair['pair_id']}_{word}"] = word
    require(len(texts) == 132, "fragment population changed")
    return texts


def render_all(spec: dict, ordered: list[dict], voices: dict, preprocessing: dict,
               output: Path) -> dict:
    """The only function that loads TTS weights or writes audio."""
    sys.path.insert(0, str(HELPER.parent))
    import render_v1 as helper
    from piper import PiperVoice
    from piper.config import SynthesisConfig
    from kokoro_onnx import Kokoro

    piper_conf, kokoro_conf = voices["piper"], voices["kokoro"]
    piper = PiperVoice.load(helper.model_file(piper_conf, "model_file", "model_sha256", download=False),
                            config_path=helper.model_file(piper_conf, "config_file", "config_sha256", download=False),
                            use_cuda=False)
    kokoro = Kokoro(str(helper.model_file(kokoro_conf, "model_file", "model_sha256", download=False)),
                    str(helper.model_file(kokoro_conf, "voices_file", "voices_sha256", download=False)))
    require("CPUExecutionProvider" in piper.session.get_providers()
            and "CPUExecutionProvider" in kokoro.sess.get_providers(),
            "TTS execution provider is not CPU")
    require(kokoro_conf["voice"] in kokoro.get_voices(), "Kokoro voice changed")
    piper_synthesis = SynthesisConfig(**piper_conf["synthesis"])
    for engine in VOICES:
        (output / "raw" / engine).mkdir(parents=True)
        (output / "processed" / engine).mkdir(parents=True)
    (output / "clips").mkdir()
    fragments = {}
    for engine in VOICES:
        for fragment_id, text in fragment_texts(spec, ordered).items():
            fragments[(engine, fragment_id)] = helper.render_fragment(
                engine=engine, fragment_id=fragment_id, text=text,
                output_dir=output, voice=piper, piper_synthesis=piper_synthesis,
                kokoro=kokoro, config=voices[engine], preprocessing=preprocessing)
    require(len(fragments) == 264, "rendered fragment count changed")
    records = []
    for item in ordered:
        pair, template, engine = item["pair"], item["template"], item["engine"]
        row = helper.assemble_pair(engine=engine, pair=pair, phrasing=template,
                                   fragments=fragments, gap_samples=1920, output_dir=output)
        row.update(split=item["split"], category=pair["category"],
                   fit_role=pair.get("fit_role"), sample_rate=16000, subtype="PCM_16")
        ids = (f"prefix_{template['id']}",
               *(f"word_{pair['pair_id']}_{word}" for word in pair["words"]),
               f"suffix_{template['id']}")
        row["fragment_sha256"] = {id_: {
            "raw": fragments[(engine, id_)]["raw"]["file_sha256"],
            "processed": fragments[(engine, id_)]["processed"]["file_sha256"]}
            for id_ in ids}
        records.append(row)
    require(len(records) == 240 and sum(len(r["outputs"]) for r in records) == 480,
            "rendered population incomplete")
    return {"fragments": [fragments[(engine, id_)] for engine in VOICES
                          for id_ in fragment_texts(spec, ordered)], "pairs": records}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--render", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    spec, report, ordered = load_population()
    voices, preprocessing, versions = preflight_tts(spec)
    if args.validate_only:
        print(json.dumps({"status": "validated_cpu_inputs_no_render_no_model_output",
                          "planned_matched_records": len(ordered),
                          "planned_wavs": 2 * len(ordered),
                          "candidate_sha256": sha(CANDIDATES),
                          "validation_sha256": sha(VALIDATION),
                          "tts_packages": versions}, indent=2))
        return
    receipt = freeze_receipt()
    output = args.output_dir.resolve()
    require(output.is_relative_to(ROOT), "output must be inside repository")
    require(not output.exists(), "output exists; refusing to overwrite or resume a partial render")
    output.mkdir(parents=True, exist_ok=False)
    try:
        body = render_all(spec, ordered, voices, preprocessing, output)
        manifest = {
            "status": "cpu_stimuli_only_pending_independent_wav_and_intelligibility_audit",
            "candidate_sha256": sha(CANDIDATES), "candidate_validation_sha256": sha(VALIDATION),
            "renderer_sha256": sha(Path(__file__)), "auditor_sha256": sha(AUDITOR),
            "protocol_sha256": sha(PROTOCOL), "freeze_receipt_sha256": sha(FREEZE),
            "source_tts_plan_sha256": sha(SOURCE_PLAN),
            "source_assembly_helper_sha256": sha(HELPER),
            "source_tts_renderer_sha256": sha(TTS_RENDER),
            "python": sys.version.split()[0], "platform": platform.platform(),
            "tts_packages": versions, "primary_pairs": 60,
            "reserve_pairs_used": 0, "matched_records": 240, "wavs": 480,
            **body,
        }
        with (output / "manifest.json").open("x") as stream:
            stream.write(json.dumps(manifest, indent=2) + "\n")
    except Exception as exc:
        with (output / "RENDER_FAILED.json").open("x") as stream:
            stream.write(json.dumps({
                "status": "incomplete_render_no_model_inference", "error_type": type(exc).__name__,
                "error": str(exc)}, indent=2) + "\n")
        raise
    print(json.dumps({"status": manifest["status"], "output": str(output),
                      "matched_records": 240, "wavs": 480,
                      "manifest_sha256": sha(output / "manifest.json"),
                      "intelligibility": "pending"}, indent=2))


if __name__ == "__main__":
    main()
