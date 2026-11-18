"""CPU-only pinned TTS rendering of the frozen 48-pair corpus selection.

No Qwen model, ASR model, or response scorer is loaded. This script refuses to
overwrite output and verifies exact outside-word int16 PCM equality.
"""

from __future__ import annotations

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
SPEC = HERE / "SPEC_V1.json"
OUTPUT = HERE / "rendered"
SOURCE_PLAN = ROOT / "eval" / "heldout_flow" / "plan_v1.json"
HELPER_PATH = ROOT / "eval" / "heldout_flow" / "render_v1.py"
EXPECTED_HELPER_SHA = "bbbd7931dd93121d32b32a9c7898d0a14a3e884f7a421a01d4ecdfd3bd3d063c"
TEMPLATE_SPLIT = {"fit_pool": "fit", "development": "development", "locked_test": "locked_test"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_spec(spec: dict) -> dict[str, list[dict]]:
    if spec["status"] != "selection_frozen_before_cpu_render_and_candidate_answers":
        raise ValueError("Corpus selection not frozen")
    if spec["spec_generator_sha256"] != sha(HERE / "make_spec.py"):
        raise ValueError("Spec generator source changed")
    if spec["source_tts_plan_sha256"] != sha(SOURCE_PLAN):
        raise ValueError("Source TTS plan changed")
    if spec["source_tts_render_helper_sha256"] != sha(ROOT / "eval" / "tts" / "render.py"):
        raise ValueError("Underlying TTS renderer changed")
    if sha(HELPER_PATH) != EXPECTED_HELPER_SHA:
        raise ValueError("TTS assembly helper changed")
    selection = spec["word_pairs"]
    if {key: len(value) for key, value in selection.items()} != {
        "fit_pool": 24, "development": 12, "locked_test": 12
    }:
        raise ValueError("Split counts changed")
    if sum(bool(p.get("map_fit")) for p in selection["fit_pool"]) != 12:
        raise ValueError("Map-fit count changed")
    all_pairs = [p for split in selection.values() for p in split]
    all_words = [word.casefold() for pair in all_pairs for word in pair["words"]]
    if len(all_words) != 96 or len(set(all_words)) != 96:
        raise ValueError("Word overlap between pairs/splits")
    if set(all_words) & set(spec["historical_excluded_words"]):
        raise ValueError("Historical word entered corpus")
    if set(spec["templates"]) != set(TEMPLATE_SPLIT.values()):
        raise ValueError("Template splits differ")
    template_ids = [t["id"] for group in spec["templates"].values() for t in group]
    if len(template_ids) != 6 or len(set(template_ids)) != 6:
        raise ValueError("Templates not split-exclusive")
    for split, templates in spec["templates"].items():
        if len(templates) != 2 or not all(t["prefix"] and t["suffix"] for t in templates):
            raise ValueError(f"Bad templates for {split}")
    aliases = spec["aliases"]
    if set(aliases) != {w for pair in all_pairs for w in pair["words"]}:
        raise ValueError("Alias/word coverage mismatch")
    for pair in all_pairs:
        a, b = pair["words"]
        ai = {x["token_id"] for x in aliases[a]}
        bi = {x["token_id"] for x in aliases[b]}
        if not ai or not bi or ai & bi:
            raise ValueError(f"Invalid alias ID sets for {pair['pair_id']}")
    return selection


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to reuse output: {OUTPUT}")
    spec = json.loads(SPEC.read_text())
    selection = check_spec(spec)
    sys.path.insert(0, str(HELPER_PATH.parent))
    import render_v1 as helper  # noqa: E402

    from piper import PiperVoice
    from piper.config import SynthesisConfig
    from kokoro_onnx import Kokoro

    voices = spec["render"]["voices"]
    if set(voices) != {"piper", "kokoro"}:
        raise ValueError("Expected pinned Piper and Kokoro")
    for engine, conf in voices.items():
        package, version = conf["package"].split("==")
        if importlib.metadata.version(package) != version:
            raise RuntimeError(f"TTS package drift: {engine}")
    source_plan = json.loads(SOURCE_PLAN.read_text())
    if voices != source_plan["render"]["voices"]:
        raise ValueError("Voice pins differ from source plan")
    source_pins_path = (SOURCE_PLAN.parent / source_plan["render"]["tts_base_pins"]).resolve()
    if sha(source_pins_path) != source_plan["render"]["tts_base_pins_sha256"]:
        raise ValueError("TTS preprocessing pins changed")
    preprocessing = json.loads(source_pins_path.read_text())["preprocessing"]
    piper_conf, kokoro_conf = voices["piper"], voices["kokoro"]
    piper_model = helper.model_file(piper_conf, "model_file", "model_sha256", download=False)
    piper_json = helper.model_file(piper_conf, "config_file", "config_sha256", download=False)
    kokoro_model = helper.model_file(kokoro_conf, "model_file", "model_sha256", download=False)
    kokoro_voices = helper.model_file(kokoro_conf, "voices_file", "voices_sha256", download=False)
    piper = PiperVoice.load(piper_model, config_path=piper_json, use_cuda=False)
    kokoro = Kokoro(str(kokoro_model), str(kokoro_voices))
    if "CPUExecutionProvider" not in piper.session.get_providers() or "CPUExecutionProvider" not in kokoro.sess.get_providers():
        raise RuntimeError("TTS provider is not CPU")
    if kokoro_conf["voice"] not in kokoro.get_voices():
        raise ValueError("Kokoro voice changed")
    piper_synthesis = SynthesisConfig(**piper_conf["synthesis"])

    for engine in voices:
        (OUTPUT / "raw" / engine).mkdir(parents=True)
        (OUTPUT / "processed" / engine).mkdir(parents=True)
    (OUTPUT / "clips").mkdir()
    all_pairs = [p for split in selection.values() for p in split]
    texts = {}
    for templates in spec["templates"].values():
        for template in templates:
            texts[f"prefix_{template['id']}"] = template["prefix"]
            texts[f"suffix_{template['id']}"] = template["suffix"]
    for pair in all_pairs:
        for word in pair["words"]:
            texts[f"word_{pair['pair_id']}_{word}"] = word
    fragments = {}
    for engine, conf in voices.items():
        for fragment_id, text in texts.items():
            fragments[(engine, fragment_id)] = helper.render_fragment(
                engine=engine, fragment_id=fragment_id, text=text,
                output_dir=OUTPUT, voice=piper, piper_synthesis=piper_synthesis,
                kokoro=kokoro, config=conf, preprocessing=preprocessing,
            )
    pairs = []
    for split, selected in selection.items():
        for engine in voices:
            for pair in selected:
                for template in spec["templates"][TEMPLATE_SPLIT[split]]:
                    record = helper.assemble_pair(
                        engine=engine, pair=pair, phrasing=template,
                        fragments=fragments,
                        gap_samples=spec["render"]["gap_samples"], output_dir=OUTPUT,
                    )
                    record["split"] = split
                    record["map_fit"] = bool(pair.get("map_fit", False))
                    pairs.append(record)
    n_clips = sum(len(p["outputs"]) for p in pairs)
    if len(pairs) != 192 or n_clips != 384:
        raise ValueError("Rendered counts differ from frozen spec")
    if not all(p["outside_slot_pcm_identical"] and p["inside_slot_pcm_differs"] for p in pairs):
        raise ValueError("PCM splice invariant failed")
    for pair in pairs:
        a, b = pair["words"]
        readback = {}
        for word in (a, b):
            record = pair["outputs"][word]
            path = ROOT / record["path"]
            if sha(path) != record["file_sha256"]:
                raise ValueError(f"Output WAV hash changed: {path}")
            pcm, rate = sf.read(path, dtype="int16")
            if rate != 16000 or pcm.ndim != 1 or len(pcm) != record["frames"]:
                raise ValueError(f"Output WAV structure changed: {path}")
            expected_pcm_sha = hashlib.sha256(pcm.astype("<i2").tobytes()).hexdigest()
            if expected_pcm_sha != record["pcm_int16_sha256"]:
                raise ValueError(f"Output PCM hash changed: {path}")
            readback[word] = pcm
        left, right = readback[a], readback[b]
        start = pair["word_slot_start_sample"]
        end = pair["word_slot_end_sample_exclusive"]
        if (len(left) != len(right) or
                not np.array_equal(left[:start], right[:start]) or
                not np.array_equal(left[end:], right[end:]) or
                np.array_equal(left[start:end], right[start:end])):
            raise ValueError(f"Independent PCM pair check failed: {pair['pair_id']}")
    report = {
        "status": "cpu_stimuli_only_no_candidate_or_asr_model_answers",
        "spec_sha256": sha(SPEC), "renderer_sha256": sha(Path(__file__)),
        "source_assembly_helper_sha256": sha(HELPER_PATH),
        "source_tts_renderer_sha256": sha(ROOT / "eval" / "tts" / "render.py"),
        "source_tts_plan_sha256": sha(SOURCE_PLAN),
        "platform": platform.platform(), "python": sys.version.split()[0],
        "seed": spec["seed"],
        "tts_versions": {
            e: importlib.metadata.version(c["package"].split("==")[0])
            for e, c in voices.items()
        },
        "system_prompt": spec["system_prompt"],
        "fragments": list(fragments.values()), "pairs": pairs,
        "expected": spec["expected"],
    }
    manifest = OUTPUT / "manifest.json"
    manifest.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    binding = {
        "spec_sha256": sha(SPEC), "manifest_sha256": sha(manifest),
        "rendered_matched_pairs": len(pairs), "rendered_clips": n_clips,
        "all_outside_word_pcm_equal": True,
        "status": "rendered_without_candidate_model_answers",
    }
    (OUTPUT / "BINDING.json").write_text(json.dumps(binding, indent=2) + "\n")
    print(json.dumps(binding, indent=2))


if __name__ == "__main__":
    main()
