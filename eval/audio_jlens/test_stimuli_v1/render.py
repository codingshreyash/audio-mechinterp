"""Render the audio-Jacobian locked test selection without model inference.

This consumes SPLIT_V1.json, reuses the already pinned CPU TTS/assembly helpers,
and writes eight matched pairs plus an immutable split-to-manifest binding.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
EVAL = HERE.parents[1]
ROOT = HERE.parents[2]
SPLIT = HERE.parent / "SPLIT_V1.json"
HELPER_PATH = EVAL / "heldout_flow" / "render_v1.py"
EXPECTED_HELPER_SHA = "bbbd7931dd93121d32b32a9c7898d0a14a3e884f7a421a01d4ecdfd3bd3d063c"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_sha(path: Path, expected: str) -> None:
    actual = sha(path)
    if actual != expected:
        raise ValueError(f"SHA-256 mismatch: {path}: {actual} != {expected}")


def main() -> None:
    for name in ("manifest.json", "BINDING.json"):
        if (HERE / name).exists():
            raise FileExistsError(f"Refusing to overwrite frozen {name}")
    for name in ("raw", "processed", "clips"):
        if (HERE / name).exists():
            raise FileExistsError(f"Refusing to reuse {name}")
    require_sha(HELPER_PATH, EXPECTED_HELPER_SHA)
    sys.path.insert(0, str(HELPER_PATH.parent))
    import render_v1 as helper  # noqa: E402

    split = json.loads(SPLIT.read_text())
    if split["status"] != "stimulus_selection_frozen_before_audio_jacobian_candidate_or_test_results":
        raise ValueError("Split is not marked frozen")
    test = split["new_locked_test"]
    if test["wav_manifest_sha256"] is not None:
        raise ValueError("Split file must remain the pre-render selection")
    source_plan = (SPLIT.parent / test["source_plan"]).resolve()
    require_sha(source_plan, test["source_plan_sha256"])
    tts_plan = json.loads(source_plan.read_text())
    for field, digest_field in (
        ("tts_render_source", "tts_render_source_sha256"),
        ("tts_requirements", "tts_requirements_sha256"),
        ("tts_base_pins", "tts_base_pins_sha256"),
    ):
        require_sha((source_plan.parent / tts_plan["render"][field]).resolve(), tts_plan["render"][digest_field])
    base_pins = json.loads((source_plan.parent / tts_plan["render"]["tts_base_pins"]).read_text())
    preprocessing = base_pins["preprocessing"]
    render = tts_plan["render"]
    voices = render["voices"]
    if render["sample_rate"] != 16000 or render["subtype"] != "PCM_16" or render["gap_samples"] != 1920:
        raise ValueError("Unexpected frozen rendering rules")
    if len(test["pairs"]) != 2 or len(test["phrasings"]) != 2 or set(voices) != {"piper", "kokoro"}:
        raise ValueError("Unexpected test selection dimensions")
    if {word.lower() for pair in test["pairs"] for word in pair["words"]} & {"red", "blue", "cat", "dog", "monday", "friday"}:
        raise ValueError("A fit/development word leaked into test")

    from piper import PiperVoice
    from piper.config import SynthesisConfig
    from kokoro_onnx import Kokoro

    for engine, conf in voices.items():
        package, expected_version = conf["package"].split("==")
        if importlib.metadata.version(package) != expected_version:
            raise RuntimeError(f"TTS package drift: {engine}")
    piper_conf, kokoro_conf = voices["piper"], voices["kokoro"]
    piper_model = helper.model_file(piper_conf, "model_file", "model_sha256", download=False)
    piper_json = helper.model_file(piper_conf, "config_file", "config_sha256", download=False)
    kokoro_model = helper.model_file(kokoro_conf, "model_file", "model_sha256", download=False)
    kokoro_voices = helper.model_file(kokoro_conf, "voices_file", "voices_sha256", download=False)
    voice = PiperVoice.load(piper_model, config_path=piper_json, use_cuda=False)
    kokoro = Kokoro(str(kokoro_model), str(kokoro_voices))
    if kokoro_conf["voice"] not in kokoro.get_voices():
        raise ValueError("Kokoro voice absent")
    if "CPUExecutionProvider" not in voice.session.get_providers() or "CPUExecutionProvider" not in kokoro.sess.get_providers():
        raise RuntimeError("Unexpected non-CPU TTS provider")
    piper_synthesis = SynthesisConfig(**piper_conf["synthesis"])

    for engine in voices:
        (HERE / "raw" / engine).mkdir(parents=True)
        (HERE / "processed" / engine).mkdir(parents=True)
    (HERE / "clips").mkdir()
    fragment_texts: dict[str, str] = {}
    for phrasing in test["phrasings"]:
        fragment_texts[f"prefix_{phrasing['id']}"] = phrasing["prefix"]
        fragment_texts[f"suffix_{phrasing['id']}"] = phrasing["suffix"]
    for pair in test["pairs"]:
        if len(pair["words"]) != 2 or len(set(pair["words"])) != 2:
            raise ValueError("A test word pair is invalid")
        for word in pair["words"]:
            fragment_texts[f"word_{pair['pair_id']}_{word}"] = word
    fragments: dict[tuple[str, str], dict] = {}
    for engine, conf in voices.items():
        for fragment_id, text in fragment_texts.items():
            fragments[(engine, fragment_id)] = helper.render_fragment(
                engine=engine, fragment_id=fragment_id, text=text,
                output_dir=HERE, voice=voice, piper_synthesis=piper_synthesis,
                kokoro=kokoro, config=conf, preprocessing=preprocessing,
            )
    pairs = [
        helper.assemble_pair(
            engine=engine, pair=pair, phrasing=phrasing,
            fragments=fragments, gap_samples=render["gap_samples"], output_dir=HERE,
        )
        for engine in voices
        for pair in test["pairs"]
        for phrasing in test["phrasings"]
    ]
    n_clips = sum(len(p["outputs"]) for p in pairs)
    if len(pairs) != test["expected_matched_pairs"] or n_clips != test["expected_clips"]:
        raise ValueError("Rendered count differs from frozen test split")
    if not all(p["outside_slot_pcm_identical"] and p["inside_slot_pcm_differs"] for p in pairs):
        raise ValueError("PCM pair invariants failed")
    report = {
        "status": "locked_stimuli_only_no_qwen_or_asr_inference",
        "split_sha256": sha(SPLIT),
        "source_tts_plan_sha256": sha(source_plan),
        "source_render_helper_sha256": sha(HELPER_PATH),
        "renderer_sha256": sha(Path(__file__)),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "seed_identifier": split["seed"],
        "tts_versions": {
            engine: importlib.metadata.version(conf["package"].split("==")[0])
            for engine, conf in voices.items()
        },
        "selection": test,
        "system_prompt": tts_plan["system_prompt"],
        "fragments": list(fragments.values()),
        "pairs": pairs,
    }
    manifest = HERE / "manifest.json"
    manifest.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    binding = {
        "split_path": str(SPLIT.relative_to(ROOT)),
        "split_sha256": sha(SPLIT),
        "manifest_path": str(manifest.relative_to(ROOT)),
        "manifest_sha256": sha(manifest),
        "rendered_pairs": len(pairs),
        "rendered_clips": n_clips,
        "all_pairs_exact_outside_word_pcm": True,
        "status": "frozen_before_jacobian_candidate_run",
    }
    (HERE / "BINDING.json").write_text(json.dumps(binding, indent=2) + "\n")
    print(json.dumps(binding, indent=2))


if __name__ == "__main__":
    main()
