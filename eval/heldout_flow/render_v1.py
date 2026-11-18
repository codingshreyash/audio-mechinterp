"""Render a frozen, model-blind held-out audio-flow transfer set.

This program only loads pinned TTS engines and produces audio. It never loads
Qwen, ASR, or any candidate evaluator, and it never filters on their responses.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
import wave
from pathlib import Path

import numpy as np
import soundfile as sf


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TTS_DIR = ROOT / "eval" / "tts"
sys.path.insert(0, str(TTS_DIR))
import render as tts_render  # noqa: E402


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verified(path: Path, expected: str) -> Path:
    actual = sha(path)
    if actual != expected:
        raise ValueError(f"SHA-256 mismatch: {path}: {actual} != {expected}")
    return path


def model_file(config: dict, key: str, hash_key: str, *, download: bool) -> Path:
    return tts_render.pinned_file(
        config, config[key], config[hash_key], download=download
    )


def render_fragment(
    *, engine: str, fragment_id: str, text: str, output_dir: Path,
    voice, piper_synthesis, kokoro, config: dict, preprocessing: dict,
) -> dict:
    raw = output_dir / "raw" / engine / f"{fragment_id}.wav"
    processed = output_dir / "processed" / engine / f"{fragment_id}.wav"
    if engine == "piper":
        with wave.open(str(raw), "wb") as stream:
            voice.synthesize_wav(text, stream, syn_config=piper_synthesis)
    else:
        samples, sample_rate = kokoro.create(
            text, voice=config["voice"], **config["synthesis"]
        )
        tts_render.write_float32_wav(raw, samples, sample_rate)
    raw_record = tts_render.audio_record(raw)
    processed_record = tts_render.to_16k(raw, processed, preprocessing)
    pcm, sample_rate = sf.read(processed, dtype="int16")
    if sample_rate != 16000 or pcm.ndim != 1 or len(pcm) == 0:
        raise ValueError(f"Bad processed fragment: {processed}")
    return {
        "fragment_id": fragment_id,
        "text": text,
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "engine": engine,
        "voice": config["voice"],
        "raw": {**raw_record, "path": str(raw.relative_to(ROOT))},
        "processed": {**processed_record, "path": str(processed.relative_to(ROOT))},
    }


def load_pcm(record: dict) -> np.ndarray:
    path = ROOT / record["processed"]["path"]
    verified(path, record["processed"]["file_sha256"])
    pcm, rate = sf.read(path, dtype="int16")
    if rate != 16000 or pcm.ndim != 1:
        raise ValueError(f"Bad fragment PCM: {path}")
    return pcm


def assemble_pair(
    *, engine: str, pair: dict, phrasing: dict, fragments: dict,
    gap_samples: int, output_dir: Path,
) -> dict:
    prefix = load_pcm(fragments[(engine, f"prefix_{phrasing['id']}")])
    suffix = load_pcm(fragments[(engine, f"suffix_{phrasing['id']}")])
    words = pair["words"]
    word_pcm = {
        word: load_pcm(fragments[(engine, f"word_{pair['pair_id']}_{word}")])
        for word in words
    }
    slot_length = max(len(pcm) for pcm in word_pcm.values())
    gap = np.zeros(gap_samples, dtype=np.int16)
    start = len(prefix) + gap_samples
    end = start + slot_length
    outputs = {}
    saved_pcm = {}
    for word in words:
        slot = np.pad(word_pcm[word], (0, slot_length - len(word_pcm[word])))
        assembled = np.concatenate((prefix, gap, slot, gap, suffix))
        name = f"{engine}_{pair['pair_id']}_{phrasing['id']}_{word.lower()}.wav"
        path = output_dir / "clips" / name
        sf.write(path, assembled, 16000, subtype="PCM_16", format="WAV")
        readback, rate = sf.read(path, dtype="int16")
        if rate != 16000 or not np.array_equal(assembled, readback):
            raise ValueError(f"PCM round trip failed: {path}")
        saved_pcm[word] = readback
        outputs[word] = {
            "path": str(path.relative_to(ROOT)),
            "file_sha256": sha(path),
            "pcm_int16_sha256": hashlib.sha256(readback.astype("<i2").tobytes()).hexdigest(),
            "frames": int(len(readback)),
            "duration_seconds": len(readback) / 16000,
            "word_fragment_frames": int(len(word_pcm[word])),
            "peak_int16": int(np.max(np.abs(readback.astype(np.int32)))),
            "spoken_script": f"{phrasing['prefix']} {word}. {phrasing['suffix']}",
        }
    a, b = (saved_pcm[word] for word in words)
    outside_equal = bool(
        len(a) == len(b)
        and np.array_equal(a[:start], b[:start])
        and np.array_equal(a[end:], b[end:])
    )
    slot_differs = not np.array_equal(a[start:end], b[start:end])
    if not outside_equal or not slot_differs:
        raise ValueError(f"Pair identity check failed: {engine}/{pair['pair_id']}/{phrasing['id']}")
    return {
        "engine": engine,
        "voice": fragments[(engine, f"prefix_{phrasing['id']}")]["voice"],
        "pair_id": pair["pair_id"],
        "phrasing_id": phrasing["id"],
        "words": words,
        "word_slot_start_sample": start,
        "word_slot_end_sample_exclusive": end,
        "word_slot_start_seconds": start / 16000,
        "word_slot_end_seconds": end / 16000,
        "outside_slot_pcm_identical": outside_equal,
        "inside_slot_pcm_differs": slot_differs,
        "outputs": outputs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=HERE / "plan_v1.json")
    parser.add_argument("--output-dir", type=Path, default=HERE / "stimuli_v1")
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()
    plan_path = args.plan.resolve()
    plan = json.loads(plan_path.read_text())
    if plan["status"] != "fixed_before_render_and_before_any_candidate_model_answers":
        raise ValueError("Plan is not frozen for this renderer")
    render = plan["render"]
    for key, hash_key in (
        ("tts_render_source", "tts_render_source_sha256"),
        ("tts_requirements", "tts_requirements_sha256"),
        ("tts_base_pins", "tts_base_pins_sha256"),
    ):
        verified((plan_path.parent / render[key]).resolve(), render[hash_key])
    base_pins = json.loads((plan_path.parent / render["tts_base_pins"]).read_text())
    preprocessing = base_pins["preprocessing"]
    if preprocessing["target_sample_rate"] != 16000 or render["subtype"] != "PCM_16":
        raise ValueError("Unexpected preprocessing plan")
    if {p["pair_id"] for p in plan["word_pairs"]} != {"animal", "weekday"}:
        raise ValueError("Unexpected word-pair selection")
    if any("red" == w.lower() or "blue" == w.lower()
           for pair in plan["word_pairs"] for w in pair["words"]):
        raise ValueError("Fit words leaked into held-out set")
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty output: {output_dir}")

    from piper import PiperVoice
    from piper.config import SynthesisConfig
    from kokoro_onnx import Kokoro

    voices = render["voices"]
    for engine, conf in voices.items():
        if importlib.metadata.version(conf["package"].split("==")[0]) != conf["package"].split("==")[1]:
            raise ValueError(f"TTS package version drift: {engine}")
    piper_conf, kokoro_conf = voices["piper"], voices["kokoro"]
    piper_model = model_file(piper_conf, "model_file", "model_sha256", download=args.allow_download)
    piper_json = model_file(piper_conf, "config_file", "config_sha256", download=args.allow_download)
    kokoro_model = model_file(kokoro_conf, "model_file", "model_sha256", download=args.allow_download)
    kokoro_voices = model_file(kokoro_conf, "voices_file", "voices_sha256", download=args.allow_download)
    voice = PiperVoice.load(piper_model, config_path=piper_json, use_cuda=False)
    kokoro = Kokoro(str(kokoro_model), str(kokoro_voices))
    if kokoro_conf["voice"] not in kokoro.get_voices():
        raise ValueError("Pinned Kokoro voice is absent")
    if "CPUExecutionProvider" not in voice.session.get_providers() or "CPUExecutionProvider" not in kokoro.sess.get_providers():
        raise RuntimeError("TTS engine is not on CPU")
    piper_synthesis = SynthesisConfig(**piper_conf["synthesis"])

    for engine in voices:
        (output_dir / "raw" / engine).mkdir(parents=True)
        (output_dir / "processed" / engine).mkdir(parents=True)
    (output_dir / "clips").mkdir(parents=True)
    texts = {}
    for phrasing in plan["phrasings"]:
        texts[f"prefix_{phrasing['id']}"] = phrasing["prefix"]
        texts[f"suffix_{phrasing['id']}"] = phrasing["suffix"]
    for pair in plan["word_pairs"]:
        if len(pair["words"]) != 2 or len(set(pair["words"])) != 2:
            raise ValueError("Every pair needs exactly two distinct words")
        for word in pair["words"]:
            texts[f"word_{pair['pair_id']}_{word}"] = word
    fragments = {}
    for engine, config in voices.items():
        for fragment_id, text in texts.items():
            fragments[(engine, fragment_id)] = render_fragment(
                engine=engine, fragment_id=fragment_id, text=text,
                output_dir=output_dir, voice=voice, piper_synthesis=piper_synthesis,
                kokoro=kokoro, config=config, preprocessing=preprocessing,
            )
    pairs = [
        assemble_pair(
            engine=engine, pair=pair, phrasing=phrasing,
            fragments=fragments, gap_samples=render["gap_samples"],
            output_dir=output_dir,
        )
        for engine in voices
        for pair in plan["word_pairs"]
        for phrasing in plan["phrasings"]
    ]
    if len(pairs) != plan["expected"]["matched_pairs"] or sum(len(p["outputs"]) for p in pairs) != plan["expected"]["wav_clips"]:
        raise ValueError("Rendered counts disagree with frozen plan")
    report = {
        "status": "stimuli_only_no_model_answers",
        "plan_sha256": sha(plan_path),
        "renderer_sha256": sha(Path(__file__)),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "seed": plan["seed"],
        "random_draws": 0,
        "tts_versions": {e: importlib.metadata.version(c["package"].split("==")[0]) for e, c in voices.items()},
        "system_prompt": plan["system_prompt"],
        "fragments": list(fragments.values()),
        "pairs": pairs,
    }
    manifest = output_dir / "manifest.json"
    manifest.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"manifest": str(manifest), "pairs": len(pairs), "clips": sum(len(p["outputs"]) for p in pairs), "all_outside_pcm_identical": all(p["outside_slot_pcm_identical"] for p in pairs)}, indent=2))


if __name__ == "__main__":
    main()
