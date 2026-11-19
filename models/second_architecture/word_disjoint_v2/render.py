"""Render 32 prospective, paired English WAVs from Piper Amy and macOS Samantha.

This program never loads an ASR or assistant model. ``--render`` requires a
separately reviewed, hash-bound freeze receipt. It refuses existing output.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
import wave
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PLAN = HERE.parent / "WORD_DISJOINT_TIME_STATE_VALIDATION_PLAN_V2.md"
V1_SPEC = ROOT / "eval/audio_jlens/corpus_v1/SPEC_V1.json"
V1_MANIFEST = ROOT / "eval/audio_jlens/corpus_v1/rendered/manifest.json"
V2_MANIFEST = ROOT / "eval/audio_jlens/corpus_v2/rendered_v2/manifest.json"
SOURCE_PLAN = ROOT / "eval/heldout_flow/plan_v1.json"
SOURCE_HELPER = ROOT / "eval/heldout_flow/render_v1.py"
TTS_SOURCE = ROOT / "eval/tts/render.py"
FREEZE = HERE / "RENDER_FREEZE_V2.json"
OUTPUT = HERE / "rendered_v2"
SAY = Path("/usr/bin/say")
AFCONVERT = Path("/usr/bin/afconvert")
EXPECTED_METADATA_SHA = {
    "v1_spec": "4c4cb6beafdd27509a073872f71dea004be686a6d8fafccbae410b8b11d7f265",
    "v1_manifest": "adf8cd02d24a43ef576bcb993dc54bd2527440a62798e12fc788def3c063971f",
    "v2_manifest": "15395c7d132eb3f1c1a6b2b7d555146faa83be4ee2b77fba4f481cf70f26761d",
}
PAIRS = (
    ("banana", "tomato"), ("carrot", "cookie"),
    ("train", "plane"), ("ferry", "taxi"),
    ("guitar", "piano"), ("violin", "trumpet"),
    ("broom", "brush"), ("button", "ribbon"),
)
PREFIX = "I said"
SUFFIX = "and then stopped."
GAP_SAMPLES = 1920
SAMPLE_RATE = 16000


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def reference_script(word: str) -> str:
    return f"{PREFIX} {word} {SUFFIX}"


def read_pcm(path: Path) -> bytes:
    with wave.open(str(path), "rb") as stream:
        require(stream.getnchannels() == 1 and stream.getframerate() == SAMPLE_RATE
                and stream.getsampwidth() == 2 and stream.getcomptype() == "NONE",
                f"not mono 16-kHz PCM16: {path}")
        data = stream.readframes(stream.getnframes())
    require(bool(data) and len(data) % 2 == 0, f"empty/odd PCM: {path}")
    return data


def write_pcm(path: Path, pcm: bytes) -> None:
    require(len(pcm) % 2 == 0, "odd PCM byte count")
    with path.open("xb") as binary:
        with wave.open(binary, "wb") as stream:
            stream.setnchannels(1)
            stream.setsampwidth(2)
            stream.setframerate(SAMPLE_RATE)
            stream.writeframes(pcm)


def verify_population() -> dict[str, str]:
    paths = {"v1_spec": V1_SPEC, "v1_manifest": V1_MANIFEST,
             "v2_manifest": V2_MANIFEST}
    hashes = {name: sha(path) for name, path in paths.items()}
    require(hashes == EXPECTED_METADATA_SHA, "prior metadata hash drift")
    v1 = json.loads(V1_SPEC.read_text())
    v2 = json.loads(V2_MANIFEST.read_text())
    words = [word.casefold() for pair in PAIRS for word in pair]
    require(len(words) == 16 and len(set(words)) == 16, "new words overlap")
    prior_splits = {
        split: {word.casefold() for pair in pairs for word in pair["words"]}
        for split, pairs in v1["word_pairs"].items()
    }
    require(set(prior_splits) == {"fit_pool", "development", "locked_test"},
            "prior split metadata changed")
    previous = set().union(*prior_splits.values())
    previous |= {word.casefold() for word in v1["historical_excluded_words"]}
    v2_words = {word.casefold() for row in v2["pairs"] for word in row["words"]}
    require(not (set(words) & previous) and not (set(words) & v2_words),
            "new word occurs in prior corpus metadata")
    return hashes


def piper_configuration() -> dict:
    from_plan = json.loads(SOURCE_PLAN.read_text())
    helper_hash = sha(SOURCE_HELPER)
    require(from_plan["render"]["tts_render_source_sha256"] == sha(TTS_SOURCE)
            and from_plan["render"]["tts_base_pins_sha256"] ==
            sha((SOURCE_PLAN.parent / from_plan["render"]["tts_base_pins"]).resolve())
            and helper_hash == "bbbd7931dd93121d32b32a9c7898d0a14a3e884f7a421a01d4ecdfd3bd3d063c",
            "pinned TTS source changed")
    conf = from_plan["render"]["voices"]["piper"]
    require(conf["voice"] == "en_US-amy-medium"
            and conf["package"] == "piper-tts==1.8.0"
            and conf["synthesis"] == {
                "speaker_id": None, "length_scale": 1.0, "noise_scale": 0.0,
                "noise_w_scale": 0.0, "normalize_audio": False, "volume": 1.0,
            }, "Piper synthesis pin changed")
    return conf


def command_identity() -> dict:
    require(SAY.is_file() and AFCONVERT.is_file(), "macOS TTS commands absent")
    voices = subprocess.run([str(SAY), "-v", "?"], check=True,
                            capture_output=True, text=True).stdout
    require(any(line.split()[:2] == ["Samantha", "en_US"]
                for line in voices.splitlines()), "Samantha en_US absent")
    version = subprocess.run(["sw_vers", "-buildVersion"], check=True,
                             capture_output=True, text=True).stdout.strip()
    help_result = subprocess.run([str(AFCONVERT), "-h"], check=False,
                                 capture_output=True, text=True)
    help_text = help_result.stdout + help_result.stderr
    require(version != "" and help_result.returncode in (0, 2)
            and "Audio File Convert" in help_text,
            "macOS conversion identity unavailable")
    return {"os_build": version, "say_path": str(SAY), "say_sha256": sha(SAY),
            "say_voice": "Samantha", "say_language": "en_US",
            "afconvert_path": str(AFCONVERT), "afconvert_sha256": sha(AFCONVERT),
            "afconvert_help_sha256": sha_bytes(help_text.encode()),
            "afconvert_argv_template": [str(AFCONVERT), "-f", "WAVE", "-d",
                                         "LEI16@16000", "-c", "1", "RAW", "CONVERTED"]}


def freeze_receipt() -> dict:
    require(FREEZE.is_file(), "independently reviewed render freeze receipt absent")
    receipt = json.loads(FREEZE.read_text())
    expected = {"plan_sha256": sha(PLAN), "renderer_sha256": sha(Path(__file__)),
                "auditor_sha256": sha(HERE / "audit.py"),
                "queue_sha256": sha(HERE / "queue.py"),
                "source_tts_plan_sha256": sha(SOURCE_PLAN),
                "source_tts_helper_sha256": sha(SOURCE_HELPER)}
    require(receipt.get("status") == "independently_reviewed_renderer_frozen"
            and all(receipt.get(key) == value for key, value in expected.items()),
            "render freeze receipt does not bind reviewed source")
    return receipt


def convert(raw: Path, processed: Path) -> dict:
    argv = [str(AFCONVERT), "-f", "WAVE", "-d", "LEI16@16000",
            "-c", "1", str(raw), str(processed)]
    subprocess.run(argv, check=True, capture_output=True)
    pcm = read_pcm(processed)
    return {"raw": {"path": str(raw.relative_to(ROOT)), "sha256": sha(raw)},
            "processed": {"path": str(processed.relative_to(ROOT)),
                          "sha256": sha(processed), "pcm_sha256": sha_bytes(pcm),
                          "frames": len(pcm) // 2},
            "conversion_argv": argv}


def render_all(output: Path, conf: dict, identity: dict, metadata_hashes: dict) -> dict:
    sys.path.insert(0, str(SOURCE_HELPER.parent))
    import render_v1 as helper  # noqa: E402
    from piper import PiperVoice  # noqa: E402
    from piper.config import SynthesisConfig  # noqa: E402

    package, version = conf["package"].split("==")
    require(importlib.metadata.version(package) == version, "Piper package drift")
    model = helper.model_file(conf, "model_file", "model_sha256", download=False)
    config = helper.model_file(conf, "config_file", "config_sha256", download=False)
    voice = PiperVoice.load(model, config_path=config, use_cuda=False)
    require("CPUExecutionProvider" in voice.session.get_providers(),
            "Piper execution provider is not CPU")
    synthesis = SynthesisConfig(**conf["synthesis"])
    fragment_texts = {"prefix": PREFIX, "suffix": SUFFIX}
    fragment_texts.update({f"word_{i:02d}_{word}": word
                           for i, pair in enumerate(PAIRS) for word in pair})
    require(len(fragment_texts) == 18, "fragment count changed")
    for engine in ("piper", "samantha"):
        (output / "raw" / engine).mkdir(parents=True)
        (output / "processed" / engine).mkdir(parents=True)
    (output / "clips").mkdir()
    fragments: dict[tuple[str, str], dict] = {}
    for engine in ("piper", "samantha"):
        for fragment_id, text in fragment_texts.items():
            raw = output / "raw" / engine / (fragment_id + (".wav" if engine == "piper" else ".aiff"))
            processed = output / "processed" / engine / f"{fragment_id}.wav"
            if engine == "piper":
                with raw.open("xb") as binary:
                    with wave.open(binary, "wb") as stream:
                        voice.synthesize_wav(text, stream, syn_config=synthesis)
            else:
                subprocess.run([str(SAY), "-v", "Samantha", "-o", str(raw), text],
                               check=True, capture_output=True)
            record = convert(raw, processed)
            record.update(engine=engine, fragment_id=fragment_id, text=text,
                          text_sha256=sha_bytes(text.encode()),
                          voice=conf["voice"] if engine == "piper" else "Samantha")
            fragments[(engine, fragment_id)] = record
    rows = []
    gap = bytes(2 * GAP_SAMPLES)
    for i, pair in enumerate(PAIRS):
        for engine in ("piper", "samantha"):
            prefix = read_pcm(ROOT / fragments[(engine, "prefix")]["processed"]["path"])
            suffix = read_pcm(ROOT / fragments[(engine, "suffix")]["processed"]["path"])
            word_pcm = {word: read_pcm(ROOT / fragments[(engine, f"word_{i:02d}_{word}")]["processed"]["path"])
                        for word in pair}
            slot_frames = max(len(pcm) for pcm in word_pcm.values()) // 2
            start = len(prefix) // 2 + GAP_SAMPLES
            end = start + slot_frames
            outputs = {}
            for word in pair:
                padded = word_pcm[word] + bytes(2 * slot_frames - len(word_pcm[word]))
                pcm = prefix + gap + padded + gap + suffix
                path = output / "clips" / f"{engine}_{i:02d}_{word}.wav"
                write_pcm(path, pcm)
                require(read_pcm(path) == pcm, "rendered WAV round trip changed PCM")
                outputs[word] = {"path": str(path.relative_to(ROOT)),
                                 "file_sha256": sha(path), "pcm_int16_sha256": sha_bytes(pcm),
                                 "frames": len(pcm) // 2, "word_fragment_frames": len(word_pcm[word]) // 2,
                                 "spoken_script": reference_script(word)}
            a = read_pcm(ROOT / outputs[pair[0]]["path"])
            b = read_pcm(ROOT / outputs[pair[1]]["path"])
            require(len(a) == len(b) and a[:2*start] == b[:2*start]
                    and a[2*end:] == b[2*end:] and a[2*start:2*end] != b[2*start:2*end],
                    "paired PCM invariant failed")
            rows.append({"pair_id": f"pair_{i:02d}", "words": list(pair),
                         "engine": engine, "voice": conf["voice"] if engine == "piper" else "Samantha",
                         "word_slot_start_sample": start, "word_slot_end_sample_exclusive": end,
                         "prefix_fragment_id": "prefix", "suffix_fragment_id": "suffix",
                         "word_fragment_ids": {word: f"word_{i:02d}_{word}" for word in pair},
                         "outside_slot_pcm_identical": True, "inside_slot_pcm_differs": True,
                         "outputs": outputs})
    require(len(fragments) == 36 and len(rows) == 16 and
            sum(len(row["outputs"]) for row in rows) == 32, "render population incomplete")
    return {"status": "synthetic_stimuli_only_no_candidate_model_answers",
            "renderer_sha256": sha(Path(__file__)), "plan_sha256": sha(PLAN),
            "freeze_receipt_sha256": sha(FREEZE),
            "source_tts_plan_sha256": sha(SOURCE_PLAN),
            "source_tts_helper_sha256": sha(SOURCE_HELPER),
            "piper_checkpoint_sha256": sha(model), "piper_config_sha256": sha(config),
            "piper_package": conf["package"], "piper_synthesis": conf["synthesis"],
            "samantha_command_identity": identity,
            "python": sys.version.split()[0], "platform": platform.platform(),
            "metadata_sha256": metadata_hashes, "sample_rate": SAMPLE_RATE,
            "subtype": "PCM_16", "gap_samples": GAP_SAMPLES,
            "prefix": PREFIX, "suffix": SUFFIX,
            "fragments": list(fragments.values()), "pairs": rows,
            "matched_pairs": 16, "wavs": 32}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--render", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    metadata_hashes = verify_population()
    conf = piper_configuration()
    identity = command_identity()
    if args.validate_only:
        print(json.dumps({"status": "validated_metadata_and_commands_no_audio_rendered",
                          "pairs": 16, "wavs": 32, "metadata_sha256": metadata_hashes,
                          "samantha": identity["say_voice"], "piper": conf["voice"]}, indent=2))
        return
    freeze_receipt()
    output = args.output_dir.resolve()
    require(output.is_relative_to(ROOT) and not output.exists(),
            "output must be new and inside repository")
    output.mkdir(parents=True)
    manifest = render_all(output, conf, identity, metadata_hashes)
    path = output / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": manifest["status"], "manifest": str(path),
                      "manifest_sha256": sha(path), "pairs": 16, "wavs": 32}, indent=2))


if __name__ == "__main__":
    main()
