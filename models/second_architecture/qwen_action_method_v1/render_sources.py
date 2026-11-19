"""Render the frozen Qwen action-label speech sources; no Qwen model is imported."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from models.second_architecture import wav2vec2_fit_noise_norm_causal_host_v1 as host
from models.second_architecture.voicechat_acoustic_action_v1 import corpus as tts
from .endpoint import SPEC_PATH, load_spec
from .plan import audio_cases

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
TTS_SPEC = ROOT / "models/second_architecture/voicechat_acoustic_action_v1/spec.json"
TTS_SOURCE = ROOT / "models/second_architecture/voicechat_acoustic_action_v1/corpus.py"
PLAN_SOURCE = HERE / "plan.py"
RATE = 16000
FRAMES = RATE * 16
WALL_LIMIT_SECONDS = 900
EVIDENCE_LIMIT_BYTES = 200 * 1024**2
VOICE = {"piper_en_US_amy_medium": "piper_amy_en_us",
         "macos_samantha_en_US": "apple_samantha_en_us",
         "kokoro_af_bella": "kokoro_bella_en_us"}


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, obj: object) -> None:
    data = (json.dumps(obj, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with path.open("xb") as out:
        out.write(data)
        out.flush()
        os.fsync(out.fileno())


def expected_sources() -> list[dict]:
    spec = load_spec()
    rows = {row["id"]: row for row in spec["scripts"]}
    return [{**case, "text": rows[case["script_id"]]["text"]}
            for case in audio_cases(spec)]


def source_identity() -> dict:
    pinned = json.loads(TTS_SPEC.read_text())
    require(tts._sha_file(TTS_SPEC) == tts.SPEC_SHA256,
            "reused TTS specification changed")
    return {"qwen_spec_sha256": sha(SPEC_PATH),
            "qwen_source_plan_sha256": sha(PLAN_SOURCE),
            "render_source_sha256": sha(Path(__file__)),
            "voicechat_tts_spec_sha256": sha(TTS_SPEC),
            "voicechat_tts_source_sha256": sha(TTS_SOURCE),
            "tts_pins": pinned["tts"], "rate": RATE, "frames": FRAMES,
            "wall_limit_seconds": WALL_LIMIT_SECONDS,
            "evidence_limit_bytes": EVIDENCE_LIMIT_BYTES}


def _alarm(_signum: int, _frame: object) -> None:
    raise TimeoutError("bounded source render exceeded 900 seconds")


def _disk_bytes(folder: Path) -> int:
    return sum(path.stat().st_size for path in folder.rglob("*") if path.is_file())


def render(output: Path) -> dict:
    require(output.is_absolute() and not output.exists(),
            "render output must be a new absolute directory")
    rows = expected_sources()
    require(len(rows) == 18 and set(VOICE) == {r["voice"] for r in rows},
            "speech source population changed")
    identity = source_identity()
    output.mkdir(parents=True, exist_ok=False)
    start_time = datetime.now(timezone.utc)
    start_mono = time.monotonic()
    started = {"schema": "qwen_action_speech_render_v1",
               "status": "started_no_model_inference",
               "start_utc": start_time.isoformat(),
               "planned_cases": [r["id"] for r in rows],
               "identity": identity, "platform": platform.platform(),
               "python": sys.version.split()[0]}
    write_json(output / "started.json", started)
    active: str | None = None
    completed: list[dict] = []
    previous_alarm = signal.signal(signal.SIGALRM, _alarm)
    signal.setitimer(signal.ITIMER_REAL, WALL_LIMIT_SECONDS)
    try:
        pre = host.probe(run_start=start_time)
        host.archive(output, pre, "pre")
        write_json(output / "pre_host.json", pre)
        renderer = tts._RealTTS({"tts": identity["tts_pins"]})
        renderer_id = renderer.identity()
        for row in rows:
            require(time.monotonic() - start_mono < WALL_LIMIT_SECONDS,
                    "source render wall cap")
            active = row["id"]
            voice = VOICE[row["voice"]]
            raw = output / f"{active}.raw.{ 'aiff' if voice == 'apple_samantha_en_us' else 'wav' }"
            converted = output / f"{active}.converted.wav"
            final = output / f"{active}.final.wav"
            command = renderer(voice, row["text"], raw)
            pcm, numeric = tts._compute_converted(raw)
            numeric.pop("clean_frames")  # The reused converter's 5-s VoiceChat tail is not ours.
            require(0 < len(pcm) < FRAMES, f"source exceeds frozen 16-second limit: {active}")
            tts._write_pcm(converted, pcm)
            final_pcm = np.concatenate((pcm, np.zeros(FRAMES - len(pcm), dtype="<i2")))
            tts._write_pcm(final, final_pcm)
            require(np.array_equal(tts._read_pcm(converted), pcm)
                    and np.array_equal(tts._read_pcm(final), final_pcm)
                    and not bool(np.any(final_pcm[len(pcm):])),
                    "raw-to-final PCM reconstruction failed")
            completed.append({"case_id": active, "script_id": row["script_id"],
                              "split": row["split"], "voice": row["voice"],
                              "label": row["label"], "text": row["text"],
                              "command": command, "raw": raw.name,
                              "converted": converted.name, "final": final.name,
                              "raw_sha256": sha(raw), "converted_sha256": sha(converted),
                              "final_sha256": sha(final),
                              "final_pcm_sha256": hashlib.sha256(final_pcm.tobytes()).hexdigest(),
                              "final_frames": FRAMES, "source_frames": len(pcm),
                              **numeric})
            require(_disk_bytes(output) < EVIDENCE_LIMIT_BYTES,
                    "render evidence cap exceeded")
        post = host.probe(run_start=start_time)
        host.archive(output, post, "post")
        write_json(output / "post_host.json", post)
        elapsed = time.monotonic() - start_mono
        require(elapsed <= WALL_LIMIT_SECONDS and _disk_bytes(output) < EVIDENCE_LIMIT_BYTES,
                "render wall/evidence cap exceeded")
        result = {**started, "status": "rendered_pending_independent_byte_audit",
                  "completed_utc": datetime.now(timezone.utc).isoformat(),
                  "elapsed_seconds": elapsed, "renderer_identity": renderer_id,
                  "maxrss_self_raw": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  "maxrss_children_raw": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
                  "evidence_bytes_before_manifest": _disk_bytes(output),
                  "pre_host_sha256": sha(output / "pre_host.json"),
                  "post_host_sha256": sha(output / "post_host.json"),
                  "rows": completed, "candidate_model_inference_authorized": False}
        write_json(output / "manifest.json", result)
        return result
    except BaseException as error:
        write_json(output / "failure.json", {
            "status": "failed_render_no_model_inference", "active_case": active,
            "completed_cases": [r["case_id"] for r in completed],
            "error_type": type(error).__name__, "error": str(error),
            "failed_utc": datetime.now(timezone.utc).isoformat()})
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_alarm)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps({"output": str(args.output), "rows": len(render(args.output)["rows"])},
                     sort_keys=True))
