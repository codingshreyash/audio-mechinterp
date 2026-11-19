"""V2 bounded source render with independently bound code; no Qwen model."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import signal
import subprocess
import sys
import threading
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
ATTEMPT = HERE / "runs/source_render_v2/attempt_001"
GO_PATH = HERE / "SPEECH_RENDER_GO_V2.json"
REVIEW_PATH = HERE / "INDEPENDENT_SPEECH_RENDER_SOURCE_REVIEW_GO_V2.md"
TTS_SPEC = ROOT / "models/second_architecture/voicechat_acoustic_action_v1/spec.json"
TTS_SOURCE = ROOT / "models/second_architecture/voicechat_acoustic_action_v1/corpus.py"
PLAN_SOURCE = HERE / "plan.py"
RATE = 16000
FRAMES = RATE * 16
WALL_LIMIT_SECONDS = 900
EVIDENCE_LIMIT_BYTES = 200 * 1024**2
TREE_RSS_LIMIT_BYTES = 8 * 1024**3
SOURCE_FILES = (
    "models/second_architecture/qwen_action_method_v1/frozen_spec_v1.json",
    "models/second_architecture/qwen_action_method_v1/endpoint.py",
    "models/second_architecture/qwen_action_method_v1/plan.py",
    "models/second_architecture/qwen_action_method_v1/render_sources.py",
    "models/second_architecture/qwen_action_method_v1/audit_sources.py",
    "models/second_architecture/qwen_action_method_v1/listening_bundle.py",
    "models/second_architecture/qwen_action_method_v1/render_sources_v2.py",
    "models/second_architecture/qwen_action_method_v1/audit_sources_v2.py",
    "models/second_architecture/qwen_action_method_v1/listening_bundle_v2.py",
    "models/second_architecture/qwen_action_method_v1/test_render_sources_v2.py",
    "models/second_architecture/qwen_action_method_v1/SPEECH_SOURCE_RENDER_HANDOFF_V2.md",
    "models/second_architecture/voicechat_acoustic_action_v1/spec.json",
    "models/second_architecture/voicechat_acoustic_action_v1/corpus.py",
    "models/second_architecture/wav2vec2_fit_noise_norm_causal_host_v1.py",
)
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


def reviewed_sources() -> dict[str, str]:
    """Exact committed blob bytes, never a self-repin from dirty local source."""
    result = {}
    for relative in SOURCE_FILES:
        path = ROOT / relative
        require(path.is_file() and not path.is_symlink(), f"source absent: {relative}")
        tracked = subprocess.run(["git", "-C", str(ROOT), "ls-files", "--error-unmatch",
                                  "--", relative], capture_output=True)
        working = subprocess.run(["git", "-C", str(ROOT), "diff", "--quiet",
                                  "--", relative])
        staged = subprocess.run(["git", "-C", str(ROOT), "diff", "--cached", "--quiet",
                                 "--", relative])
        require(tracked.returncode == working.returncode == staged.returncode == 0,
                f"source untracked or dirty: {relative}")
        result[relative] = sha(path)
    return result


def validate_go() -> dict:
    require(GO_PATH.is_file() and REVIEW_PATH.is_file()
            and not GO_PATH.is_symlink() and not REVIEW_PATH.is_symlink(),
            "independent V2 render GO missing")
    for path in (GO_PATH, REVIEW_PATH):
        relative = str(path.relative_to(ROOT))
        require(subprocess.run(["git", "-C", str(ROOT), "ls-files", "--error-unmatch",
                                "--", relative], capture_output=True).returncode == 0
                and subprocess.run(["git", "-C", str(ROOT), "diff", "--quiet",
                                    "--", relative]).returncode == 0
                and subprocess.run(["git", "-C", str(ROOT), "diff", "--cached", "--quiet",
                                    "--", relative]).returncode == 0,
                "independent GO/review is not committed exact")
    receipt = json.loads(GO_PATH.read_text())
    require(receipt.get("schema") == "qwen_action_speech_render_go_v2"
            and receipt.get("status") == "independent_executable_go"
            and receipt.get("sources") == reviewed_sources()
            and receipt.get("review_sha256") == sha(REVIEW_PATH)
            and "**Decision: GO" in REVIEW_PATH.read_text(),
            "independent V2 render GO does not bind current source")
    return receipt


def _tree_rss_bytes(root_pid: int) -> int:
    proc = subprocess.run(["ps", "-axo", "pid=,ppid=,rss="], capture_output=True,
                          text=True, check=True, timeout=5)
    rows = []
    for line in proc.stdout.splitlines():
        fields = line.split()
        if len(fields) == 3 and all(x.isdigit() for x in fields):
            rows.append(tuple(int(x) for x in fields))
    children: dict[int, list[int]] = {}
    rss = {}
    for pid, ppid, kib in rows:
        children.setdefault(ppid, []).append(pid)
        rss[pid] = kib * 1024
    seen = set()
    stack = [root_pid]
    while stack:
        pid = stack.pop()
        if pid not in seen:
            seen.add(pid)
            stack.extend(children.get(pid, ()))
    return sum(rss.get(pid, 0) for pid in seen)


class RssMonitor:
    def __init__(self, output: Path):
        self.output = output
        self.stop = threading.Event()
        self.samples: list[dict] = []
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self) -> None:
        while not self.stop.is_set():
            try:
                value = _tree_rss_bytes(os.getpid())
                self.samples.append({"utc": datetime.now(timezone.utc).isoformat(),
                                     "tree_rss_bytes": value})
                if value > TREE_RSS_LIMIT_BYTES:
                    os.kill(os.getpid(), signal.SIGUSR1)
                    return
            except BaseException as error:
                self.samples.append({"monitor_error": repr(error)})
                os.kill(os.getpid(), signal.SIGUSR1)
                return
            self.stop.wait(0.25)

    def start(self) -> None:
        self.thread.start()

    def finish(self) -> None:
        self.stop.set()
        self.thread.join(timeout=5)
        require(not self.thread.is_alive(), "RSS monitor did not stop")
        with (self.output / "rss_samples.jsonl").open("xb") as sink:
            for sample in self.samples:
                sink.write((json.dumps(sample,sort_keys=True)+"\n").encode())
            sink.flush()
            os.fsync(sink.fileno())
        require(self.samples and all("tree_rss_bytes" in s for s in self.samples)
                and max(s["tree_rss_bytes"] for s in self.samples) <= TREE_RSS_LIMIT_BYTES,
                "RSS sampler failed or limit exceeded")


def _memory_alarm(_signum: int, _frame: object) -> None:
    raise MemoryError("TTS process tree RSS exceeded 8 GiB or monitor failed")


def _alarm(_signum: int, _frame: object) -> None:
    raise TimeoutError("bounded source render exceeded 900 seconds")


def _disk_bytes(folder: Path) -> int:
    return sum(path.stat().st_size for path in folder.rglob("*") if path.is_file())


def render(output: Path) -> dict:
    require(output.is_absolute() and output.resolve() == ATTEMPT.resolve()
            and not output.exists(),
            "render output must be the new fixed V2 attempt")
    go = validate_go()
    rows = expected_sources()
    require(len(rows) == 18 and set(VOICE) == {r["voice"] for r in rows},
            "speech source population changed")
    identity = {**source_identity(), "reviewed_sources": go["sources"],
                "go_sha256": sha(GO_PATH), "review_sha256": sha(REVIEW_PATH)}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=False)
    start_time = datetime.now(timezone.utc)
    start_mono = time.monotonic()
    started = {"schema": "qwen_action_speech_render_v2",
               "status": "started_no_model_inference",
               "start_utc": start_time.isoformat(),
               "planned_cases": [r["id"] for r in rows],
               "identity": identity, "platform": platform.platform(),
               "python": sys.version.split()[0]}
    write_json(output / "started.json", started)
    active: str | None = None
    completed: list[dict] = []
    previous_alarm = signal.signal(signal.SIGALRM, _alarm)
    previous_memory = signal.signal(signal.SIGUSR1, _memory_alarm)
    signal.setitimer(signal.ITIMER_REAL, WALL_LIMIT_SECONDS)
    monitor: RssMonitor | None = None
    try:
        pre = host.probe(run_start=start_time)
        host.archive(output, pre, "pre")
        write_json(output / "pre_host.json", pre)
        renderer = tts._RealTTS({"tts": identity["tts_pins"]})
        renderer_id = renderer.identity()
        monitor = RssMonitor(output)
        monitor.start()
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
        monitor.finish()
        host.archive(output, post, "post")
        write_json(output / "post_host.json", post)
        elapsed = time.monotonic() - start_mono
        require(elapsed <= WALL_LIMIT_SECONDS and _disk_bytes(output) < EVIDENCE_LIMIT_BYTES,
                "render wall/evidence cap exceeded")
        result = {**started, "status": "rendered_pending_terminal_deadline_proof",
                  "completed_utc": datetime.now(timezone.utc).isoformat(),
                  "elapsed_seconds": elapsed, "renderer_identity": renderer_id,
                  "maxrss_self_raw": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  "maxrss_children_raw": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
                  "evidence_bytes_before_manifest": _disk_bytes(output),
                  "rss_samples_sha256": sha(output / "rss_samples.jsonl"),
                  "sampled_peak_tree_rss_bytes": max(s["tree_rss_bytes"] for s in monitor.samples),
                  "rss_sample_count": len(monitor.samples),
                  "pre_host_sha256": sha(output / "pre_host.json"),
                  "post_host_sha256": sha(output / "post_host.json"),
                  "rows": completed, "candidate_model_inference_authorized": False}
        write_json(output / "manifest.json", result)
        after_manifest = time.monotonic() - start_mono
        require(after_manifest <= WALL_LIMIT_SECONDS - 2,
                "no time reserve for terminal proof")
        terminal = {"schema": "qwen_action_speech_terminal_v2",
                    "status": "completed_pending_independent_byte_audit",
                    "manifest_sha256": sha(output / "manifest.json"),
                    "started_sha256": sha(output / "started.json"),
                    "terminal_utc_after_manifest": datetime.now(timezone.utc).isoformat(),
                    "elapsed_seconds_after_manifest": after_manifest}
        write_json(output / "terminal.json", terminal)
        after_terminal = time.monotonic() - start_mono
        if after_terminal > WALL_LIMIT_SECONDS or _disk_bytes(output) >= EVIDENCE_LIMIT_BYTES:
            (output / "terminal.json").unlink()
            raise TimeoutError("post-terminal wall/evidence cap exceeded")
        return result
    except BaseException as error:
        if monitor is not None and monitor.thread.is_alive():
            monitor.stop.set()
            monitor.thread.join(timeout=5)
        if (output / "terminal.json").exists():
            (output / "terminal.json").unlink()
        write_json(output / "failure.json", {
            "status": "failed_render_no_model_inference", "active_case": active,
            "completed_cases": [r["case_id"] for r in completed],
            "error_type": type(error).__name__, "error": str(error),
            "failed_utc": datetime.now(timezone.utc).isoformat()})
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_alarm)
        signal.signal(signal.SIGUSR1, previous_memory)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps({"output": str(args.output), "rows": len(render(args.output)["rows"])},
                     sort_keys=True))
