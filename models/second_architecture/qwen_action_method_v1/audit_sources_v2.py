"""V2 read-only source audit with terminal, source, host and copy gates."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from models.second_architecture import wav2vec2_fit_noise_norm_causal_host_v1 as host
from models.second_architecture.voicechat_acoustic_action_v1 import corpus as tts
from . import render_sources_v2 as source


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def audit(folder: Path) -> dict:
    require(folder.is_dir() and not folder.is_symlink()
            and folder.resolve() == source.ATTEMPT.resolve()
            and not (folder / "failure.json").exists(),
            "source directory invalid or failed attempt")
    go = source.validate_go()
    started = json.loads((folder / "started.json").read_text())
    manifest = json.loads((folder / "manifest.json").read_text())
    terminal = json.loads((folder / "terminal.json").read_text())
    identity = {**source.source_identity(), "reviewed_sources": go["sources"],
                "go_sha256": source.sha(source.GO_PATH),
                "review_sha256": source.sha(source.REVIEW_PATH)}
    expected = source.expected_sources()
    require(started.get("schema") == manifest.get("schema") == "qwen_action_speech_render_v2"
            and started.get("status") == "started_no_model_inference"
            and manifest.get("status") == "rendered_pending_terminal_deadline_proof"
            and started["identity"] == manifest["identity"] == identity
            and started["planned_cases"] == [r["id"] for r in expected]
            and manifest["planned_cases"] == started["planned_cases"]
            and manifest["candidate_model_inference_authorized"] is False
            and len(manifest["rows"]) == 18,
            "source start/manifest identity invalid")
    require(terminal.get("schema") == "qwen_action_speech_terminal_v2"
            and terminal.get("status") == "completed_pending_independent_byte_audit"
            and terminal.get("manifest_sha256") == source.sha(folder / "manifest.json")
            and terminal.get("started_sha256") == source.sha(folder / "started.json"),
            "post-manifest terminal proof invalid")
    require(manifest["pre_host_sha256"] == source.sha(folder / "pre_host.json")
            and manifest["post_host_sha256"] == source.sha(folder / "post_host.json"),
            "host record hash mismatch")
    pre = json.loads((folder / "pre_host.json").read_text())
    post = json.loads((folder / "post_host.json").read_text())
    start = datetime.fromisoformat(started["start_utc"])
    pre_time = datetime.fromisoformat(pre["collected_utc"])
    post_time = datetime.fromisoformat(post["collected_utc"])
    finished = datetime.fromisoformat(manifest["completed_utc"])
    terminal_time = datetime.fromisoformat(terminal["terminal_utc_after_manifest"])
    terminal_mtime = datetime.fromtimestamp((folder / "terminal.json").stat().st_mtime,
                                            tz=start.tzinfo)
    require(start <= pre_time <= post_time <= finished
            <= terminal_time <= terminal_mtime
            and terminal_mtime - start <= timedelta(seconds=source.WALL_LIMIT_SECONDS)
            and 0 < manifest["elapsed_seconds"]
            <= terminal["elapsed_seconds_after_manifest"] <= source.WALL_LIMIT_SECONDS,
            "render time/deadline invalid")
    host.validate_power(pre)
    host.validate_power(post)
    host.validate_log(host.load_archive(folder, pre, "pre"), start-timedelta(minutes=10),
                      pre_time, pre_time)
    host.validate_log(host.load_archive(folder, post, "post"), start, post_time, post_time)
    rss_path = folder / "rss_samples.jsonl"
    samples = [json.loads(line) for line in rss_path.read_text().splitlines()]
    require(samples and all(isinstance(s.get("tree_rss_bytes"), int)
                            and 0 < s["tree_rss_bytes"] <= source.TREE_RSS_LIMIT_BYTES
                            for s in samples)
            and len(samples) == manifest["rss_sample_count"]
            and max(s["tree_rss_bytes"] for s in samples)
                == manifest["sampled_peak_tree_rss_bytes"]
            and source.sha(rss_path) == manifest["rss_samples_sha256"],
            "RSS memory evidence invalid")
    seen: set[str] = set()
    for plan, row in zip(expected, manifest["rows"], strict=True):
        require(all(row.get(key) == value for key, value in plan.items() if key != "id")
                and row.get("case_id") == plan["id"]
                and row.get("final_frames") == source.FRAMES,
                "render case/word/voice changed")
        voice = source.VOICE[plan["voice"]]
        suffix = "aiff" if voice == "apple_samantha_en_us" else "wav"
        names = (f"{plan['id']}.raw.{suffix}", f"{plan['id']}.converted.wav",
                 f"{plan['id']}.final.wav")
        require(tuple(row[key] for key in ("raw", "converted", "final")) == names,
                "source paths differ from fixed pattern")
        raw, converted, final = [folder / name for name in names]
        require(all(path.is_file() and not path.is_symlink() for path in (raw, converted, final))
                and row["raw_sha256"] == source.sha(raw)
                and row["converted_sha256"] == source.sha(converted)
                and row["final_sha256"] == source.sha(final),
                "source file hash changed")
        seen.update(names)
        pcm, numeric = tts._compute_converted(raw)
        numeric.pop("clean_frames")  # Our 16-s final WAV supersedes VoiceChat's 5-s tail.
        require(0 < len(pcm) < source.FRAMES
                and all(row.get(key) == value for key, value in numeric.items())
                and row["source_frames"] == len(pcm),
                "raw conversion reconstruction changed")
        final_expected = np.concatenate((pcm, np.zeros(source.FRAMES-len(pcm), dtype="<i2")))
        require(np.array_equal(tts._read_pcm(converted), pcm)
                and np.array_equal(tts._read_pcm(final), final_expected)
                and row["final_pcm_sha256"] == hashlib.sha256(final_expected.tobytes()).hexdigest(),
                "final PCM differs from raw reconstruction")
    all_audio = {p.name for p in folder.iterdir() if p.suffix in (".wav", ".aiff")}
    require(all_audio == seen and source._disk_bytes(folder) < source.EVIDENCE_LIMIT_BYTES,
            "missing/extra audio or evidence cap exceeded")
    return {"schema": "qwen_action_speech_byte_audit_v2",
            "status": "technical_bytes_pass_human_review_pending",
            "cases": 18, "raw_and_converted_and_final_files": 54,
            "source_manifest_sha256": source.sha(folder / "manifest.json"),
            "all_audio_bytes": sum((folder / name).stat().st_size for name in seen),
            "human_reviews": 0, "candidate_model_inference_authorized": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source.write_json(args.output, audit(args.folder))
    print(args.output)
