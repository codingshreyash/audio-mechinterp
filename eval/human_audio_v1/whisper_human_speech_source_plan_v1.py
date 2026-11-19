"""Build the prospective human-speech Whisper source plan without model inference."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MANIFEST = HERE / "source_v1/manifest.json"
PROTOCOL = HERE / "WHISPER_HUMAN_SPEECH_MECHANISM_PROSPECTIVE_V1.md"
PROXY_REVIEW = HERE / "ASR_PROXY_INDEPENDENT_RESULT_V1.md"
PROXY_AUDIT = HERE / "runs/asr_proxy_v1/attempt_001_audit_v2.json"
PINS = ROOT / "models/second_architecture/nonspeech_probe_v1/descriptive_pins_v1.json"
OUTPUT = HERE / "WHISPER_HUMAN_SPEECH_SOURCE_PLAN_V1.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def source_rows(utterance: str, boundary: int) -> list[int]:
    candidates = range(8, boundary - 7)
    rank = lambda r: hashlib.sha256(
        f"whisper-human-speech-v1:{utterance}:displaced:{r}".encode()
    ).digest()
    return sorted(sorted(candidates, key=lambda r: (rank(r), r))[:14])


def build() -> dict:
    manifest = json.loads(MANIFEST.read_text())
    pins = json.loads(PINS.read_text())
    require(manifest["schema"] == "human_audio_source_v1" and len(manifest["rows"]) == 4,
            "wrong human source manifest")
    model = pins["models"]["whisper_multilingual"]
    require(model["repo"] == "openai/whisper-tiny" and model["revision"] ==
            "169d4a4341b33bc18d8881c4b69c2e104e1cc0af", "target pin changed")
    require(sha(PROXY_AUDIT) ==
            "119491174d4740d14c1ee4a6c8cafa16ee14b32a828e43d13fb625fde22ce6e5",
            "independently reviewed proxy receipt changed")
    expected = [("3081", "development"), ("652", "development"),
                ("6241", "sealed_test"), ("251", "sealed_test")]
    calls, comparisons = [], []
    for row, (speaker, split) in zip(manifest["rows"], expected):
        require((row["speaker_id"], row["split"]) == (speaker, split), "speaker split changed")
        require(row["sample_rate"] == 16000 and row["active_frames"] >= 32000,
                "source geometry changed")
        by = {c["condition"]: c for c in row["conditions"]}
        require(set(by) == {"base", "z32000", "n32000_s0", "n32000_s1"},
                "four-condition census changed")
        for condition in ("base", "z32000", "n32000_s0", "n32000_s1"):
            item = by[condition]
            wav = HERE / "source_v1" / item["wav"]
            require(sha(wav) == item["wav_sha256"], f"source WAV changed: {wav.name}")
            calls.append({"speaker_id": speaker, "split": split,
                          "utterance_id": row["utterance_id"], "condition": condition,
                          "wav": str(wav.relative_to(ROOT)), "wav_sha256": item["wav_sha256"],
                          "frames": item["frames"]})
        n = row["active_frames"]
        b, e = (n + 160) // 320, (n + 32160) // 320
        boundary_rows = list(range(b - 3, b + 4)) + list(range(e - 3, e + 4))
        displaced_rows = source_rows(row["utterance_id"], b)
        require(len(boundary_rows) == len(displaced_rows) == 14 and
                not (set(boundary_rows) & set(displaced_rows)) and e + 3 < 1500,
                "causal row geometry changed")
        for seed in (0, 1):
            noise = by[f"n32000_s{seed}"]
            zero = by["z32000"]
            require(noise["frames"] == zero["frames"] == n + 32000,
                    "equal-duration comparison changed")
            comparisons.append({"speaker_id": speaker, "split": split,
                "utterance_id": row["utterance_id"], "seed": seed,
                "source_frames": n, "tail_frames": 32000,
                "zero_wav_sha256": zero["wav_sha256"],
                "noise_wav_sha256": noise["wav_sha256"],
                "source_tail_encoder_center": b,
                "tail_padding_encoder_center": e,
                "boundary_rows": boundary_rows,
                "displaced_source_rows": displaced_rows,
                "human_screen": "pending_real_two_listener_records",
                "target_result": "sealed_no_inference"})
    require(len(calls) == 16 and len(comparisons) == 8 and
            sum(c["split"] == "development" for c in comparisons) == 4,
            "source denominator changed")
    return {"schema": "whisper_human_speech_source_plan_v1",
            "status": "source_only_hold_no_target_inference",
            "bindings": {"manifest_sha256": sha(MANIFEST),
                         "protocol_sha256": sha(PROTOCOL),
                         "proxy_review_sha256": sha(PROXY_REVIEW),
                         "proxy_audit_sha256": sha(PROXY_AUDIT),
                         "pins_sha256": sha(PINS)},
            "target": {"repo": model["repo"], "revision": model["revision"],
                       "weight_sha256": model["weight_sha256"]},
            "counts": {"speakers": 4, "wav_calls": 16,
                       "base_replays": 4, "primary_comparisons": 8,
                       "development_comparisons": 4, "sealed_test_comparisons": 4},
            "calls": calls, "comparisons": comparisons,
            "causal_sites": ["post_conv_pos", "post_block0"],
            "causal_directions": ["zero_to_noise", "noise_to_zero"],
            "causal_arms_per_site_direction": ["boundary_donor", "identity_self",
                "displaced_source_donor", "displaced_equal_norm_delta", "full_donor"],
            "gate": "two-listener blinded QA, source-only admission, independent GO"}


def main() -> None:
    require(not OUTPUT.exists(), "frozen plan already exists")
    OUTPUT.write_text(json.dumps(build(), sort_keys=True, separators=(",", ":")) + "\n")
    print(json.dumps({"status": "source_only_hold_no_target_inference",
                      "path": str(OUTPUT), "sha256": sha(OUTPUT)}))


if __name__ == "__main__":
    main()
