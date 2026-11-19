"""Fake-only corpus tests; synthetic reviewer entries never leave temp folders."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
import wave

import numpy as np

from models.second_architecture.voicechat_acoustic_action_v1 import corpus


def fake_renderer(voice_id: str, script: str, raw: Path) -> list[str]:
    # This is deliberately a recognizably fake tone. It is never human QA.
    frames = np.arange(16000, dtype=np.float64)
    samples = np.rint(8000 * np.sin(2 * np.pi * 330 * frames / 16000)).astype("<i2")
    with wave.open(str(raw), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(samples.tobytes())
    return ["FAKE_TONE", voice_id, script]


def filled_fake_annotations(template: dict) -> dict:
    filled = json.loads(json.dumps(template))
    for record in filled["records"]:
        record["annotator_a"] = {"reviewer_id": "fake_a", "decision": [1200, 2000],
                                 "displaced": [6000, 6800]}
        record["annotator_b"] = {"reviewer_id": "fake_b", "decision": [1200, 2000],
                                 "displaced": [6000, 6800]}
        record["adjudication"] = {"by": "fake_consensus", "decision": [1200, 2000],
                                  "displaced": [6000, 6800], "note": "Synthetic test only"}
    return filled


def forge_fake_source_identity(source: Path, *, change_script: bool) -> Path:
    """Adversarial local edit used only to test production admission rejection."""
    manifest_path = source / "source_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    pins = corpus._spec()["tts"]
    identity = {
        "piper_package_version": "1.8.0", "kokoro_package_version": "0.6.1",
        "onnxruntime_version": "fake", "piper_voice": pins["piper"]["voice"],
        "kokoro_voice": pins["kokoro"]["voice"],
        "samantha_voice_line": pins["samantha"]["voice_line"],
        "piper_model_sha256": pins["piper"]["model_sha256"],
        "piper_config_sha256": pins["piper"]["config_sha256"],
        "kokoro_model_sha256": pins["kokoro"]["model_sha256"],
        "kokoro_voices_sha256": pins["kokoro"]["voices_sha256"],
        "say_sha256": pins["samantha"]["command_sha256"],
        "os_build": pins["samantha"]["os_build"],
    }
    manifest["renderer_identity"] = identity
    if change_script:
        manifest["rows"][0]["script"] = "A totally unrelated sentence."
    else:
        for row in manifest["rows"]:
            text = row["script"]
            raw = source / row["raw_path"]
            if row["voice_id"] == "piper_amy_en_us":
                command = ["PiperVoice.synthesize_wav", pins["piper"]["voice"], text]
            elif row["voice_id"] == "kokoro_bella_en_us":
                command = ["Kokoro.create", pins["kokoro"]["voice"], text]
            else:
                command = [pins["samantha"]["command"], "-v", "Samantha", "-o", str(raw), text]
            row["render_command"] = command
    manifest_path.write_text(json.dumps(manifest))
    started = dict(manifest)
    started["status"] = "started_no_model_inference"
    started["rows"] = []
    (source / "started.json").write_text(json.dumps(started))
    template = corpus.annotation_template(manifest, corpus._sha_file(manifest_path))
    annotations = source / "forged_annotations.json"
    annotations.write_text(json.dumps(filled_fake_annotations(template)))
    return annotations


class CorpusSourceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def render(self) -> Path:
        folder = self.root / "source"
        result = corpus.render_sources(folder, renderer=fake_renderer,
                                       renderer_identity={"synthetic_test_only": True})
        self.assertEqual(result["status"], "complete_sources_no_human_intervals")
        return folder

    def annotations(self, source: Path) -> Path:
        template = json.loads((source / "annotation_template.json").read_text())
        path = self.root / "fake_annotations.json"
        path.write_text(json.dumps(filled_fake_annotations(template)))
        return path

    def test_stage1_exact_population_tail_and_blank_template(self) -> None:
        source = self.render()
        result = json.loads((source / "source_manifest.json").read_text())
        self.assertEqual(len(result["rows"]), 12)
        self.assertEqual(len({row["source_id"] for row in result["rows"]}), 12)
        self.assertEqual({row["split"] for row in result["rows"]},
                         {"development", "locked"})
        self.assertEqual(sum(row["split"] == "locked" for row in result["rows"]), 8)
        template = json.loads((source / "annotation_template.json").read_text())
        self.assertEqual(len(template["records"]), 12)
        self.assertIsNone(template["records"][0]["annotator_a"]["decision"])
        for row in result["rows"]:
            audio = corpus._read_pcm(source / row["clean_path"])
            self.assertEqual(len(audio), row["converted_frames"] + 80000)
            self.assertEqual(int(np.count_nonzero(audio[-80000:])), 0)
            self.assertLessEqual(row["post_gain_peak"], 0.8500001)
            self.assertGreater(row["level_match_gain"], 0)
        with self.assertRaises(ValueError):
            corpus.render_sources(source, renderer=fake_renderer,
                                  renderer_identity={"synthetic_test_only": True})

    def test_stage2_requires_two_real_fields_and_preserves_failure(self) -> None:
        source = self.render()
        absent = self.root / "absent.json"
        with self.assertRaisesRegex(ValueError, "annotation file absent"):
            corpus.make_variants(source, absent, self.root / "failed_missing",
                                 allow_synthetic_test_only=True)
        self.assertTrue((self.root / "failed_missing" / "failure.json").is_file())
        path = self.annotations(source)
        data = json.loads(path.read_text())
        data["records"][0]["annotator_b"]["reviewer_id"] = "fake_a"
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "distinct IDs"):
            corpus.make_variants(source, path, self.root / "failed_duplicate",
                                 allow_synthetic_test_only=True)
        self.assertTrue((self.root / "failed_duplicate" / "failure.json").is_file())

    def test_stage2_exact_48_deterministic_arms_and_masks(self) -> None:
        source = self.render()
        annotations = self.annotations(source)
        with self.assertRaisesRegex(ValueError, "Synthetic test renderer"):
            corpus.make_variants(source, annotations, self.root / "public_hold")
        first = corpus.make_variants(source, annotations, self.root / "v1",
                                     allow_synthetic_test_only=True)
        second = corpus.make_variants(source, annotations, self.root / "v2",
                                      allow_synthetic_test_only=True)
        self.assertEqual(len(first["rows"]), 48)
        self.assertEqual([(r["source_id"], r["arm"], r["wav_sha256"]) for r in first["rows"]],
                         [(r["source_id"], r["arm"], r["wav_sha256"]) for r in second["rows"]])
        self.assertEqual({r["arm"] for r in first["rows"]}, set(corpus.ARMS))
        by_key = {(r["source_id"], r["arm"]): r for r in first["rows"]}
        for source_id in {r["source_id"] for r in first["rows"]}:
            clean = corpus._read_pcm(self.root / "v1" / by_key[source_id, "clean"]["wav_path"])
            full = corpus._read_pcm(self.root / "v1" / by_key[source_id, "full_noise"]["wav_path"])
            decision = corpus._read_pcm(self.root / "v1" / by_key[source_id, "decision_noise"]["wav_path"])
            displaced = corpus._read_pcm(self.root / "v1" / by_key[source_id, "displaced_noise"]["wav_path"])
            self.assertTrue(np.array_equal(clean[-80000:], full[-80000:]))
            self.assertTrue(np.array_equal(clean[-80000:], decision[-80000:]))
            self.assertTrue(np.array_equal(clean[-80000:], displaced[-80000:]))
            self.assertTrue(np.array_equal(clean[:1200], decision[:1200]))
            self.assertTrue(np.array_equal(clean[2000:], decision[2000:]))
            self.assertTrue(np.array_equal(clean[:6000], displaced[:6000]))
            self.assertTrue(np.array_equal(clean[6800:], displaced[6800:]))
            self.assertTrue(np.array_equal(full[1200:2000], decision[1200:2000]))
            self.assertTrue(np.array_equal(full[6000:6800], displaced[6000:6800]))
        self.assertFalse(first["candidate_inference_authorized"])
        self.assertTrue(first["synthetic_test_only"])

    def test_noise_clipping_fails_before_writing_arm(self) -> None:
        clean = np.zeros(10000 + 80000, dtype="<i2")
        clean[:10000] = 32767
        clean[-80000:] = 0
        noise = np.ones(10000, dtype=np.float64) * 100
        with self.assertRaisesRegex(ValueError, "would clip"):
            corpus._noisy_arm(clean, 10000, noise, 0.5, (0, 10000))

    def test_uniform_level_rule_handles_float_source_over_one(self) -> None:
        frames = np.arange(24000, dtype=np.float64)
        floating = 1.2 * np.sin(2 * np.pi * 330 * frames / 24000)
        raw = self.root / "over_one.raw.wav"
        converted = self.root / "over_one.converted.wav"
        clean = self.root / "over_one.clean.wav"
        corpus._raw_float_wav(raw, floating, 24000)
        result = corpus._convert_raw(raw, converted, clean)
        self.assertGreater(result["pre_gain_peak"], 1)
        self.assertLessEqual(result["post_gain_peak"], 0.8500001)
        self.assertEqual(int(np.count_nonzero(corpus._read_pcm(clean)[-80000:])), 0)

    def test_different_proposals_require_third_adjudicator(self) -> None:
        source = self.render()
        path = self.annotations(source)
        data = json.loads(path.read_text())
        data["records"][0]["annotator_b"]["decision"] = [1201, 2001]
        data["records"][0]["adjudication"]["by"] = "fake_a"
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "distinct third adjudicator"):
            corpus.make_variants(source, path, self.root / "failed_disagreement",
                                 allow_synthetic_test_only=True)

    def test_agreed_proposals_cannot_drift_to_coordinator_window(self) -> None:
        source = self.render()
        path = self.annotations(source)
        data = json.loads(path.read_text())
        record = data["records"][0]
        record["adjudication"]["decision"] = [3000, 3800]
        record["adjudication"]["displaced"] = [7000, 7800]
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "distinct third adjudicator"):
            corpus.make_variants(source, path, self.root / "failed_consensus_drift",
                                 allow_synthetic_test_only=True)
        record["third_annotation"] = {"reviewer_id": "fake_third",
                                      "decision": [3000, 3800], "displaced": [7000, 7800]}
        record["adjudication"]["by"] = "fake_third"
        path.write_text(json.dumps(data))
        result = corpus.make_variants(source, path, self.root / "third_ok",
                                      allow_synthetic_test_only=True)
        self.assertEqual(len(result["rows"]), 48)

    def test_changed_a1_script_and_fake_identity_cannot_self_repin(self) -> None:
        source = self.render()
        annotations = forge_fake_source_identity(source, change_script=True)
        with self.assertRaisesRegex(ValueError, "frozen script"):
            corpus.make_variants(source, annotations, self.root / "forged_script")
        self.assertFalse((self.root / "forged_script" / "variant_manifest.json").exists())

        other = self.root / "other_source"
        corpus.render_sources(other, renderer=fake_renderer,
                              renderer_identity={"synthetic_test_only": True})
        annotations = forge_fake_source_identity(other, change_script=False)
        with self.assertRaisesRegex(ValueError, "Independent admission file absent"):
            corpus.make_variants(other, annotations, self.root / "forged_identity")
        self.assertFalse((self.root / "forged_identity" / "variant_manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
