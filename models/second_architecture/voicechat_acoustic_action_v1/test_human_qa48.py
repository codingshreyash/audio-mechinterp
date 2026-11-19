"""Fake-only QA tests. No human judgment or VoiceChat inference."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from . import human_qa48 as qa

STAMP = "2026-11-19T07:00:00Z"
LATE = "2026-11-19T08:00:00Z"
LATER = "2026-11-19T08:30:00Z"


def fake_variants(root: Path) -> dict:
    spec, source_plan = qa._plan()
    rows = []
    for source in source_plan:
        sid = source["script_id"] + "__" + source["voice_id"]
        for arm in qa.ARMS:
            filename = sid + "__" + arm + ".wav"
            payload = ("FAKE-WAV|" + filename).encode()
            (root / filename).write_bytes(payload)
            rows.append({**source, "source_id": sid, "arm": arm,
                         "wav_path": filename, "wav_sha256": qa.digest(payload)})
    manifest = {
        "schema": "voicechat_acoustic_action_variants_v1",
        "status": "complete_48_wavs_no_human_qa_or_model_inference",
        "spec_sha256": qa.SPEC_SHA256,
        "planned_sources": 12, "planned_arms": 48,
        "synthetic_test_only": True,
        "candidate_inference_authorized": False, "rows": rows,
    }
    (root / "variant_manifest.json").write_bytes(qa.canonical(manifest) + b"\n")
    return manifest


def first_export(queue: dict, reviewer: str) -> dict:
    return {
        "schema": "voicechat_acoustic_qa48_first_v1",
        "reviewer_id": reviewer, "queue_sha256": qa.digest(qa.canonical(queue)),
        "role": queue["role"], "started_at_utc": STAMP, "exported_at_utc": STAMP,
        "records": [{
            "opaque_id": clip["opaque_id"], "wav_sha256": clip["wav_sha256"],
            "reviewer_id": reviewer, "submitted_at_utc": STAMP,
            "verbatim_transcript": "Fake first words", "uncertain_spans": "none",
            "date_heard": "November nineteenth", "event_heard": "fake event",
            "authorization_heard": "fake authorization", "clarity": "clear",
            "confidence": 4,
        } for clip in queue["clips"]],
    }


def comparison(private: dict, queue: dict, first: dict) -> dict:
    result = qa.comparison_template(private, queue, first)
    for record in result["records"]:
        for key in ("date_match", "event_match", "authorization_match",
                    "same_critical_meaning_as_clean"):
            record[key] = "yes"
        record["compared_at_utc"] = STAMP
    result["exported_at_utc"] = STAMP
    return result


def asr(private: dict) -> dict:
    return {"schema": "voicechat_acoustic_qa48_asr_raw_v1",
            "records": [{
                "opaque_id": case["opaque_id"], "wav_sha256": case["wav_sha256"],
                "backend": backend, "raw_transcript": "synthetic diagnostic",
                "status": "complete",
            } for case in private["cases"] for backend in ("whisper", "wav2vec2")]}


class HumanQA48Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.variants = self.root / "variants"
        self.variants.mkdir()
        self.manifest = fake_variants(self.variants)
        self.private, self.queues = qa.make_private_and_queues(
            self.manifest, qa.sha(self.variants / "variant_manifest.json"))
        self.first = {role: first_export(self.queues[role], "person_" + role)
                      for role in ("primary", "secondary")}
        self.compare = {role: comparison(self.private, self.queues[role],
                                         self.first[role])
                        for role in ("primary", "secondary")}

    def test_exact_48_population_and_blind_render(self) -> None:
        self.assertEqual(qa.validate_variants(self.variants, synthetic_test=True),
                         self.manifest)
        self.assertEqual(len(self.private["cases"]), 48)
        self.assertNotEqual([r["opaque_id"] for r in self.queues["primary"]["clips"]],
                            [r["opaque_id"] for r in self.queues["secondary"]["clips"]])
        folder = self.root / "blind"
        qa.build_fake_blind_bundle(self.manifest, self.variants, folder)
        qa.verify_blind_bundle(folder, self.private, self.queues)
        page = (folder / "primary" / "FIRST_FORM.html").read_text()
        self.assertNotIn("For November nineteenth", page)
        self.assertNotIn("Design Review", page)
        self.assertEqual(len(list((folder / "primary").glob("*.wav"))), 48)
        with self.assertRaisesRegex(ValueError, "production QA HOLD"):
            qa.validate_variants(self.variants)
        (folder / "primary" / "FIRST_FORM.html").write_text("<script>forged</script>")
        with self.assertRaisesRegex(ValueError, "rendered form bytes differ"):
            qa.verify_blind_bundle(folder, self.private, self.queues)
        (folder / "primary" / "FIRST_FORM.html").write_bytes(
            qa.render_first_form(self.queues["primary"]))
        (folder / "secondary" / "unexpected.txt").write_text("extra")
        with self.assertRaisesRegex(ValueError, "inventory differs"):
            qa.verify_blind_bundle(folder, self.private, self.queues)
        (folder / "secondary" / "unexpected.txt").unlink()
        changed_private = json.loads(json.dumps(self.private))
        changed_private["cases"][0]["reference_script"] = "changed"
        with self.assertRaisesRegex(ValueError, "reference differs"):
            qa.validate_private_queues(changed_private, self.queues)

    def test_post_blind_reveal_requires_both_locked_exports(self) -> None:
        with self.assertRaisesRegex(ValueError, "two exports"):
            qa.build_post_blind_forms(self.private, self.queues,
                                      {"primary": self.first["primary"]},
                                      self.root / "one")
        folder = self.root / "post"
        qa.build_post_blind_forms(self.private, self.queues, self.first, folder)
        self.assertIn("Design Review", (folder / "primary_COMPARE_FORM.html").read_text())
        self.assertEqual(len(list(folder.iterdir())), 2)

    def test_all_48_reviewed_source_eligibility_and_exact_receipt(self) -> None:
        report = qa.evaluate(self.private, self.queues, self.first, self.compare,
                             asr(self.private))
        self.assertEqual(report["reviewed_arms"], 48)
        self.assertEqual(report["eligible_sources"], 12)
        self.assertTrue(report["locked_ab_all_eligible"])
        self.assertFalse(report["candidate_inference_authorized"])
        receipt = self.root / "fake_receipt.json"
        self.assertEqual(qa.write_synthetic_receipt(receipt, report), qa.sha(receipt))
        with self.assertRaisesRegex(ValueError, "already used"):
            qa.write_synthetic_receipt(receipt, report)

    def test_failed_arm_keeps_source_and_locked_denominator(self) -> None:
        ident = next(c["opaque_id"] for c in self.private["cases"]
                     if c["source_id"] == "A2__apple_samantha_en_us"
                     and c["arm"] == "full_noise")
        for record in self.first["primary"]["records"]:
            if record["opaque_id"] == ident:
                record["clarity"] = "unclear"
        for record in self.first["secondary"]["records"]:
            if record["opaque_id"] == ident:
                record["clarity"] = "unclear"
        self.compare = {role: comparison(self.private, self.queues[role],
                                         self.first[role])
                        for role in ("primary", "secondary")}
        report = qa.evaluate(self.private, self.queues, self.first, self.compare,
                             asr(self.private))
        self.assertEqual(report["eligible_sources"], 11)
        self.assertFalse(report["locked_ab_all_eligible"])
        self.assertEqual(len(report["case_rows"]), 48)
        self.assertEqual(len(report["sources"]), 12)

    def test_critical_disagreement_requires_independent_third(self) -> None:
        ident = self.private["cases"][0]["opaque_id"]
        for record in self.first["secondary"]["records"]:
            if record["opaque_id"] == ident:
                record["clarity"] = "unclear"
        self.compare["secondary"] = comparison(
            self.private, self.queues["secondary"], self.first["secondary"])
        with self.assertRaisesRegex(ValueError, "third review required"):
            qa.evaluate(self.private, self.queues, self.first, self.compare,
                        asr(self.private))
        third_queue = qa.third_queue_for_flags(self.private, [ident])
        blind = self.root / "blind"
        qa.build_fake_blind_bundle(self.manifest, self.variants, blind)
        third_bundle = self.root / "third"
        self.assertEqual(
            qa.build_third_blind_bundle(
                self.private, self.queues, self.first, self.compare,
                blind / "primary", third_bundle), third_queue)
        self.assertEqual(len(list(third_bundle.glob("*.wav"))), 1)
        self.assertNotIn("For November nineteenth",
                         (third_bundle / "FIRST_FORM.html").read_text())
        third = first_export(third_queue, "person_third")
        third_compare = comparison(self.private, third_queue, third)
        qa.build_third_compare_form(self.private, third_queue, third,
                                    self.root / "third_COMPARE_FORM.html")
        self.assertTrue((self.root / "third_COMPARE_FORM.html").is_file())
        report = qa.evaluate(self.private, self.queues, self.first, self.compare,
                             asr(self.private), tertiary_first=third,
                             tertiary_compare=third_compare)
        self.assertEqual(report["critical_disagreement_arms"], 1)
        self.assertTrue(report["third_reviewed"])
        self.assertEqual(report["eligible_sources"], 11)

    def test_export_and_asr_tamper_rejected(self) -> None:
        self.first["primary"]["records"][0]["wav_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "clip/reviewer binding"):
            qa.evaluate(self.private, self.queues, self.first, self.compare,
                        asr(self.private))
        self.first["primary"] = first_export(self.queues["primary"], "person_primary")
        diagnostics = asr(self.private)
        diagnostics["records"].pop()
        with self.assertRaisesRegex(ValueError, "two raw ASR"):
            qa.evaluate(self.private, self.queues, self.first, self.compare,
                        diagnostics)

    def test_both_first_exports_must_precede_any_reference_comparison(self) -> None:
        secondary = self.first["secondary"]
        secondary["started_at_utc"] = LATE
        secondary["exported_at_utc"] = LATE
        for record in secondary["records"]:
            record["submitted_at_utc"] = LATE
        self.compare["secondary"] = comparison(
            self.private, self.queues["secondary"], secondary)
        for record in self.compare["secondary"]["records"]:
            record["compared_at_utc"] = LATE
        self.compare["secondary"]["exported_at_utc"] = LATE
        with self.assertRaisesRegex(ValueError, "preceded both locked first exports"):
            qa.evaluate(self.private, self.queues, self.first, self.compare,
                        asr(self.private))
        for record in self.compare["primary"]["records"]:
            record["compared_at_utc"] = LATE
        self.compare["primary"]["exported_at_utc"] = LATE
        report = qa.evaluate(self.private, self.queues, self.first, self.compare,
                             asr(self.private))
        self.assertEqual(report["reviewed_arms"], 48)

    def test_sparse_third_cannot_start_before_flags_are_final(self) -> None:
        ident = self.private["cases"][0]["opaque_id"]
        for record in self.first["secondary"]["records"]:
            if record["opaque_id"] == ident:
                record["clarity"] = "unclear"
        self.compare["secondary"] = comparison(
            self.private, self.queues["secondary"], self.first["secondary"])
        for role in ("primary", "secondary"):
            for record in self.compare[role]["records"]:
                record["compared_at_utc"] = LATE
            self.compare[role]["exported_at_utc"] = LATE
        third_queue = qa.third_queue_for_flags(self.private, [ident])
        third = first_export(third_queue, "person_third")
        third_compare = comparison(self.private, third_queue, third)
        with self.assertRaisesRegex(ValueError, "third review preceded"):
            qa.evaluate(self.private, self.queues, self.first, self.compare,
                        asr(self.private), tertiary_first=third,
                        tertiary_compare=third_compare)
        third["started_at_utc"] = LATER
        third["exported_at_utc"] = LATER
        for record in third["records"]:
            record["submitted_at_utc"] = LATER
        third_compare = comparison(self.private, third_queue, third)
        for record in third_compare["records"]:
            record["compared_at_utc"] = LATER
        third_compare["exported_at_utc"] = LATER
        report = qa.evaluate(self.private, self.queues, self.first, self.compare,
                             asr(self.private), tertiary_first=third,
                             tertiary_compare=third_compare)
        self.assertTrue(report["third_reviewed"])

    def test_comparison_export_cannot_precede_last_mark(self) -> None:
        self.compare["primary"]["records"][-1]["compared_at_utc"] = LATE
        with self.assertRaisesRegex(ValueError, "export precedes"):
            qa.evaluate(self.private, self.queues, self.first, self.compare,
                        asr(self.private))


if __name__ == "__main__":
    unittest.main()
