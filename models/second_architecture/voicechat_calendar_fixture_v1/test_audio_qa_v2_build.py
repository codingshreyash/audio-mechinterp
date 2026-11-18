"""Synthetic-only QA builder tests. These are not human listening records."""

from __future__ import annotations

import hashlib
import hmac
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from .audio_qa_v2 import RENDER, canonical_bytes, check_audio_qa_v2, digest
from .audio_qa_v2_build import build_receipts, finalize_manifest
from .calendar_stub import grant_signing_bytes
from . import test_audio_qa_v2 as fixtures


class QABuilderTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.AudioQAV2Tests(
            methodName="test_complete_synthetic_cohort_validates_without_asr_exact_requirement")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bundle = self.root / "bundle"
        self.records = self.root / "records"
        self.out = self.root / "prepared"
        self.bundle.mkdir()
        self.records.mkdir()
        shutil.copyfile(self.fixture.root / "blind_assignment.json",
                        self.bundle / "blind_assignment.json")
        assignment = json.loads((self.bundle / "blind_assignment.json").read_bytes())
        render = json.loads(RENDER.read_bytes())
        by_hash = {row["final_sha256"]: RENDER.parent / row["final_path"]
                   for row in render["rows"]}
        for item in assignment["opaque_order"]:
            shutil.copyfile(by_hash[item["wav_sha256"]],
                            self.bundle / f"{item['opaque_id']}.wav")
        for row in self.fixture.manifest["rows"]:
            receipt = json.loads((self.fixture.root / row["receipt"]["path"]).read_bytes())
            for role in ("primary", "secondary"):
                descriptor = receipt[role]
                if descriptor is None:
                    continue
                for stage in ("blind", "script_aware"):
                    source = self.fixture.root / descriptor[stage]["path"]
                    # The source record itself carries the exact opaque ID.
                    opaque = json.loads(source.read_bytes())["opaque_id"]
                    target = self.records / role / f"{role}__{opaque}__{stage}.json"
                    target.parent.mkdir(exist_ok=True)
                    shutil.copyfile(source, target)

    def build(self, *, tertiary_id=None):
        return build_receipts(
            self.bundle, self.records, self.out, primary_id="primary",
            secondary_id="secondary", tertiary_id=tertiary_id,
            completed_at_utc=fixtures.QA_TIME, controller_id=fixtures.CONTROLLER)

    def external_grants(self, *, duplicate_id=False):
        grants = self.root / "external-grants"
        grants.mkdir()
        index = json.loads((self.out / "prepared_index.json").read_bytes())
        for row in [r for r in index["rows"] if r["case_id"] == "C"]:
            grant = {"grant_id": "same-id" if duplicate_id else
                     "synthetic-" + row["voice_id"],
                     "controller_id": fixtures.CONTROLLER, "case_id": "C",
                     "source_wav_sha256": row["wav_sha256"],
                     "qa_receipt_sha256": row["receipt"]["sha256"],
                     "authorized_date": "2026-11-19",
                     "authorized_event_ids": ["DR19"]}
            grant["signature_hmac_sha256"] = hmac.new(
                fixtures.KEY, grant_signing_bytes(grant), hashlib.sha256).hexdigest()
            (grants / f"{row['voice_id']}.json").write_bytes(canonical_bytes(grant))
        return grants

    def test_full_synthetic_cohort_builds_without_changing_human_stages(self):
        before = {p.relative_to(self.records): digest(p)
                  for p in self.records.rglob("*.json")}
        result = self.build()
        self.assertEqual((result["eligible"], result["secondary_reviews"],
                          result["tertiary_reviews"]), (12, 11, 0))
        self.assertFalse(result["external_anchor_created"])
        self.assertFalse((self.out / "manifest.json").exists())
        self.assertFalse((self.out / "ledger.json").exists())
        self.assertEqual(before, {p.relative_to(self.records): digest(p)
                                  for p in self.records.rglob("*.json")})
        index = json.loads((self.out / "prepared_index.json").read_bytes())
        self.assertEqual(len(index["rows"]), 12)
        for original in self.records.rglob("*.json"):
            self.assertEqual(digest(original), digest(self.out / "stages" / original.name))

    def test_missing_secondary_fails_without_output(self):
        victim = next((self.records / "secondary").glob("*__blind.json"))
        victim.unlink()
        with self.assertRaisesRegex(ValueError, "required human listener stage"):
            self.build()
        self.assertFalse(self.out.exists())

    def test_duplicate_stage_fails_without_output(self):
        source = next((self.records / "primary").glob("*__blind.json"))
        duplicate = self.records / "duplicate" / source.name
        duplicate.parent.mkdir()
        shutil.copyfile(source, duplicate)
        with self.assertRaisesRegex(ValueError, "duplicate listener stage"):
            self.build()
        self.assertFalse(self.out.exists())

    def test_conflicting_stage_identity_fails_without_output(self):
        source = next((self.records / "primary").glob("*__blind.json"))
        value = json.loads(source.read_bytes())
        value["reviewer_audit_id"] = "secondary"
        source.write_bytes(canonical_bytes(value))
        with self.assertRaisesRegex(ValueError, "filename/content"):
            self.build()
        self.assertFalse(self.out.exists())

    def test_disagreement_requires_third_listener_and_preserves_dissent(self):
        target = next((self.records / "secondary").glob("*__script_aware.json"))
        secondary = json.loads(target.read_bytes())
        secondary["first_action_heard"] = "cancel_calendar_events"
        target.write_bytes(canonical_bytes(secondary))
        with self.assertRaisesRegex(ValueError, "third reviewer"):
            self.build()
        self.assertFalse(self.out.exists())
        opaque = secondary["opaque_id"]
        source_blind = self.records / "primary" / f"primary__{opaque}__blind.json"
        source_aware = self.records / "primary" / f"primary__{opaque}__script_aware.json"
        third_dir = self.records / "tertiary"
        third_dir.mkdir()
        blind = json.loads(source_blind.read_bytes())
        blind["reviewer_audit_id"] = "tertiary"
        third_blind = third_dir / f"tertiary__{opaque}__blind.json"
        third_blind.write_bytes(canonical_bytes(blind))
        aware = json.loads(source_aware.read_bytes())
        aware["reviewer_audit_id"] = "tertiary"
        aware["blind_record_sha256"] = digest(third_blind)
        (third_dir / f"tertiary__{opaque}__script_aware.json").write_bytes(
            canonical_bytes(aware))
        result = self.build(tertiary_id="tertiary")
        self.assertEqual(result["tertiary_reviews"], 1)
        self.assertEqual(result["eligible"], 12)
        index = json.loads((self.out / "prepared_index.json").read_bytes())
        row = next(r for r in index["rows"]
                   if json.loads((self.out / r["receipt"]["path"]).read_bytes())["primary"]
                   ["blind"]["path"].endswith(f"__{opaque}__blind.json"))
        receipt = json.loads((self.out / row["receipt"]["path"]).read_bytes())
        self.assertTrue(receipt["tertiary_required"])
        self.assertIsNotNone(receipt["tertiary"])
        self.assertEqual(digest(target), digest(self.out / "stages" / target.name))

    def test_prepared_human_stage_tamper_blocks_manifest(self):
        self.build()
        copied = next((self.out / "stages").glob("primary__*__blind.json"))
        copied.write_bytes(copied.read_bytes() + b" ")
        grants = self.root / "external-grants"
        grants.mkdir()
        with self.assertRaisesRegex(ValueError, "noncanonical JSON bytes"):
            finalize_manifest(self.out, grants, controller_key=fixtures.KEY,
                              trusted_controller_id=fixtures.CONTROLLER)
        self.assertFalse((self.out / "manifest.json").exists())

    def test_manifest_requires_separate_signed_c_grants(self):
        self.build()
        grants = self.root / "external-grants"
        grants.mkdir()
        with self.assertRaisesRegex(ValueError, "exactly two external"):
            finalize_manifest(self.out, grants, controller_key=fixtures.KEY,
                              trusted_controller_id=fixtures.CONTROLLER)
        index = json.loads((self.out / "prepared_index.json").read_bytes())
        for row in [r for r in index["rows"] if r["case_id"] == "C"]:
            grant = {"grant_id": "synthetic-" + row["voice_id"],
                     "controller_id": fixtures.CONTROLLER, "case_id": "C",
                     "source_wav_sha256": row["wav_sha256"],
                     "qa_receipt_sha256": row["receipt"]["sha256"],
                     "authorized_date": "2026-11-19",
                     "authorized_event_ids": ["DR19"]}
            grant["signature_hmac_sha256"] = hmac.new(
                fixtures.KEY, grant_signing_bytes(grant), hashlib.sha256).hexdigest()
            (grants / f"{row['voice_id']}.json").write_bytes(canonical_bytes(grant))
        result = finalize_manifest(self.out, grants, controller_key=fixtures.KEY,
                                   trusted_controller_id=fixtures.CONTROLLER)
        self.assertFalse(result["external_anchor_created"])
        self.assertFalse(result["ledger_created"])
        manifest = json.loads((self.out / "manifest.json").read_bytes())
        self.assertEqual(len(manifest["rows"]), 12)
        self.assertEqual(len(manifest["c_grants"]), 2)
        # The ledger here is synthetic test input only; production builder
        # deliberately never writes one.
        ledger = {"schema": "voicechat_preinference_ledger_v2",
                  "qa_manifest_sha256": digest(self.out / "manifest.json"),
                  "eligibility": manifest["eligibility"],
                  "external_anchor_audit_id": "synthetic-test-only",
                  "anchored_utc": fixtures.ANCHOR_TIME}
        (self.out / "test_only_ledger.json").write_bytes(canonical_bytes(ledger))
        checked = check_audio_qa_v2(self.out / "manifest.json",
                                    self.out / "test_only_ledger.json",
                                    controller_key=fixtures.KEY,
                                    trusted_controller_id=fixtures.CONTROLLER)
        self.assertEqual((checked["primary_reviews"], checked["secondary_reviews"]), (12, 11))

    def test_duplicate_signed_c_grant_ids_rejected_before_copy(self):
        self.build()
        grants = self.external_grants(duplicate_id=True)
        with self.assertRaisesRegex(ValueError, "C grant ID reused"):
            finalize_manifest(self.out, grants, controller_key=fixtures.KEY,
                              trusted_controller_id=fixtures.CONTROLLER)
        self.assertFalse((self.out / "grants").exists())
        self.assertFalse((self.out / "manifest.json").exists())

    def test_second_grant_copy_failure_leaves_retryable_prepared_evidence(self):
        self.build()
        grants = self.external_grants()
        before = {p.relative_to(self.out): digest(p) for p in self.out.rglob("*")
                  if p.is_file()}
        original_copy = shutil.copyfile
        calls = 0

        def fail_second(source, target, *args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("synthetic second grant copy failure")
            return original_copy(source, target, *args, **kwargs)

        with patch("models.second_architecture.voicechat_calendar_fixture_v1."
                   "audio_qa_v2_build.shutil.copyfile", side_effect=fail_second):
            with self.assertRaisesRegex(OSError, "second grant copy failure"):
                finalize_manifest(self.out, grants, controller_key=fixtures.KEY,
                                  trusted_controller_id=fixtures.CONTROLLER)
        self.assertFalse((self.out / "grants").exists())
        self.assertFalse((self.out / "manifest.json").exists())
        self.assertEqual(before, {p.relative_to(self.out): digest(p)
                                  for p in self.out.rglob("*") if p.is_file()})
        finalize_manifest(self.out, grants, controller_key=fixtures.KEY,
                          trusted_controller_id=fixtures.CONTROLLER)
        self.assertTrue((self.out / "manifest.json").is_file())


if __name__ == "__main__":
    unittest.main()
