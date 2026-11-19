"""Fake human-export tests; these records are never study QA."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from . import timing_bundle as bundle
from . import corpus


class TimingBundleTest(unittest.TestCase):
    def _built(self, root: Path) -> tuple[Path, dict]:
        output = root / "bundle"
        result = bundle.build(output)
        self.assertEqual(result["rows"], 12)
        return output, json.loads((output / "QUEUE.json").read_text())

    def _export(self, path: Path, queue: dict, reviewer: str,
                *, change_first: bool = False) -> None:
        queue_bytes = (json.dumps(queue, sort_keys=True, indent=2) + "\n").encode()
        records = []
        for idx, row in enumerate(queue["rows"]):
            decision = [1000, 2000] if not (change_first and idx == 0) else [1100, 2100]
            records.append({"source_id": row["source_id"],
                            "clean_wav_sha256": row["clean_wav_sha256"],
                            "submitted_at_utc": "2026-11-19T00:00:00Z",
                            "status": "marked", "decision": decision,
                            "displaced": [5000, 6000], "note": "fake unit test"})
        data = {"schema": bundle.SCHEMA, "reviewer_id": reviewer,
                "queue_sha256": hashlib.sha256(queue_bytes).hexdigest(),
                "source_manifest_sha256": bundle.SOURCE_MANIFEST_SHA256,
                "records": records}
        path.write_text(json.dumps(data))

    def test_bundle_binds_all_real_clean_wavs(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            output, queue = self._built(Path(temp))
            self.assertEqual(len(queue["rows"]), 12)
            self.assertEqual(queue["source_manifest_sha256"], bundle.SOURCE_MANIFEST_SHA256)
            for row in queue["rows"]:
                self.assertEqual(bundle._sha(output / row["clean_wav_path"]),
                                 row["clean_wav_sha256"])
            self.assertIn(bundle._sha(output / "QUEUE.json"),
                          (output / "TIMING_FORM.html").read_text())

    def test_fake_consensus_assembly_matches_stage2_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output, queue = self._built(root)
            first, second = root / "one.json", root / "two.json"
            self._export(first, queue, "reviewer-one")
            self._export(second, queue, "reviewer-two")
            result = bundle.assemble(output, first, second, root / "assembled.json")
            self.assertEqual(result["third_reviewed_sources"], [])
            data = json.loads((root / "assembled.json").read_text())
            self.assertEqual(data["records"][0]["adjudication"]["decision"], [1000, 2000])
            manifest = json.loads((bundle.SOURCE / "source_manifest.json").read_text())
            checked = corpus._checked_annotations(
                bundle.SOURCE, corpus._spec(), manifest, root / "assembled.json")
            self.assertEqual(len(checked["records"]), 12)

    def test_disagreement_requires_third_review(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output, queue = self._built(root)
            first, second, third = (root / name for name in ("one.json", "two.json", "three.json"))
            self._export(first, queue, "reviewer-one")
            self._export(second, queue, "reviewer-two", change_first=True)
            with self.assertRaisesRegex(ValueError, "Third reviewer required"):
                bundle.assemble(output, first, second, root / "assembled.json")
            self._export(third, queue, "reviewer-three")
            result = bundle.assemble(output, first, second, root / "assembled.json", third)
            self.assertEqual(result["third_reviewed_sources"], [queue["rows"][0]["source_id"]])
            data = json.loads((root / "assembled.json").read_text())
            self.assertEqual(data["records"][0]["adjudication"]["by"], "reviewer-three")
            manifest = json.loads((bundle.SOURCE / "source_manifest.json").read_text())
            checked = corpus._checked_annotations(
                bundle.SOURCE, corpus._spec(), manifest, root / "assembled.json")
            self.assertEqual(len(checked["records"]), 12)

    def test_tampered_bundle_wav_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output, queue = self._built(root)
            first, second = root / "one.json", root / "two.json"
            self._export(first, queue, "reviewer-one")
            self._export(second, queue, "reviewer-two")
            wav = output / queue["rows"][0]["clean_wav_path"]
            with wav.open("ab") as stream:
                stream.write(b"tamper")
            with self.assertRaisesRegex(ValueError, "WAV changed"):
                bundle.assemble(output, first, second, root / "assembled.json")

    def test_tampered_queue_order_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output, queue = self._built(root)
            first, second = root / "one.json", root / "two.json"
            self._export(first, queue, "reviewer-one")
            self._export(second, queue, "reviewer-two")
            queue["rows"][0], queue["rows"][1] = queue["rows"][1], queue["rows"][0]
            (output / "QUEUE.json").write_text(json.dumps(queue))
            with self.assertRaisesRegex(ValueError, "queue differs"):
                bundle.assemble(output, first, second, root / "assembled.json")


if __name__ == "__main__":
    unittest.main()
