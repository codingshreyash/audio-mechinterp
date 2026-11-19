"""No-model tests for the blind form's public-source and hash boundaries."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from . import listening_form as form


class BlindFormTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.bundle = Path(self.temp.name)
        self.ids = [f"{i:020x}" for i in range(32)]
        for ident in self.ids:
            (self.bundle / f"{ident}.wav").write_bytes(b"RIFF" + ident.encode())
        self.queue = {
            "status": "blinded_listening_queue_no_results",
            "scope": "Listen to anonymous clips",
            "provenance": {name: "a" * 64 for name in (
                "manifest_sha256", "technical_binding_sha256",
                "auditor_sha256", "queue_builder_sha256")},
            "planned_clips": 32, "opaque_ids_in_order": self.ids.copy(),
            "review_form_fields": form.EXPECTED_FIELDS,
        }
        self.save_queue()

    def save_queue(self) -> None:
        (self.bundle / "QUEUE.json").write_text(json.dumps(self.queue))

    def test_build_embeds_only_public_ids_and_clip_hashes(self) -> None:
        private = self.bundle.parent / f"QA_PRIVATE_CASES_{self.bundle.name}.json"
        private.write_text("SECRET_REFERENCE_WORD_DO_NOT_LEAK")
        self.addCleanup(private.unlink)
        output = self.bundle / "LISTENING_FORM.html"
        result = form.build(self.bundle, output)
        html = output.read_text()
        self.assertEqual(result["clips"], 32)
        self.assertEqual(result["queue_sha256"], form.sha(self.bundle / "QUEUE.json"))
        self.assertIn(self.ids[0], html)
        self.assertIn(form.sha(self.bundle / f"{self.ids[0]}.wav"), html)
        self.assertNotIn("SECRET_REFERENCE_WORD_DO_NOT_LEAK", html)
        self.assertNotIn("reference_script", html)
        with self.assertRaises(FileExistsError):
            form.build(self.bundle, output)

    def test_rejects_leaked_reference_field(self) -> None:
        self.queue["reference_word"] = "banana"
        self.save_queue()
        with self.assertRaisesRegex(ValueError, "schema"):
            form.public_data(self.bundle)

    def test_rejects_private_map_in_listener_directory(self) -> None:
        (self.bundle / "QA_PRIVATE_CASES.json").write_text("SECRET_REFERENCE_WORD")
        with self.assertRaisesRegex(ValueError, "unexpected files"):
            form.public_data(self.bundle)

    def test_rejects_duplicate_or_missing_clip(self) -> None:
        self.queue["opaque_ids_in_order"][1] = self.ids[0]
        self.save_queue()
        with self.assertRaisesRegex(ValueError, "duplicated"):
            form.public_data(self.bundle)
        self.queue["opaque_ids_in_order"] = self.ids.copy()
        self.save_queue()
        (self.bundle / f"{self.ids[0]}.wav").unlink()
        with self.assertRaisesRegex(ValueError, "population"):
            form.public_data(self.bundle)

    def test_rejects_extra_wav_and_symlink(self) -> None:
        (self.bundle / "extra.wav").write_bytes(b"RIFFextra")
        with self.assertRaisesRegex(ValueError, "population"):
            form.public_data(self.bundle)
        (self.bundle / "extra.wav").unlink()
        path = self.bundle / f"{self.ids[0]}.wav"
        path.unlink()
        target = self.bundle / f"{self.ids[1]}.wav"
        path.symlink_to(target)
        with self.assertRaisesRegex(ValueError, "population|linked"):
            form.public_data(self.bundle)


if __name__ == "__main__":
    unittest.main()
