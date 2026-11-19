"""No-model tests for the blind form's public-source and hash boundaries."""

from __future__ import annotations

import hashlib
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
        self.manifest_sha = "a" * 64
        self.ids = []
        for i in range(32):
            payload = b"RIFF" + i.to_bytes(4, "little")
            wav_sha = hashlib.sha256(payload).hexdigest()
            ident = hashlib.sha256(("word-disjoint-listening-v2|"
                                    + self.manifest_sha + "|" + wav_sha).encode()).hexdigest()[:20]
            self.ids.append(ident)
            (self.bundle / f"{ident}.wav").write_bytes(payload)
        self.queue = {
            "status": "blinded_listening_queue_no_results",
            "scope": "Listen to anonymous clips",
            "provenance": {name: self.manifest_sha for name in (
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

    def test_rejects_swapped_wavs_under_unchanged_queue(self) -> None:
        first = self.bundle / f"{self.ids[0]}.wav"
        second = self.bundle / f"{self.ids[1]}.wav"
        original_queue_sha = form.sha(self.bundle / "QUEUE.json")
        temp = self.bundle / "swap.tmp"
        first.rename(temp)
        second.rename(first)
        temp.rename(second)
        self.assertEqual(form.sha(self.bundle / "QUEUE.json"), original_queue_sha)
        with self.assertRaisesRegex(ValueError, "derived opaque ID"):
            form.public_data(self.bundle)


if __name__ == "__main__":
    unittest.main()
