"""No-model tests for paired PCM, saved-data audit, and blind queue."""

from __future__ import annotations

import copy
import hashlib
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from . import audit, queue, render


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class FixtureSourceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(dir=render.HERE)
        self.output = Path(self.temp.name)
        for engine in audit.ENGINES:
            (self.output / "raw" / engine).mkdir(parents=True)
            (self.output / "processed" / engine).mkdir(parents=True)
        (self.output / "clips").mkdir()
        self.manifest = self.make_manifest()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def file_record(self, path: Path, pcm: bytes) -> dict:
        render.write_pcm(path, pcm)
        return {"path": str(path.relative_to(render.ROOT)), "sha256": render.sha(path)}

    def make_manifest(self) -> dict:
        fragments = []
        data_by_key = {}
        for engine_no, engine in enumerate(audit.ENGINES):
            texts = {"prefix": "I said", "suffix": "and then stopped."}
            texts.update({f"word_{i:02d}_{word}": word
                          for i, pair in enumerate(render.PAIRS) for word in pair})
            for fragment_id, text in texts.items():
                if fragment_id == "prefix":
                    pcm = struct.pack("<hh", 100, 101)
                elif fragment_id == "suffix":
                    pcm = struct.pack("<hh", 200, 201)
                else:
                    i = int(fragment_id[5:7])
                    word_no = render.PAIRS[i].index(text)
                    values = [1000 + 10*i + word_no + engine_no, 5, 6]
                    if word_no:
                        values.append(7)
                    pcm = struct.pack("<" + "h"*len(values), *values)
                data_by_key[(engine, fragment_id)] = pcm
                raw = self.output / "raw" / engine / f"{fragment_id}.wav"
                processed = self.output / "processed" / engine / f"{fragment_id}.wav"
                raw_rec = self.file_record(raw, pcm)
                processed_rec = self.file_record(processed, pcm)
                processed_rec.update(pcm_sha256=digest(pcm), frames=len(pcm)//2)
                fragments.append({"engine": engine, "fragment_id": fragment_id,
                                  "text": text, "text_sha256": digest(text.encode()),
                                  "raw": raw_rec, "processed": processed_rec})
        rows = []
        for i, pair in enumerate(render.PAIRS):
            for engine in audit.ENGINES:
                prefix = data_by_key[(engine, "prefix")]
                suffix = data_by_key[(engine, "suffix")]
                words = {word: data_by_key[(engine, f"word_{i:02d}_{word}")]
                         for word in pair}
                slot = max(len(data) for data in words.values()) // 2
                start = len(prefix)//2 + 1920
                end = start + slot
                outputs = {}
                for word in pair:
                    raw = prefix + bytes(3840) + words[word] + \
                        bytes(2*slot-len(words[word])) + bytes(3840) + suffix
                    path = self.output / "clips" / f"{engine}_{i:02d}_{word}.wav"
                    render.write_pcm(path, raw)
                    outputs[word] = {"path": str(path.relative_to(render.ROOT)),
                                     "file_sha256": render.sha(path),
                                     "pcm_int16_sha256": digest(raw),
                                     "frames": len(raw)//2,
                                     "word_fragment_frames": len(words[word])//2,
                                     "spoken_script": render.reference_script(word)}
                rows.append({"pair_id": f"pair_{i:02d}", "engine": engine,
                             "words": list(pair), "word_slot_start_sample": start,
                             "word_slot_end_sample_exclusive": end,
                             "prefix_fragment_id": "prefix", "suffix_fragment_id": "suffix",
                             "word_fragment_ids": {word: f"word_{i:02d}_{word}" for word in pair},
                             "outside_slot_pcm_identical": True,
                             "inside_slot_pcm_differs": True, "outputs": outputs})
        return {"status": "synthetic_stimuli_only_no_candidate_model_answers",
                "matched_pairs": 16, "wavs": 32, "sample_rate": 16000,
                "subtype": "PCM_16", "gap_samples": 1920,
                "prefix": "I said", "suffix": "and then stopped.",
                "fragments": fragments, "pairs": rows}

    def test_full_fake_population_and_blind_queue(self) -> None:
        result = audit.verify(self.manifest, self.output)
        self.assertEqual((result["matched_pairs"], result["wavs"]), (16, 32))
        binding = {**result, "manifest_sha256": "a"*64,
                   "auditor_sha256": render.sha(audit.Path(audit.__file__))}
        self.assertEqual(audit.verify_saved_report(self.manifest, binding, self.output,
                                                   "a"*64), binding)
        private, public = queue.build(self.manifest, binding, "a"*64, "b"*64)
        self.assertEqual(len(private["cases"]), 32)
        self.assertEqual(len(public["opaque_ids_in_order"]), 32)
        self.assertEqual(len(set(public["opaque_ids_in_order"])), 32)
        self.assertNotIn("banana", json.dumps(public))
        self.assertNotIn("source_path", json.dumps(public))
        by_id = {case["opaque_id"]: case for case in private["cases"]}
        order = public["opaque_ids_in_order"]
        for a, b in zip(order, order[1:]):
            self.assertNotEqual((by_id[a]["pair_id"], by_id[a]["engine"]),
                                (by_id[b]["pair_id"], by_id[b]["engine"]))

    def test_tamper_inside_word_and_outside_word_fail(self) -> None:
        row = self.manifest["pairs"][0]
        word = row["words"][0]
        record = row["outputs"][word]
        path = render.ROOT / record["path"]
        original = render.read_pcm(path)
        for frame in (row["word_slot_start_sample"], 0):
            altered = bytearray(original)
            altered[2*frame] ^= 1
            path.unlink()
            render.write_pcm(path, bytes(altered))
            record["file_sha256"] = render.sha(path)
            record["pcm_int16_sha256"] = digest(bytes(altered))
            with self.assertRaisesRegex(ValueError, "final clip differs"):
                audit.verify(self.manifest, self.output)
            path.unlink()
            render.write_pcm(path, original)
            record["file_sha256"] = render.sha(path)
            record["pcm_int16_sha256"] = digest(original)

    def test_missing_or_duplicate_case_rejected(self) -> None:
        missing = copy.deepcopy(self.manifest)
        missing["pairs"].pop()
        with self.assertRaisesRegex(ValueError, "pair population incomplete"):
            audit.verify(missing, self.output)
        duplicate = copy.deepcopy(self.manifest)
        duplicate["pairs"][1] = copy.deepcopy(duplicate["pairs"][0])
        with self.assertRaisesRegex(ValueError, "pair identity/words mismatch"):
            audit.verify(duplicate, self.output)

    def test_metadata_exclusion_and_command_identity(self) -> None:
        self.assertEqual(render.verify_population(), render.EXPECTED_METADATA_SHA)
        self.assertEqual(render.piper_configuration()["voice"], "en_US-amy-medium")
        self.assertEqual(render.command_identity()["say_language"], "en_US")

    def test_forged_positive_freeze_receipt_is_rejected(self) -> None:
        forged = self.output / "FORGED_FREEZE.json"
        forged_review = self.output / "FORGED_GO.md"
        forged_review.write_text("**Decision: GO**\n```json\n{}\n```\n")
        forged.write_text(json.dumps({"status": "independently_reviewed_renderer_frozen",
                                      "plan_sha256": render.sha(render.PLAN),
                                      "renderer_sha256": render.sha(Path(render.__file__)),
                                      "auditor_sha256": render.sha(Path(audit.__file__)),
                                      "queue_sha256": render.sha(Path(queue.__file__)),
                                      "source_tts_plan_sha256": render.sha(render.SOURCE_PLAN),
                                      "source_tts_helper_sha256": render.sha(render.SOURCE_HELPER)}))
        with mock.patch.object(render, "FREEZE", forged), \
                mock.patch.object(render, "SOURCE_GO_REVIEW", forged_review):
            with self.assertRaisesRegex(ValueError, "uncommitted"):
                render.freeze_receipt()

    def test_forged_technical_binding_is_rejected_before_queue(self) -> None:
        report = audit.verify(self.manifest, self.output)
        forged = {"status": report["status"], "matched_pairs": 16, "wavs": 32,
                  "all_outside_word_pcm_equal": True,
                  "candidate_model_inference_authorized": False,
                  "manifest_sha256": "a"*64,
                  "auditor_sha256": render.sha(Path(audit.__file__)),
                  "pairs": []}
        # The former queue admission accepted these superficial fields.
        queue.build(self.manifest, forged, "a"*64, "b"*64)
        with self.assertRaisesRegex(ValueError, "differs from independent saved-data audit"):
            audit.verify_saved_report(self.manifest, forged, self.output, "a"*64)
        (self.output / "manifest.json").write_text(json.dumps(self.manifest))
        forged["manifest_sha256"] = render.sha(self.output / "manifest.json")
        (self.output / "TECHNICAL_BINDING.json").write_text(json.dumps(forged))
        argv = ["queue.py", "--validate-only", "--output-dir", str(self.output)]
        # The production entry still calls validate_provenance; this fake
        # fixture bypasses only that distinct source gate to isolate re-audit.
        with mock.patch.object(audit, "validate_provenance"), \
                mock.patch.object(sys, "argv", argv):
            with self.assertRaisesRegex(ValueError, "differs from independent saved-data audit"):
                queue.main()


if __name__ == "__main__":
    unittest.main()
