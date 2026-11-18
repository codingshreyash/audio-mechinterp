"""CPU-only structural tests; never import or load Voxtral weights."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import voxtral_fit_pilot_capability_v1 as pilot  # noqa: E402


class PilotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.index = json.loads(pilot.INDEX.read_text())

    def test_exact_32_fit_jobs_only(self):
        jobs = pilot.select_pilot(self.index)
        self.assertEqual(len(jobs), 32)
        self.assertEqual({row["pair_id"] for row, _, _ in jobs}, set(pilot.PILOT_PAIR_IDS))
        self.assertTrue(all(row["split"] == "fit_pool" and row["map_fit"] for row, _, _ in jobs))
        self.assertEqual({row["engine"] for row, _, _ in jobs}, {"piper", "kokoro"})
        self.assertEqual(len({(row["pair_id"], row["engine"], row["phrasing_id"]) for row, _, _ in jobs}), 16)

    def test_reject_nonfit_marked_pilot(self):
        changed = copy.deepcopy(self.index)
        next(row for row in changed["rows"] if row["split"] == "development")["pilot_fit"] = True
        with self.assertRaises(ValueError):
            pilot.select_pilot(changed)

    def test_answer_classification_requires_complete_exact_and_eos(self):
        self.assertTrue(pilot.classify_answer("  Deer  ", "deer", True)["exact_expected_word_and_eos"])
        for answer, eos in (("Deer.", True), ("Deer Moose", True), ("Deer", False), ("", True)):
            self.assertFalse(pilot.classify_answer(answer, "deer", eos)["exact_expected_word_and_eos"])
        hidden = pilot.classify_answer("Deer", "deer", True, [24])
        self.assertFalse(hidden["exact_expected_word_and_eos"])
        self.assertEqual(hidden["classification"], "disallowed_special")

    def test_canonical_sequences_are_frozen_and_include_eos(self):
        row, word, clip = pilot.select_pilot(self.index)[0]
        ids = pilot.canonical_ids(clip, word, 2)
        self.assertEqual(ids[-1], 2)
        self.assertGreaterEqual(len(ids), 2)
        self.assertEqual(row["split"], "fit_pool")

    def test_reject_changed_source_binding(self):
        changed = copy.deepcopy(self.index)
        changed["source_manifest_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            pilot.select_pilot(changed)

    def test_snapshot_rehashes_shards_and_index_bound_assets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = {
                "config.json": b"config A", "preprocessor_config.json": b"processor A",
                "generation_config.json": b"generation A", "tekken.json": b"tokenizer A",
                "model.safetensors.index.json": b"index A",
                "model-00001-of-00002.safetensors": b"first shard A",
                "model-00002-of-00002.safetensors": b"second shard A",
            }
            for name, content in files.items():
                (root / name).write_bytes(content)
            tree = root / ".cache/huggingface/trees" / f"{pilot.REVISION}.json"
            tree.parent.mkdir(parents=True)
            tree.write_text(json.dumps({"files": {
                name: {"size": len(files[name]), "lfs_size": len(files[name]),
                       "lfs_sha256": hashlib.sha256(files[name]).hexdigest()}
                for name in ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors")
            } | {
                "preprocessor_config.json": {"size": len(files["preprocessor_config.json"])},
                "tekken.json": {"size": len(files["tekken.json"])},
                "generation_config.json": {"size": len(files["generation_config.json"])},
            }}))
            index = {
                "processor_config_sha256": hashlib.sha256(files["preprocessor_config.json"]).hexdigest(),
                "tokenizer_asset_sha256": hashlib.sha256(files["tekken.json"]).hexdigest(),
            }
            config_sha = hashlib.sha256(files["config.json"]).hexdigest()
            generation_sha = hashlib.sha256(files["generation_config.json"]).hexdigest()
            with patch.object(pilot, "SNAPSHOT", root), patch.object(pilot, "MODEL_CONFIG_SHA256", config_sha), patch.object(pilot, "GENERATION_CONFIG_SHA256", generation_sha):
                good = pilot.validate_snapshot(index)
                self.assertEqual(good["weight_shards"]["model-00001-of-00002.safetensors"]["sha256"],
                                 hashlib.sha256(files["model-00001-of-00002.safetensors"]).hexdigest())
                for name in ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors",
                             "preprocessor_config.json", "tekken.json", "generation_config.json"):
                    original = files[name]
                    (root / name).write_bytes(b"X" + original[1:])  # same size, different hash
                    with self.assertRaises(ValueError, msg=name):
                        pilot.validate_snapshot(index)
                    (root / name).write_bytes(original)


if __name__ == "__main__":
    unittest.main()
