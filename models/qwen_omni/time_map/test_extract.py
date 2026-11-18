"""CPU-only guards and rowwise artifact checks; no checkpoint load."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from extract import assert_pinned_transformers, read_fit_records, save_matrix


class ExtractorTests(unittest.TestCase):
    def test_transformers_install_has_expected_commit(self):
        self.assertEqual(assert_pinned_transformers(), "e7e568dc7673ac99a87660c7c50e84830268285c")

    def test_read_fit_records_rejects_nonfit_and_bad_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            audio = base / "fit.wav"
            audio.write_bytes(b"stand-in-audio")
            record = {
                "item_id": "fit:one",
                "source_item": {"split": "fit"},
                "source_line_sha256": "source-line",
                "outputs": {"piper": {"processed": {
                    "path": "fit.wav",
                    "file_sha256": hashlib.sha256(audio.read_bytes()).hexdigest(),
                    "sample_rate": 16000,
                    "channels": 1,
                }}},
            }
            manifest = {"selection": {"split": "fit"}, "records": [record]}
            path = base / "render.json"
            path.write_text(json.dumps(manifest))
            _, rows = read_fit_records(path.resolve(), limit=None, engines=("piper",))
            self.assertEqual(rows[0][0]["item_id"], "fit:one")
            record["source_item"]["split"] = "dev"
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "non-fit"):
                read_fit_records(path.resolve(), limit=None, engines=("piper",))
            record["source_item"]["split"] = "fit"
            record["outputs"]["piper"]["processed"]["file_sha256"] = "bad"
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                read_fit_records(path.resolve(), limit=None, engines=("piper",))

    def test_matrix_rows_are_addressable_and_shape_checked(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "matrix.npy"
            source = torch.tensor([[1.5, 2.5], [3.5, 4.5]], dtype=torch.bfloat16)
            info = save_matrix(path, source, expected_rows=2, expected_width=2)
            mapped = np.load(path, mmap_mode="r")
            self.assertEqual(mapped.shape, (2, 2))
            self.assertEqual(mapped[1].tolist(), [3.5, 4.5])
            self.assertEqual(info["source_dtype"], "torch.bfloat16")
            with self.assertRaisesRegex(ValueError, "Activation shape mismatch"):
                save_matrix(path, source, expected_rows=3, expected_width=2)


if __name__ == "__main__":
    unittest.main()
