"""Corruption checks for the independent Whisper saved-result auditor."""

from __future__ import annotations

import copy
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from models.second_architecture import whisper_fit_word_window_causal_audit_v1 as audit


def fake_result(word: str, *, hook_count: int | None = None) -> dict:
    token = 100 if word == "silver" else 101
    return {"token_ids": [50257, token, 50256], "transcript": f"Say {word} now.",
            "eos_seen": True, "censored": False, "step_scores": [],
            "elapsed_seconds": 0.1, "rss_bytes": 100, "hook_count": hook_count}


class WhisperSavedAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name)
        (self.output / "records.jsonl").write_text("{}\n")
        self.cases = {f"{engine}_{word}": {} for engine in audit.WINDOWS for word in audit.WORDS}
        hashes = {case: {site: hashlib.sha256(f"{case}/{site}".encode()).hexdigest()
                         for site in ("mel", *audit.CAPTURES)} for case in self.cases}
        rows = [{"type": "preflight"}]
        for case in self.cases:
            word = case.rsplit("_", 1)[1]
            rows.append({"type": "baseline", "case": case,
                         "result": fake_result(word), "capture_sha256": hashes[case],
                         "eligible": True, "capture_complete": True,
                         "capture_hook_counts": dict.fromkeys(audit.CAPTURES, 1)})
        rows.extend({"type": "trace"} for _ in range(2))
        for recipient, donor, site, condition, seed in sorted(audit.expected_matrix()):
            engine, recipient_word = recipient.rsplit("_", 1)
            donor_word = donor.rsplit("_", 1)[1]
            word_window, prefix = audit.WINDOWS[engine]
            window = ((0, 1500) if condition == "full_donor" else
                      prefix if condition == "prefix" else word_window)
            output_word = donor_word if condition in {"word", "full_donor"} else recipient_word
            result = fake_result(output_word, hook_count=1)
            base = fake_result(recipient_word)
            outcome = audit.recompute_outcome(result["transcript"], recipient_word,
                                               donor_word, base["transcript"])
            outcome["token_difference_count"] = int(output_word != recipient_word)
            row = {"type": "intervention", "recipient": recipient, "donor": donor,
                   "site": site, "condition": condition, "window": list(window),
                   "result": result, "hook_count": 1, "valid_hook": True,
                   "recipient_state_sha256": hashes[recipient][site],
                   "source_state_sha256": hashes[recipient if condition == "self" else donor][site],
                   "outcome": outcome,
                   "control_pass": True if condition in {"self", "full_donor"} else None}
            if condition in {"full_donor", "random"}:
                row["seed"] = seed
                row["patch_audit"] = {}
            if condition == "random":
                row["patch_audit"] = self.random_patch(engine, seed)
            rows.append(row)
        rows.append({"type": "complete", "elapsed_seconds": 1.0, "rss_bytes": 100})
        self.rows = rows
        self.silver = np.zeros(audit.STATE_SHAPE, dtype=np.float32)
        self.teal = np.ones(audit.STATE_SHAPE, dtype=np.float32)

    @staticmethod
    def random_patch(engine: str, seed: int) -> dict:
        import torch

        start, end = audit.WINDOWS[engine][0]
        target = torch.linalg.vector_norm(torch.ones((1, end - start, 384)))
        generator = torch.Generator(device="cpu").manual_seed(seed)
        noise = torch.randn((1, end - start, 384), generator=generator)
        noise *= target / torch.linalg.vector_norm(noise)
        return {"random_edit_sha256": hashlib.sha256(noise.numpy().tobytes()).hexdigest(),
                "random_edit_norm": float(torch.linalg.vector_norm(noise)),
                "random_target_norm": float(target)}

    def check_rows(self, rows: list[dict]) -> dict:
        def capture(path: Path, _hash: str) -> np.ndarray:
            return self.teal if "_teal_" in path.name else self.silver

        with patch.object(audit, "load_capture", side_effect=capture), \
             patch.object(audit, "load_mel", return_value=np.zeros((1, 80, 3000), dtype=np.float32)), \
             patch.object(audit, "audit_traces", return_value={}):
            return audit.audit_rows(rows, self.output, self.cases)

    def test_complete_fake_matrix_and_controls(self) -> None:
        report = self.check_rows(self.rows)
        self.assertEqual((report["baseline_count"], report["intervention_count"]), (4, 48))

    def test_duplicate_arm_rejected(self) -> None:
        rows = copy.deepcopy(self.rows)
        rows[-2] = copy.deepcopy(rows[-3])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.check_rows(rows)

    def test_self_patch_token_change_rejected(self) -> None:
        rows = copy.deepcopy(self.rows)
        row = next(row for row in rows if row.get("condition") == "self")
        row["result"]["token_ids"][1] = 999
        row["outcome"]["token_difference_count"] = 1
        with self.assertRaisesRegex(ValueError, "self-patch"):
            self.check_rows(rows)

    def test_random_edit_hash_rejected(self) -> None:
        rows = copy.deepcopy(self.rows)
        row = next(row for row in rows if row.get("condition") == "random")
        row["patch_audit"]["random_edit_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "random norm or edit hash"):
            self.check_rows(rows)

    def test_reported_outcome_corruption_rejected(self) -> None:
        rows = copy.deepcopy(self.rows)
        row = next(row for row in rows if row.get("condition") == "word")
        row["outcome"]["target_replacement"] = False
        with self.assertRaisesRegex(ValueError, "stored outcome"):
            self.check_rows(rows)

    def test_zero_or_multiple_hook_rejected(self) -> None:
        for count in (0, 2):
            with self.subTest(count=count):
                rows = copy.deepcopy(self.rows)
                row = next(row for row in rows if row.get("condition") == "word")
                row["hook_count"] = count
                row["result"]["hook_count"] = count
                row["valid_hook"] = False
                with self.assertRaisesRegex(ValueError, "hook did not fire once"):
                    self.check_rows(rows)

    def test_trace_recomputation_rejects_corrupt_sum(self) -> None:
        captures = {}
        mel_zero = np.zeros((1, 80, 3000), dtype=np.float32)
        mel_one = np.ones((1, 80, 3000), dtype=np.float32)
        for engine in audit.WINDOWS:
            captures[f"{engine}_silver"] = {"mel": mel_zero, **dict.fromkeys(audit.CAPTURES, self.silver)}
            captures[f"{engine}_teal"] = {"mel": mel_one, **dict.fromkeys(audit.CAPTURES, self.teal)}
        (self.output / "captures").mkdir()
        rows = []
        for engine in audit.WINDOWS:
            details = {}
            for site in ("mel", *audit.CAPTURES):
                silver = captures[f"{engine}_silver"][site]
                teal = captures[f"{engine}_teal"][site]
                delta = np.linalg.norm(silver - teal, axis=1 if site == "mel" else -1)[0]
                word, prefix = audit.WINDOWS[engine]
                if site == "mel":
                    word = tuple(value * 2 for value in word)
                    prefix = tuple(value * 2 for value in prefix)
                mask = np.ones(len(delta), dtype=bool)
                mask[slice(*word)] = False
                mask[slice(*prefix)] = False
                relative = f"captures/{engine}_{site}_silver_teal_l2.npy"
                path = self.output / relative
                np.save(path, delta, allow_pickle=False)
                details[site] = {"path": relative, "sha256": audit.sha(path),
                                 "shape": list(delta.shape), "word_window": list(word),
                                 "prefix_window": list(prefix),
                                 "word_l2_sum": float(delta[slice(*word)].sum()),
                                 "prefix_l2_sum": float(delta[slice(*prefix)].sum()),
                                 "remaining_l2_sum": float(delta[mask].sum()),
                                 "first_nonzero_frame": 0}
            rows.append({"type": "trace", "engine": engine,
                         "words": list(audit.WORDS), "traces": details})
        self.assertEqual(set(audit.audit_traces(rows, self.output, captures)), set(audit.WINDOWS))
        rows[0]["traces"]["post_norm"]["word_l2_sum"] += 20
        with self.assertRaisesRegex(ValueError, "word_l2_sum"):
            audit.audit_traces(rows, self.output, captures)

    def test_failed_control_is_preserved_without_outcome(self) -> None:
        rows = copy.deepcopy(self.rows[:8])
        row = next(row for row in self.rows if row.get("condition") == "self")
        row = copy.deepcopy(row)
        row["result"]["token_ids"][1] = 999
        row["control_pass"] = False
        rows.append(row)
        rows.append({"type": "failure", "error_type": "ValueError",
                     "error": "self patch changed tokens", "elapsed_seconds": 1.0})
        report = audit.audit_failed_attempt(rows, self.output)
        self.assertEqual(report["status"], "invalid_or_incomplete_attempt")
        self.assertFalse(report["scientific_outcome_available"])
        self.assertEqual(len(report["explicit_control_failures"]), 1)

    def test_trusted_fit_pcm_pair(self) -> None:
        cases = audit.trusted_inputs()
        self.assertEqual(set(cases), self.cases.keys())


if __name__ == "__main__":
    unittest.main()
