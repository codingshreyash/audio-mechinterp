"""Independent corruption tests for the FIT spread saved-result auditor."""

from __future__ import annotations

import copy
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from models.second_architecture import whisper_fit_spread_causal_audit_v1 as audit


def fake_result(pair_id: str, word: str, hook_count: int | None = None) -> dict:
    candidates = {"silver": 100, "teal": 101, "plate": 102, "cup": 103}
    selected = candidates[word]
    a, b = audit.PAIRS[pair_id]["words"]
    field = f"generation_processed_{a}_minus_{b}_first_token_score"
    scores = [{"token_id": token, "selected_score": 1.0,
               "top2_ids": [token, 200], "top2_scores": [1.0, 0.0],
               field: None, "alias_margin_status": "multi_token_alias"}
              for token in (selected, audit.EOS_ID)]
    return {"token_ids": [50257, selected, audit.EOS_ID],
            "transcript": f"Say {word} now.", "eos_seen": True,
            "eos_terminated": True, "censored": False,
            "step_scores": scores, "elapsed_seconds": 0.1,
            "rss_bytes": 100, "hook_count": hook_count}


class SpreadSavedAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name)
        (self.output / "records.jsonl").write_text("{}\n")
        self.cases = {}
        self.states = {}
        self.hashes = {}
        rows = [{"type": "preflight"}]
        for pair_id, pair in audit.PAIRS.items():
            for engine in audit.ENGINES:
                for position, word in enumerate(pair["words"]):
                    key = audit.case_key(pair_id, engine, word)
                    self.cases[key] = {"pair_id": pair_id, "engine": engine,
                                       "word": word}
                    self.states[key] = np.full(audit.STATE_SHAPE, position,
                                               dtype=np.float32)
                    self.hashes[key] = {site: hashlib.sha256(
                        f"{key}/{site}".encode()).hexdigest()
                        for site in ("mel", *audit.CAPTURES)}
                    rows.append({"type": "baseline", "case": key,
                                 "pair_id": pair_id, "engine": engine,
                                 "result": fake_result(pair_id, word),
                                 "capture_sha256": self.hashes[key],
                                 "eligible": True, "capture_complete": True,
                                 "capture_hook_counts": dict.fromkeys(audit.CAPTURES, 1)})
        rows.extend({"type": "trace"} for _ in range(4))
        for pair_id, recipient, donor, site, condition, base_seed in sorted(audit.expected_matrix()):
            engine = self.cases[recipient]["engine"]
            rec_word = self.cases[recipient]["word"]
            donor_word = self.cases[donor]["word"]
            output_word = donor_word if condition in {"word", "full_donor"} else rec_word
            result = fake_result(pair_id, output_word, hook_count=1)
            baseline = fake_result(pair_id, rec_word)
            outcome = audit.recompute_outcome(result["transcript"], rec_word,
                                              donor_word, baseline["transcript"])
            outcome["token_difference_count"] = int(output_word != rec_word)
            source = recipient if condition == "self" else donor
            row = {"type": "intervention", "pair_id": pair_id, "engine": engine,
                   "recipient": recipient, "donor": donor, "site": site,
                   "condition": condition,
                   "region_spans": audit.region_spans(pair_id, engine, condition),
                   "base_seed": base_seed, "arm_seed": None,
                   "patch_audit": {}, "result": result,
                   "hook_count": 1, "valid_hook": True,
                   "state_shape": list(audit.STATE_SHAPE),
                   "state_dtype": "float32", "finite_checked": True,
                   "recipient_state_sha256": self.hashes[recipient][site],
                   "source_state_sha256": self.hashes[source][site],
                   "control_pass": (True if condition in {"self", "full_donor"} else None),
                   "outcome": outcome}
            if condition == "random":
                row["arm_seed"] = audit.derived_arm_seed(
                    pair_id, engine, rec_word, donor_word, site, base_seed)
                row["patch_audit"] = self.random_patch(row)
            rows.append(row)
        rows.append({"type": "complete", "elapsed_seconds": 1.0,
                     "rss_bytes": 100})
        self.rows = rows

    def random_patch(self, row: dict) -> dict:
        import torch

        spans = audit.region_spans(row["pair_id"], row["engine"], "complement")
        width = sum(end - start for start, end in spans)
        target = torch.linalg.vector_norm(torch.ones((1, width, 384)))
        gen = torch.Generator(device="cpu").manual_seed(row["arm_seed"])
        noise = torch.randn((1, width, 384), generator=gen, dtype=torch.float32)
        noise *= target / torch.linalg.vector_norm(noise)
        return {"random_edit_sha256": hashlib.sha256(noise.numpy().tobytes()).hexdigest(),
                "random_edit_norm": float(torch.linalg.vector_norm(noise)),
                "random_target_norm": float(target)}

    def check_rows(self, rows: list[dict]) -> dict:
        def capture(path: Path, _hash: str) -> np.ndarray:
            key = next(key for key in self.cases if path.name.startswith(f"{key}_"))
            return self.states[key]

        with patch.object(audit, "load_capture", side_effect=capture), \
             patch.object(audit, "load_mel", return_value=np.zeros((1, 80, 3000), dtype=np.float32)), \
             patch.object(audit, "audit_traces", return_value={}):
            return audit.audit_rows(rows, self.output, self.cases)

    def test_frozen_input_bytes_and_matrix(self) -> None:
        self.assertEqual(len(audit.trusted_inputs()), 8)
        self.assertEqual(len(audit.expected_matrix()), 120)
        result = self.check_rows(self.rows)
        self.assertEqual((result["baseline_count"], result["intervention_count"]),
                         (8, 120))
        self.assertEqual((result["random_unique_seed_count"],
                          result["random_unique_edit_count"]), (32, 32))

    def test_duplicate_intervention_rejected(self) -> None:
        rows = copy.deepcopy(self.rows)
        rows[-2] = copy.deepcopy(rows[-3])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.check_rows(rows)

    def test_corrupt_complement_span_rejected(self) -> None:
        rows = copy.deepcopy(self.rows)
        arm = next(row for row in rows if row.get("condition") == "complement")
        arm["region_spans"][1][0] += 1
        with self.assertRaisesRegex(ValueError, "region spans changed"):
            self.check_rows(rows)

    def test_postword_span_rejected(self) -> None:
        rows = copy.deepcopy(self.rows)
        arm = next(row for row in rows if row.get("condition") == "postword")
        arm["region_spans"] = [[0, 1]]
        with self.assertRaisesRegex(ValueError, "region spans changed"):
            self.check_rows(rows)

    def test_reused_random_seed_rejected(self) -> None:
        rows = copy.deepcopy(self.rows)
        arms = [row for row in rows if row.get("condition") == "random"]
        arms[1]["arm_seed"] = arms[0]["arm_seed"]
        with self.assertRaisesRegex(ValueError, "random arm seed differs"):
            self.check_rows(rows)

    def test_corrupt_random_edit_hash_rejected(self) -> None:
        rows = copy.deepcopy(self.rows)
        arm = next(row for row in rows if row.get("condition") == "random")
        arm["patch_audit"]["random_edit_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "random norm or edit hash"):
            self.check_rows(rows)

    def test_self_patch_token_corruption_rejected(self) -> None:
        rows = copy.deepcopy(self.rows)
        arm = next(row for row in rows if row.get("condition") == "self")
        arm["result"]["token_ids"][1] = 999
        arm["result"]["step_scores"][0]["token_id"] = 999
        arm["outcome"]["token_difference_count"] = 1
        with self.assertRaisesRegex(ValueError, "self-patch"):
            self.check_rows(rows)

    def test_duplicate_eos_cannot_count_as_complete(self) -> None:
        result = fake_result("color_05", "silver")
        result["token_ids"] = [50257, 100, audit.EOS_ID, audit.EOS_ID]
        result["step_scores"] = result["step_scores"] + [copy.deepcopy(result["step_scores"][-1])]
        result["eos_terminated"] = False
        result["censored"] = True
        checked = audit.checked_result(result, "color_05")
        self.assertFalse(checked["eos_terminated"])

    def make_trace_fixture(self) -> tuple[list[dict], dict]:
        captures = {}
        rows = []
        (self.output / "captures").mkdir(exist_ok=True)
        for pair_id, pair in audit.PAIRS.items():
            a, b = pair["words"]
            for engine in audit.ENGINES:
                spec = pair[engine]
                left, right = audit.case_key(pair_id, engine, a), audit.case_key(pair_id, engine, b)
                zero = np.zeros(audit.STATE_SHAPE, dtype=np.float32)
                pre = zero.copy()
                pre[:, slice(*spec["W"]), :] = 1
                spread = np.ones(audit.STATE_SHAPE, dtype=np.float32)
                mel_zero = np.zeros((1, 80, 3000), dtype=np.float32)
                mel_word = mel_zero.copy()
                mel_word[:, :, slice(*(value * 2 for value in spec["W"]))] = 1
                captures[left] = {"mel": mel_zero, **dict.fromkeys(audit.CAPTURES, zero)}
                captures[right] = {"mel": mel_word, **{
                    site: pre if site == "post_conv_pos" else spread
                    for site in audit.CAPTURES}}
                regions = {"W": list(spec["W"]), "P": list(spec["P"]),
                           "Q": list(spec["Q"]), "N": spec["N"],
                           "C": audit.region_spans(pair_id, engine, "complement"),
                           "slot_samples": list(spec["slot"])}
                details = {}
                for site in ("mel", *audit.CAPTURES):
                    difference = captures[left][site] - captures[right][site]
                    axis = 1 if site == "mel" else -1
                    l2 = np.linalg.norm(difference, axis=axis)[0]
                    maxabs = np.max(np.abs(difference), axis=axis)[0]
                    factor = 2 if site == "mel" else 1
                    scaled = {name: [value * factor for value in spec[name]]
                              for name in ("W", "P", "Q")}
                    c_spans = [[start * factor, end * factor] for start, end in regions["C"]]
                    name = f"captures/{pair_id}_{engine}_{site}_{a}_{b}"
                    l2_path, maxabs_path = (self.output / f"{name}_{suffix}.npy"
                                            for suffix in ("l2", "maxabs"))
                    np.save(l2_path, l2, allow_pickle=False)
                    np.save(maxabs_path, maxabs, allow_pickle=False)
                    details[site] = {
                        "path": f"{name}_l2.npy", "sha256": audit.sha(l2_path),
                        "maxabs_path": f"{name}_maxabs.npy",
                        "maxabs_sha256": audit.sha(maxabs_path), "shape": list(l2.shape),
                        "word_window": scaled["W"], "prefix_window": scaled["P"],
                        "postword_window": scaled["Q"],
                        "complement_spans": c_spans, "active_rows": spec["N"] * factor,
                        "word_l2_sum": float(l2[slice(*scaled["W"])].sum()),
                        "prefix_l2_sum": float(l2[slice(*scaled["P"])].sum()),
                        "postword_l2_sum": float(l2[slice(*scaled["Q"])].sum()),
                        "complement_l2_sum": float(sum(l2[start:end].sum()
                                                        for start, end in c_spans)),
                        "global_max_abs_difference": float(maxabs.max()),
                        "first_nonzero_frame": int(np.flatnonzero(l2 > 0)[0]),
                    }
                rows.append({"type": "trace", "pair_id": pair_id, "engine": engine,
                             "words": list(pair["words"]), "regions": regions,
                             "traces": details, "input_locality_pass": True,
                             "outside_preconv_equal": True, "word_preconv_differs": True})
        return rows, captures

    def test_trace_sidecars_and_input_locality_recomputed(self) -> None:
        rows, captures = self.make_trace_fixture()
        report = audit.audit_traces(rows, self.output, captures)
        self.assertEqual(len(report), 4)
        first = rows[0]
        first["traces"]["post_block0"]["complement_l2_sum"] += 1
        with self.assertRaisesRegex(ValueError, "complement_l2_sum"):
            audit.audit_traces(rows, self.output, captures)
        first["traces"]["post_block0"]["complement_l2_sum"] -= 1
        pair_id, engine = first["pair_id"], first["engine"]
        b = audit.PAIRS[pair_id]["words"][1]
        captures[audit.case_key(pair_id, engine, b)]["post_conv_pos"][:, 0, 0] = 1
        with self.assertRaisesRegex(ValueError, "input-locality"):
            audit.audit_traces(rows, self.output, captures)


if __name__ == "__main__":
    unittest.main()
