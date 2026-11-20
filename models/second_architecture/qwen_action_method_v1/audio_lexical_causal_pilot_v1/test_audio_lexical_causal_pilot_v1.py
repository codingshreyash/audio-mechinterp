"""Source-only negative tests. These tests never load model weights."""
from __future__ import annotations

import hashlib
import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from . import plan, saved_audit, worker


class TinyTokenizer:
    all_special_ids = [plan.EOS_ID, 999]

    def decode(self, ids, **_kwargs):
        return "".join({1: "red", 2: "blue", 999: "<special>",
                        plan.EOS_ID: "<eos>"}.get(i, "?") for i in ids)


class SourceOnlyAuditTests(unittest.TestCase):
    def test_cross_platform_processor_rule_preserves_geometry_and_local_support(self):
        saved = json.loads(plan.PROCESSOR_RECEIPT.read_text())
        changed = copy.deepcopy(saved)
        changed["records"]["calibrated"]["red"]["tensor_sha256"]["input_features"] = "0" * 64
        plan.verify_runtime_processor_receipt(changed)
        wrong_ids = copy.deepcopy(changed)
        wrong_ids["records"]["calibrated"]["red"]["prompt_ids"][0] += 1
        with self.assertRaises(ValueError):
            plan.verify_runtime_processor_receipt(wrong_ids)
        wrong_window = copy.deepcopy(changed)
        wrong_window["comparison"]["calibrated"]["changed_feature_columns"] = 42
        with self.assertRaises(ValueError):
            plan.verify_runtime_processor_receipt(wrong_window)
        wrong_floor = copy.deepcopy(changed)
        wrong_floor["comparison"]["calibrated"]["blue_padding_floor"] += 0.01
        with self.assertRaises(ValueError):
            plan.verify_runtime_processor_receipt(wrong_floor)

    def test_complete_path_grading_requires_clean_terminal_eos(self):
        tokenizer = TinyTokenizer()
        self.assertEqual(worker.classify([1, plan.EOS_ID], tokenizer)["label"], "red")
        self.assertIsNone(worker.classify([999, 1, plan.EOS_ID], tokenizer)["label"])
        self.assertEqual(worker.classify([1]*plan.MAX_NEW_TOKENS, tokenizer)
                         ["stop_reason"], "length_censored")
        self.assertEqual(worker.classify([1, plan.EOS_ID, 2], tokenizer)
                         ["stop_reason"], "early_eos_malformed")

    def test_jsonl_duplicate_key_and_empty_line_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/"rows.jsonl"
            p.write_text('{"x":1,"x":2}\n')
            with self.assertRaises(ValueError):
                saved_audit.read_jsonl(p)
            p.write_text('{"x":1}\n\n')
            with self.assertRaises(ValueError):
                saved_audit.read_jsonl(p)

    def test_sidecar_requires_exact_name_bytes_and_finite_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            value = np.array([1.0, 2.0], dtype=np.float32)
            meta = worker.save_array(folder, "known", value)
            np.testing.assert_array_equal(saved_audit.load_sidecar(folder, meta,
                                                                    "known.npy"), value)
            with self.assertRaises(ValueError):
                saved_audit.load_sidecar(folder, meta, "../known.npy")
            (folder/"known.npy").write_bytes(b"tampered")
            with self.assertRaises(ValueError):
                saved_audit.load_sidecar(folder, meta, "known.npy")

    def test_clean_exit_boundary_must_follow_last_sample(self):
        result = {"heartbeat_clean_exit_boundary": True,
                  "heartbeat_boundary_elapsed_seconds": 10.2,
                  "worker_elapsed_seconds": 10.3,
                  "worker_returncode": 0,
                  "worker_stop_reason": "normal_exit"}
        saved_audit.verify_heartbeat_boundary(result, 10.0)
        with self.assertRaises(ValueError):
            saved_audit.verify_heartbeat_boundary(result, 10.21)
        result["worker_returncode"] = 1
        with self.assertRaises(ValueError):
            saved_audit.verify_heartbeat_boundary(result, 10.0)

    def test_intervention_control_norm_tamper_fails(self):
        tokenizer = TinyTokenizer()
        states = {word: {} for word in plan.WORDS}
        for site in plan.SITES:
            w = plan.WINDOWS[site]
            shape = (w["total"], w["width"])
            states["red"][site] = np.zeros(shape, dtype=np.float32)
            states["blue"][site] = np.zeros(shape, dtype=np.float32)
            lo, hi = w["local"]
            states["blue"][site][lo:hi, 0] = 1.0
        baselines = {"calibrated": {"red": worker.classify([1, plan.EOS_ID], tokenizer),
                                    "blue": worker.classify([2, plan.EOS_ID], tokenizer)}}
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            rows = []
            for si, site in enumerate(plan.SITES):
                w = plan.WINDOWS[site]
                norm = float(np.sqrt(w["local"][1]-w["local"][0]))
                for di, recipient in enumerate(plan.WORDS):
                    donor = "blue" if recipient == "red" else "red"
                    source = states[recipient][site]
                    target = states[donor][site]
                    lo, hi = w["local"]
                    local_delta = target[lo:hi]-source[lo:hi]
                    for arm in plan.ARMS:
                        window = (w["relocated"] if arm == "relocated_delta" else
                                  w["chunk"] if arm == "chunk_donor" else
                                  [0, w["total"]] if arm in ("full_donor", "self") else w["local"])
                        effective = np.zeros_like(source)
                        a, b = window
                        if arm in ("local_donor", "chunk_donor", "full_donor"):
                            effective[a:b] = target[a:b]-source[a:b]
                        elif arm == "relocated_delta":
                            effective[a:b] = local_delta
                        elif arm == "equal_l2_noise":
                            seed = plan.NOISE_SEED+si*100+di*10
                            noise = np.random.default_rng(seed).standard_normal(
                                tuple(local_delta.shape)).astype(np.float32)
                            effective[a:b] = noise * (norm/float(np.linalg.norm(noise)))
                        arm_id = f"{si}_{di}_{arm}"
                        nonce = f"{len(rows)+1:032x}"
                        sidecar = worker.save_array(folder,
                            f"effective_delta_{arm_id}_{nonce}", effective)
                        output_word = donor if arm == "full_donor" else recipient
                        rows.append({"site": site, "recipient": recipient, "donor": donor,
                                     "arm": arm, "arm_id": arm_id, "call_nonce": nonce,
                                     "prepatch_state_sha256": hashlib.sha256(
                                         source.tobytes()).hexdigest(),
                                     "outside_window_maxabs": 0.0,
                                     "effective_delta": sidecar,
                                     "output": worker.classify(
                                         [1 if output_word == "red" else 2,
                                          plan.EOS_ID], tokenizer),
                                     "hook_calls": 1, "source_drift_maxabs": 0.0,
                                     "source_drift_l2": 0.0,
                                     "local_delta_l2": norm,
                                     "realized_edit_l2": float(np.linalg.norm(effective)),
                                     "seed": (plan.NOISE_SEED+si*100+di*10)
                                             if arm == "equal_l2_noise" else None,
                                     "window": window})
            payload = {"interventions": rows, "baselines": baselines}
            saved_audit.verify_interventions(folder, payload, states, tokenizer)
            rows[0]["hook_calls"] = 0
            with self.assertRaises(ValueError):
                saved_audit.verify_interventions(folder, payload, states, tokenizer)
            rows[0]["hook_calls"] = 1
            control = next(row for row in rows if row["arm"] == "relocated_delta")
            control["realized_edit_l2"] *= 0.5
            with self.assertRaises(ValueError):
                saved_audit.verify_interventions(folder, payload, states, tokenizer)
            control["realized_edit_l2"] *= 2.0
            # Preserve tensor shape and total norm while moving one edit to a
            # wrong coordinate; update the declared hash to test content audit.
            first = rows[0]
            sidecar_path = folder/first["effective_delta"]["path"]
            changed = np.load(sidecar_path, allow_pickle=False)
            lo = plan.WINDOWS[first["site"]]["local"][0]
            changed[lo, 0], changed[lo, 1] = changed[lo, 1], changed[lo, 0]
            with sidecar_path.open("wb") as stream:
                np.save(stream, changed, allow_pickle=False)
            first["effective_delta"]["sha256"] = plan.sha(sidecar_path)
            with self.assertRaises(ValueError):
                saved_audit.verify_interventions(folder, payload, states, tokenizer)


if __name__ == "__main__":
    unittest.main()
