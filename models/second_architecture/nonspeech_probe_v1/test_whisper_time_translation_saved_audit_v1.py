"""No-model tests for strict saved-evidence interpretation and patch audit."""
from __future__ import annotations

import unittest
import tempfile
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from . import whisper_tail_causal_saved_audit_v3 as prior
from . import whisper_time_translation_plan_v1 as plan
from . import whisper_time_translation_saved_audit_v1 as audit
from . import whisper_time_translation_worker_v1 as worker


def checked(token: int, raw: int) -> dict:
    return {"output": {"termination": "eos", "eos_seen": True,
                       "censored": False, "sequence_ids": [token, 50257]},
            "raw": np.array([raw], dtype=np.float32)}


def row(layout: str, site: str, region: str, location: str, kind: str,
        output: dict, norm: float = 1.0) -> dict:
    return {"plan": {"type": "arm", "layout": layout, "site": site,
                     "region": region, "location": location, "kind": kind},
            "checked": output, "patch": {"inside_delta_l2": norm}}


class SavedAuditTests(unittest.TestCase):
    def test_frozen_wav_full_mel_binding_rejects_boundary_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pcm = np.arange(16, dtype="<i2").tobytes()
            wav = root / "frozen.wav"
            with wave.open(str(wav), "wb") as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(16_000)
                output.writeframes(pcm)
            item = {"input": {"wav": wav.name, "frames": 16,
                              "wav_sha256": audit.gate.sha(wav),
                              "pcm_sha256": worker.hashlib.sha256(pcm).hexdigest()}}
            mel = np.zeros((80, 3000), dtype=np.float32)
            mask = np.zeros(3000, dtype=np.int64)
            class FakeProcessor:
                def __call__(self, samples, **kwargs):
                    return SimpleNamespace(input_features=mel[None],
                                           attention_mask=mask[None])
            checked = {"mel": mel.copy(), "mask": mask.copy()}
            audit.verify_baseline_source(item, checked, FakeProcessor(), root)
            checked["mel"][0, 1000] = 0.01  # Outside the predeclared interior checks.
            with self.assertRaisesRegex(ValueError, "full mel/mask"):
                audit.verify_baseline_source(item, checked, FakeProcessor(), root)
            checked["mel"][0, 1000] = 0
            with wav.open("ab") as stream:
                stream.write(b"x")
            with self.assertRaisesRegex(ValueError, "WAV hash"):
                audit.verify_baseline_source(item, checked, FakeProcessor(), root)

    def test_joint_decision_requires_original_isolated_viability(self) -> None:
        identity = {"frozen": "same"}
        original = {"schema": "whisper_time_translation_saved_audit_v1",
                    "status": "PASS", "stage": "original", "plan_sha256": "a",
                    "source_identity": identity,
                    "summary": {"original_qualified": {
                        "post_conv_pos:union": True, "post_block0:union": True,
                        "post_conv_pos:source_tail": False,
                        "post_block0:source_tail": False,
                        "post_conv_pos:tail_end": True,
                        "post_block0:tail_end": False}}}
        def translated(stage: str) -> dict:
            return {"schema": original["schema"], "status": "PASS", "stage": stage,
                    "plan_sha256": "a", "source_identity": identity,
                    "summary": {"translated_cells": [
                        {"site": site, "region": region,
                         "strict_translated_crossover": True}
                        for region, site in plan.TRANSLATED_WINDOWS]}}
        result = audit._join_verified_reports(original, translated("offset_8000"),
                                    translated("offset_16000"))
        self.assertEqual(result["union_qualified_count_out_of_4"], 4)
        self.assertEqual(result["isolated_source_tail_qualified_count_out_of_2"], 0)
        original["summary"]["original_qualified"]["post_conv_pos:source_tail"] = True
        result = audit._join_verified_reports(original, translated("offset_8000"),
                                    translated("offset_16000"))
        self.assertEqual(result["isolated_source_tail_qualified_count_out_of_2"], 2)

    def test_joint_entrypoint_reaudits_fixed_raw_attempts(self) -> None:
        with mock.patch.object(audit, "audit", side_effect=ValueError("tampered raw evidence")) as rerun:
            with self.assertRaisesRegex(ValueError, "tampered raw evidence"):
                audit.audit_joint()
            rerun.assert_called_once_with("original", audit.gate.ADMISSIONS["original"],
                                          audit.gate.GOS["original"])

    def test_no_contrast_skips_all_crossover_claims(self) -> None:
        baselines = {(layout, kind): checked(10 if kind == "zero" else 20, 1)
                     for layout in ("prepend", "append") for kind in ("zero", "noise")}
        result = audit.summarize("offset_8000", [], [], baselines,
            {"prepend": {"eligible": False}, "append": {"eligible": True}},
            {"pass_le_1e_5": True})
        self.assertTrue(result["translated_cells"])
        self.assertFalse(any(cell["strict_translated_crossover"]
                             for cell in result["translated_cells"]))

    def test_equal_norm_relocation_must_preserve_recipient(self) -> None:
        baselines = {(layout, kind): checked(10 if kind == "zero" else 20,
                                             10 if kind == "zero" else 20)
                     for layout in ("prepend", "append") for kind in ("zero", "noise")}
        rows = []
        for layout, correct, opposite in (("prepend", "translated", "fixed"),
                                          ("append", "fixed", "translated")):
            donor, recipient = baselines[(layout, "zero")], baselines[(layout, "noise")]
            for kind, outcome in (("target_transfer", donor), ("identity_sham", recipient),
                                  ("source_random", recipient),
                                  ("norm_matched_shifted_delta", recipient)):
                rows.append(row(layout, "post_conv_pos", "union", correct, kind, outcome))
            rows.append(row(layout, "post_conv_pos", "all", "fixed", "full_donor", donor))
            rows.append(row(layout, "post_conv_pos", "union", opposite,
                            "target_transfer", recipient))
            rows.append(row(layout, "post_conv_pos", "union", opposite,
                            "relocated_correct_delta", recipient))
        eligible = {layout: {"eligible": True} for layout in ("prepend", "append")}
        result = audit.summarize("offset_8000", [], rows, baselines,
                                 eligible, {"pass_le_1e_5": True})
        union = next(cell for cell in result["translated_cells"]
                     if cell["site"] == "post_conv_pos" and cell["region"] == "union")
        self.assertTrue(union["strict_translated_crossover"])
        relocated = next(item for item in rows if item["plan"]["layout"] == "append"
                         and item["plan"]["kind"] == "relocated_correct_delta")
        relocated["checked"] = baselines[("append", "zero")]
        result = audit.summarize("offset_8000", [], rows, baselines,
                                 eligible, {"pass_le_1e_5": True})
        union = next(cell for cell in result["translated_cells"]
                     if cell["site"] == "post_conv_pos" and cell["region"] == "union")
        self.assertFalse(union["strict_translated_crossover"])

    def test_relocated_delta_patch_rejects_mutated_competing_row(self) -> None:
        arm = next(item for item in plan.arms_for("offset_8000")
                   if item["kind"] == "relocated_correct_delta")
        mapped = worker.execution_arm(arm)
        recipient = np.zeros((1500, 384), dtype=np.float32)
        donor = recipient.copy()
        donor[mapped["target_rows"], 0] = 1.0
        edited = recipient.copy()
        edited[mapped["source_rows"], 0] = 1.0
        selected = mapped["source_rows"]
        delta = float(np.linalg.norm(edited[selected].astype(np.float64)))
        receipt = {"hook_calls": 1, "selected_rows": selected,
                   "donor_state_sha256": worker.hashlib.sha256(donor.tobytes()).hexdigest(),
                   "recipient_state_sha256": worker.hashlib.sha256(recipient.tobytes()).hexdigest(),
                   "inside_delta_l2": delta, "target_delta_l2": delta,
                   "outside_maxabs": 0.0}
        prior.verify_patch(recipient, edited, recipient, donor, mapped, receipt)
        edited[selected[0], 0] = 0.0
        with self.assertRaisesRegex(ValueError, "edited hook rows"):
            prior.verify_patch(recipient, edited, recipient, donor, mapped, receipt)


if __name__ == "__main__":
    unittest.main()
