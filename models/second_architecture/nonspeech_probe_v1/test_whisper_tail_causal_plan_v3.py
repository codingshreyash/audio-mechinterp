"""No-weight preregistration checks for the cross-family Whisper patch plan."""
from __future__ import annotations

import json
import unittest
from unittest import mock

from . import whisper_tail_causal_plan_v3 as plan


class CausalPlanNoWeightTests(unittest.TestCase):
    def test_frozen_pair_and_arm_census(self) -> None:
        value=plan.build()
        self.assertEqual(value["status"],"source_only_no_inference")
        self.assertEqual(value["generation_calls_total"],56)
        self.assertEqual(len(value["arms"]),54)
        self.assertEqual([(u["model"],u["v4_changed_sequence"]) for u in value["units"]],
            [("whisper_multilingual",True)])
        self.assertEqual(len(value["regions"]["boundary"]),10)
        self.assertEqual(len(value["regions"]["tail_interior"]),95)
        for unit in value["units"]:
            self.assertEqual(unit["case_id"],"d08_band_noise_low")
            self.assertEqual(unit["tail_samples"],32000)
            self.assertEqual(unit["seed"],0)
            self.assertEqual(unit["zero"]["termination"],"eos")
            self.assertEqual(unit["noise"]["termination"],"eos")
            self.assertEqual(unit["zero"]["prefix_ids"],unit["noise"]["prefix_ids"])
            self.assertEqual(unit["zero"]["sequence_ids"],
                             [50258,50259,50359,50363,291,50257])
            self.assertEqual(unit["noise"]["sequence_ids"],
                             [50258,50259,50359,50363,1044,291,13,50257])
            for region,rows in value["regions"].items():
                control=unit["random_source_rows"][region]
                self.assertEqual(len(control),len(rows))
                self.assertEqual(len(control),len(set(control)))
                self.assertTrue(set(control).isdisjoint(rows))

    def test_four_local_and_full_donor_controls_each_site(self) -> None:
        arms=plan.build()["arms"]
        for model in ("whisper_multilingual",):
            for site in plan.SITES:
                full=[a for a in arms if a["model"]==model and a["site"]==site
                      and a["kind"]=="full_donor"]
                self.assertEqual(len(full),2)
                self.assertTrue(all(len(a["target_rows"])==1500 for a in full))
                for region in plan.REGIONS:
                    for recipient in ("zero","noise"):
                        kinds={a["kind"] for a in arms if a["model"]==model
                               and a["site"]==site and a["region"]==region
                               and a["recipient"]==recipient}
                        self.assertEqual(kinds,set(plan.KINDS))

    def test_saved_status_and_mask_faults_reject(self) -> None:
        original=json.loads
        def bad_audit(data,*args,**kwargs):
            value=original(data,*args,**kwargs)
            if isinstance(value,dict) and value.get("schema")=="whisper_natural_tail_saved_audit_v4":
                value["status"]="HOLD"
            return value
        with mock.patch.object(plan.json,"loads",side_effect=bad_audit):
            with self.assertRaisesRegex(ValueError,"audit not passed"):
                plan.build()

    def test_changed_prior_v2_receipt_rejects(self) -> None:
        from pathlib import Path
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as temp:
            fake=Path(temp)/"audit.json"
            fake.write_text('{"status":"PASS","full_replication":true}\n')
            with mock.patch.object(plan,"PRIOR_V2_AUDIT",fake):
                with self.assertRaisesRegex(ValueError,"prior V2 causal result changed"):
                    plan.build()


if __name__=="__main__":
    unittest.main()
