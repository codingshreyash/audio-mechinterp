"""No-weight preregistration checks for the staged Whisper patch plan."""
from __future__ import annotations

import json
import unittest
from unittest import mock

from . import whisper_tail_causal_plan_v1 as plan


class CausalPlanNoWeightTests(unittest.TestCase):
    def test_frozen_pair_and_arm_census(self) -> None:
        value=plan.build()
        self.assertEqual(value["status"],"source_only_no_inference")
        self.assertEqual(value["generation_calls_total"],112)
        self.assertEqual(len(value["arms"]),108)
        self.assertEqual([(u["model"],u["v4_changed_sequence"]) for u in value["units"]],
            [("whisper_multilingual",True),("whisper_english",False)])
        self.assertEqual(len(value["regions"]["boundary"]),10)
        self.assertEqual(len(value["regions"]["tail_interior"]),95)
        for unit in value["units"]:
            self.assertEqual(unit["case_id"],"d08_beep_train_low")
            self.assertEqual(unit["tail_samples"],32000)
            self.assertEqual(unit["seed"],0)
            self.assertEqual(unit["zero"]["termination"],"eos")
            self.assertEqual(unit["noise"]["termination"],"eos")
            for region,rows in value["regions"].items():
                control=unit["random_source_rows"][region]
                self.assertEqual(len(control),len(rows))
                self.assertEqual(len(control),len(set(control)))
                self.assertTrue(set(control).isdisjoint(rows))

    def test_four_local_and_full_donor_controls_each_site(self) -> None:
        arms=plan.build()["arms"]
        for model in plan.v4.MODELS:
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


if __name__=="__main__":
    unittest.main()
