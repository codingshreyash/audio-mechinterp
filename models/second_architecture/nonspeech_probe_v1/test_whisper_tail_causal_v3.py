"""No-weight V3 cross-family admission and method-inheritance regressions."""
from __future__ import annotations

import ast
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from . import whisper_tail_causal_plan_v3 as plan
from . import whisper_tail_causal_worker_v1 as prior_worker
from . import whisper_tail_causal_worker_v3 as worker
from . import whisper_tail_causal_parent_v1 as prior_parent
from . import whisper_tail_causal_parent_v3 as parent
from . import whisper_tail_causal_saved_audit_v1 as prior_audit
from . import whisper_tail_causal_saved_audit_v3 as auditor


def function_ast(module,name:str)->str:
    tree=ast.parse(Path(module.__file__).read_text())
    found=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name==name)
    return ast.dump(found,include_attributes=False)


class CrossFamilyNoWeightTests(unittest.TestCase):
    def test_frozen_band_noise_plan_and_prior_v1_v2_binding(self):
        frozen=json.loads(plan.PLAN_PATH.read_text())
        self.assertEqual(frozen,plan.build())
        self.assertEqual(len(frozen["units"]),1)
        self.assertEqual(frozen["units"][0]["seed"],0)
        self.assertEqual(frozen["units"][0]["case_id"],"d08_band_noise_low")
        self.assertIn("n32000_s0",frozen["units"][0]["noise"]["wav"])
        self.assertEqual(len(frozen["arms"]),54)
        self.assertEqual(len(frozen["primary_cross_family_cells"]),3)
        old_plan=json.loads(prior_worker.plan_source.PLAN_PATH.read_text())
        self.assertEqual(frozen["units"][0]["random_source_rows"],
                         old_plan["units"][0]["random_source_rows"])
        self.assertEqual(frozen["inputs"]["prior_v1_audit_sha256"],plan.sha(plan.PRIOR_AUDIT))
        self.assertEqual(frozen["inputs"]["prior_v1_interpretation_sha256"],plan.sha(plan.PRIOR_MEMO))
        self.assertEqual(frozen["inputs"]["prior_v2_audit_sha256"],plan.sha(plan.PRIOR_V2_AUDIT))
        self.assertEqual(frozen["inputs"]["prior_v2_historical_sha256"],plan.sha(plan.PRIOR_V2_HISTORICAL))

    def test_inherited_scientific_and_monitor_methods_unchanged(self):
        for name in ("_site_module","run_one","_pcm","_save_array"):
            self.assertEqual(function_ast(worker,name),function_ast(prior_worker,name),name)
        for name in ("run_bounded","progress_census","complete_worker_output"):
            self.assertEqual(function_ast(parent,name),function_ast(prior_parent,name),name)
        for name in ("state","verify_patch","validate_whisper_record"):
            self.assertEqual(function_ast(auditor,name),function_ast(prior_audit,name),name)

    def test_direct_worker_without_live_pipe_rejects_before_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            attempt=Path(tmp);(attempt/"launch_start.json").write_text("{}")
            with mock.patch.dict(worker.ATTEMPTS,{"whisper_multilingual":attempt}),\
                 mock.patch.object(worker.v4.old,"_snapshot_load",side_effect=AssertionError("weights")):
                with self.assertRaises(ValueError):
                    worker.run("whisper_multilingual",attempt,Path("fake"),Path("fake"),-1)

    def test_stale_v1_admission_go_rejected(self):
        with self.assertRaisesRegex(ValueError,"fixed causal admission/GO paths"):
            worker.validate_go("whisper_multilingual",
                prior_worker.ADMISSIONS["whisper_multilingual"],
                prior_worker.GOS["whisper_multilingual"])


if __name__=="__main__":unittest.main()
