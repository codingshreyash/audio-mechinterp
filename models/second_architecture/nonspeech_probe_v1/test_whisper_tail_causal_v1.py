"""No-weight fault tests for the bounded Whisper causal pilot."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest import mock

import numpy as np

from . import whisper_tail_causal_parent_v1 as parent
from . import whisper_tail_causal_saved_audit_v1 as auditor
from . import whisper_tail_causal_worker_v1 as worker


class CausalNoWeightTests(unittest.TestCase):
    def test_plan_census_and_control_support(self):
        frozen=json.loads(worker.plan_source.PLAN_PATH.read_text())
        self.assertEqual(frozen,worker.plan_source.build())
        self.assertEqual(len(frozen["arms"]),108)
        for model in worker.v4.MODELS:
            rows=[a for a in frozen["arms"] if a["model"]==model]
            self.assertEqual(len(rows),54)
            self.assertEqual(sum(a["kind"]=="full_donor" for a in rows),6)
            for a in rows:
                if a["kind"]=="norm_matched_shifted_delta":
                    self.assertEqual(len(a["target_rows"]),len(a["source_rows"]))
                    self.assertFalse(set(a["target_rows"]) & set(a["source_rows"]))

    def test_patch_reconstruction_rejects_changed_unselected_row(self):
        native=np.zeros((1500,384),dtype=np.float32)
        donor=native.copy();donor[398:403]=1
        call={"kind":"target_transfer","target_rows":list(range(398,403)),"source_rows":[]}
        edited=native.copy();edited[398:403]=1
        receipt={"hook_calls":1,"selected_rows":call["target_rows"],
                 "donor_state_sha256":worker.hashlib.sha256(donor.tobytes()).hexdigest(),
                 "recipient_state_sha256":worker.hashlib.sha256(native.tobytes()).hexdigest(),
                 "inside_delta_l2":float(np.linalg.norm((edited[398:403]-native[398:403]).astype(np.float64))),
                 "outside_maxabs":0.0}
        auditor.verify_patch(native,edited,native,donor,call,receipt)
        edited[397,0]=1
        with self.assertRaisesRegex(ValueError,"edited hook rows"):
            auditor.verify_patch(native,edited,native,donor,call,receipt)

    def test_missing_live_parent_rejects_before_weight_import(self):
        with tempfile.TemporaryDirectory() as tmp:
            attempt=Path(tmp);(attempt/"launch_start.json").write_text("{}")
            with mock.patch.dict(worker.ATTEMPTS,{"whisper_english":attempt}),\
                 mock.patch.object(worker.v4.old,"_snapshot_load",side_effect=AssertionError("weight load")):
                with self.assertRaises(ValueError):
                    worker.run("whisper_english",attempt,Path("fake"),Path("fake"),-1)

    def test_fake_encoder_hook_patch_and_lossless_sidecars(self):
        import torch
        class Encoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers=torch.nn.ModuleList([torch.nn.Identity()])
                self.layer_norm=torch.nn.Identity()
            def forward(self,x):return self.layer_norm(self.layers[0](x))
        fake=SimpleNamespace(model=SimpleNamespace(encoder=Encoder()))
        def infer(_name,_processor,model,pcm,_sidecars,_stem):
            model.model.encoder(torch.full((1,1500,384),float(pcm[0])))
            return {"output":{"sequence_ids":[]}},
        with tempfile.TemporaryDirectory() as tmp,\
             mock.patch.object(worker.v4,"infer_one",side_effect=infer):
            folder=Path(tmp);sidecars=folder/"sidecars";sidecars.mkdir()
            _,zero,_=worker.run_one("whisper_english",None,fake,np.array([0]),sidecars,"z",
                                     arm=None,states={})
            _,noise,_=worker.run_one("whisper_english",None,fake,np.array([1]),sidecars,"n",
                                      arm=None,states={})
            states={"zero":zero,"noise":noise}
            for site in worker.plan_source.SITES:
                arm={"recipient":"zero","donor":"noise","site":site,
                     "kind":"target_transfer","target_rows":list(range(398,403)),
                     "source_rows":list(range(8,13))}
                _,_,patch=worker.run_one("whisper_english",None,fake,np.array([0]),sidecars,
                                          site,arm=arm,states=states)
                native=auditor.state(folder,patch["native_ref"])
                edited=auditor.state(folder,patch["edited_ref"])
                auditor.verify_patch(native,edited,zero[site],noise[site],arm,patch)
                self.assertEqual(patch["hook_calls"],1)

    def test_exact_394_sidecar_complete_census(self):
        with tempfile.TemporaryDirectory() as tmp:
            attempt=Path(tmp);(attempt/"sidecars").mkdir()
            frozen=json.loads(worker.plan_source.PLAN_PATH.read_text())
            model="whisper_english"
            calls=[{"type":"baseline","variant":v} for v in ("zero","noise")]+[
                {"type":"arm",**a} for a in frozen["arms"] if a["model"]==model]
            records=[];progress=[];names=[]
            for index,call in enumerate(calls,1):
                progress.extend([{"type":"attempt","index":index},{"type":"complete","index":index}])
                refs={k:{"file":f"{index}_{k}.npy"} for k in (
                    "mel","mask","first_raw_logits","first_processed_scores","first_decoder_state")}
                names.extend(ref["file"] for ref in refs.values())
                row={"type":"call","index":index,"plan":call,"output":{"sidecars":refs}}
                if index<=2:
                    row["state_refs"]={k:{"file":f"{index}_{k}.npy"} for k in worker.plan_source.SITES}
                    names.extend(ref["file"] for ref in row["state_refs"].values())
                else:
                    row["patch"]={k:{"file":f"{index}_{k}.npy"} for k in ("native_ref","edited_ref")}
                    names.extend(ref["file"] for ref in row["patch"].values())
                records.append(row)
            self.assertEqual(len(names),394)
            for name in names:(attempt/"sidecars"/name).write_bytes(b"x")
            (attempt/"records.jsonl").write_text("\n".join(json.dumps(x) for x in
                [*records,{"type":"complete","calls":56}])+"\n")
            (attempt/"progress.jsonl").write_text("\n".join(json.dumps(x) for x in progress)+"\n")
            self.assertTrue(parent.complete_worker_output(attempt,model))
            (attempt/"sidecars"/names[-1]).unlink()
            self.assertFalse(parent.complete_worker_output(attempt,model))

    def test_async_probe_does_not_block_heartbeat_or_rss(self):
        with tempfile.TemporaryDirectory() as tmp:
            attempt=Path(tmp)
            start=time.monotonic();samples=[{"monotonic_seconds":start}]
            def slow_probe(*,initial=False):
                time.sleep(2.3)
                return {"monotonic_seconds":time.monotonic(),"battery_percent":99,
                        "lid_open":True,"thermal_ok":True}
            with (attempt/"host_samples.jsonl").open("x") as host_out,\
                 mock.patch.object(parent.host,"quick_probe",side_effect=slow_probe):
                result=parent.run_bounded([sys.executable,"-c","import time;time.sleep(3.0)"],
                    attempt,"whisper_english",nonce="a"*32,start=start,deadline=start+1800,
                    samples=samples,host_out=host_out)
            self.assertEqual(result["worker_stop_reason"],"normal_exit")
            self.assertGreater(result["parent_heartbeat_count"],8)
            self.assertGreater(result["worker_rss_samples"],8)
            self.assertEqual(len(samples),2)

    def test_exiting_worker_does_not_discard_failing_pending_probe(self):
        with tempfile.TemporaryDirectory() as tmp:
            attempt=Path(tmp)
            start=time.monotonic();samples=[{"monotonic_seconds":start}]
            def failing_probe(*,initial=False):
                time.sleep(0.3)
                raise ValueError("battery unreadable")
            with (attempt/"host_samples.jsonl").open("x") as host_out,\
                 mock.patch.object(parent.host,"quick_probe",side_effect=failing_probe):
                result=parent.run_bounded([sys.executable,"-c","import time;time.sleep(2.15)"],
                    attempt,"whisper_english",nonce="a"*32,start=start,deadline=start+1800,
                    samples=samples,host_out=host_out)
            self.assertEqual(result["worker_stop_reason"],"host_probe_failed")
            self.assertIn("battery unreadable",result["worker_monitor_error"])


if __name__=="__main__":unittest.main()
