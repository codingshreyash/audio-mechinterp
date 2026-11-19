"""No-weight checks for frozen WAV schedule, signatures, and saved audit helpers."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from . import natural_tail_runner_v3 as runner
from . import natural_tail_parent_v3 as parent
from . import natural_tail_saved_audit_v3 as audit


class NaturalTailRunnerTests(unittest.TestCase):
    def test_frozen_320_call_schedule_and_wavs(self) -> None:
        english=runner.schedule("whisper_english")
        multi=runner.schedule("whisper_multilingual")
        self.assertEqual(len(english),len(multi),160)
        self.assertNotEqual(runner.ATTEMPTS["whisper_english"],
                            runner.ATTEMPTS["whisper_multilingual"])
        self.assertTrue(str(runner.ATTEMPTS["whisper_english"]).endswith(
            "whisper_english_attempt_v3"))
        self.assertEqual([x["index"] for x in english],list(range(1,161)))
        self.assertEqual(sum(x["replay"] for x in english),20)
        self.assertEqual(len({x["wav"] for x in english}),140)
        self.assertEqual(english,audit._expected_plan("whisper_english"))
        runner.verify_wavs(english)
        for base,replay in zip((x for x in english if x["condition"] == "base" and not x["replay"]),
                               english[140:],strict=True):
            self.assertEqual(base["wav_sha256"],replay["wav_sha256"])
            self.assertEqual(base["pcm_sha256"],replay["pcm_sha256"])
            self.assertEqual(base["wav"],replay["wav"])

    def test_installed_whisper_call_signatures_without_weights(self) -> None:
        from transformers import GenerationConfig
        from transformers.models.whisper.generation_whisper import WhisperGenerationMixin
        pins=json.loads(runner.old.PINS.read_text())["models"]
        for model in runner.MODELS:
            spec=pins[model]
            path=runner.old._snapshot(model,spec)
            config=GenerationConfig.from_pretrained(path,local_files_only=True)
            kwargs=runner.generation_kwargs(model,object(),object())
            self.assertIs(kwargs["do_sample"],False)
            self.assertEqual(kwargs["num_beams"],1)
            self.assertEqual(kwargs["max_new_tokens"],128)
            if model == "whisper_english":
                self.assertIs(config.is_multilingual,False)
                self.assertNotIn("task",kwargs)
                self.assertNotIn("language",kwargs)
                with self.assertRaisesRegex(ValueError,"Cannot specify `task` or `language`"):
                    WhisperGenerationMixin._set_language_and_task(
                        None,"transcribe",None,copy.deepcopy(config))
                WhisperGenerationMixin._set_language_and_task(
                    None,None,None,copy.deepcopy(config))
            else:
                self.assertIs(config.is_multilingual,True)
                self.assertEqual(kwargs["task"],"transcribe")
                self.assertIsNone(kwargs["language"])
                WhisperGenerationMixin._set_language_and_task(
                    None,kwargs["task"],kwargs["language"],copy.deepcopy(config))

    def test_progress_census_detects_partial_and_reordering(self) -> None:
        plan=[{"index":1},{"index":2}]
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/"progress.jsonl"
            self.assertEqual(parent.progress_census(path,plan)["not_attempted"],2)
            path.write_text('{"type":"attempt","index":1}\n')
            self.assertEqual(parent.progress_census(path,plan)["attempted_incomplete"],1)
            path.write_text('{"type":"attempt","index":2}\n')
            with self.assertRaisesRegex(ValueError,"order"):
                parent.progress_census(path,plan)

    def test_pre_host_probe_consumes_whole_attempt_worker_budget(self) -> None:
        clock_value=[100.0]
        def clock():return clock_value[0]
        def delayed_probe():
            clock_value[0] += 75.0
            return {"source_only_fake":True}
        started,deadline,_,host=parent.begin_host_window(
            probe=delayed_probe,clock=clock)
        self.assertEqual(started,100.0)
        self.assertEqual(deadline,1900.0)
        self.assertEqual(host,{"source_only_fake":True})
        self.assertEqual(parent.remaining_worker_seconds(deadline,now=175.0),
                         parent.remaining_worker_seconds(deadline,now=100.0)-75.0)

    def test_forged_direct_worker_has_no_live_parent_pipe(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            attempt=Path(temp)/"forged_attempt"
            attempt.mkdir()
            (attempt/"launch_start.json").write_text(json.dumps({
                "schema":"whisper_natural_tail_parent_v3",
                "parent_pid":os.getppid(),"deadline_monotonic":10**20,
                "nonce_sha256":"0"*64}))
            (attempt/"pre_host.json").write_text("{}")
            with mock.patch.dict(runner.ATTEMPTS,{"whisper_english":attempt}), \
                    mock.patch.object(runner,"validate_go",side_effect=AssertionError(
                        "admission/model import reached")), \
                    mock.patch.object(runner.old,"_snapshot_load",side_effect=AssertionError(
                        "model load reached")):
                with self.assertRaisesRegex(ValueError,"inherited parent pipe"):
                    runner.run("whisper_english",attempt,Path(temp)/"fake_admission",
                               Path(temp)/"fake_go",-1)

    def test_direct_cli_with_forged_parent_files_rejects_before_inference(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            attempt=Path(temp)/"forged_attempt"
            attempt.mkdir()
            (attempt/"launch_start.json").write_text(json.dumps({
                "schema":"whisper_natural_tail_parent_v3",
                "parent_pid":os.getpid(),"deadline_monotonic":10**20,
                "nonce_sha256":"0"*64}))
            (attempt/"pre_host.json").write_text("{}")
            process=subprocess.run([sys.executable,"-m",
                "models.second_architecture.nonspeech_probe_v1.natural_tail_runner_v3",
                "run","--model","whisper_english","--attempt",str(attempt),
                "--admission",str(Path(temp)/"fake_admission"),
                "--review",str(Path(temp)/"fake_go")],cwd=runner.ROOT,
                text=True,capture_output=True,timeout=20)
            self.assertNotEqual(process.returncode,0)
            self.assertIn("inherited parent pipe required",process.stderr)
            self.assertFalse((attempt/"records.jsonl").exists())

    def test_parent_pipe_rejects_wrong_one_use_nonce(self) -> None:
        read_fd,write_fd=os.pipe()
        try:
            with tempfile.TemporaryDirectory() as temp:
                attempt=Path(temp)
                (attempt/"launch_start.json").write_text(json.dumps({
                    "parent_pid":os.getppid(),
                    "deadline_monotonic":time.monotonic()+10,
                    "nonce_sha256":hashlib.sha256(b"correct"*4).hexdigest()}))
                os.write(write_fd,b"HELLO "+b"0"*32+b"\n")
                with self.assertRaisesRegex(ValueError,"nonce differs"):
                    runner.start_parent_guard(attempt,read_fd)
        finally:
            os.close(read_fd);os.close(write_fd)

    def test_lossless_sidecar_and_mel_regions(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)
            (path/"sidecars").mkdir()
            data=np.arange(12,dtype=np.float32).reshape(3,4)
            ref=runner._npy_once(path/"sidecars"/"a.npy",data)
            self.assertTrue(np.array_equal(audit.load_sidecar(path,ref,(3,4),"float32"),data))
            altered=ref.copy();altered["values_sha256"]="0"*64
            with self.assertRaisesRegex(ValueError,"sidecar"):
                audit.load_sidecar(path,altered,(3,4),"float32")
        a=np.zeros((80,3000),dtype=np.float32)
        b=a.copy();b[:,100]=1
        regions=audit._regions(a,b,200,200,250)
        self.assertEqual(regions["source_interior"]["changed_frames"],1)
        self.assertEqual(regions["nominal_padding_interior"]["changed_frames"],0)

    def test_fake_model_both_calls_save_complete_output(self) -> None:
        class Tokenizer:
            def __init__(self,eos):
                self.eos_token_id=self.pad_token_id=eos
            def decode(self,ids,skip_special_tokens=False):
                return "" if skip_special_tokens else str(ids)

        class Processor:
            def __init__(self,eos):self.tokenizer=Tokenizer(eos)
            def __call__(self,audio,**kwargs):
                self.kwargs=kwargs
                mask=torch.zeros((1,3000),dtype=torch.int64)
                mask[:,:100]=1
                return SimpleNamespace(input_features=torch.zeros((1,80,3000)),
                                       attention_mask=mask)

        class FakeCapture:
            def __init__(self,model,kind):
                self.kind=kind
                self.counts={"generate":{"post_conv_pos":1,"post_block0":1,
                                         "final_encoder":1}}
                self.detection=None
                if kind == "whisper_multilingual":
                    self.counts["detect"]={"post_conv_pos":1,"post_block0":1,
                                           "final_encoder":1}
                    self.detection={"chosen_language_ids":[50259],
                                    "language_logits":{"<|en|>":2.0},
                                    "candidate_count":1}
            def install(self):pass
            def _record(self,name,_tensor):
                self.counts["generate"][name]=1
            def close(self):pass

        class Model:
            def __init__(self,multilingual):
                self.multilingual=multilingual
                self.generation_config=SimpleNamespace(
                    is_multilingual=multilingual,language=None,
                    decoder_start_token_id=50258,task_to_id={"transcribe":50359},
                    no_timestamps_token_id=50363)
            def generate(self,**kwargs):
                self.kwargs=kwargs
                prefix=[50258,50259,50359,50363] if self.multilingual else [50257,50362]
                eos=50257 if self.multilingual else 50256
                raw=torch.zeros((1,51865 if self.multilingual else 51864),dtype=torch.float32)
                raw[0,eos]=10
                return SimpleNamespace(
                    sequences=torch.tensor([prefix+[eos]],dtype=torch.int64),
                    scores=(raw.clone(),),logits=(raw.clone(),),
                    decoder_hidden_states=((torch.zeros((1,len(prefix),384)),),))

        with mock.patch.object(runner.old,"Capture",FakeCapture):
            for model_name in runner.MODELS:
                multilingual=model_name == "whisper_multilingual"
                processor=Processor(50257 if multilingual else 50256)
                model=Model(multilingual)
                with tempfile.TemporaryDirectory() as temp:
                    record=runner.infer_one(model_name,processor,model,
                        np.zeros(16000,dtype=np.int16),Path(temp),"fake")
                    self.assertEqual(record["output"]["termination"],"eos")
                    self.assertEqual(record["output"]["actual_generation_steps"],1)
                    self.assertEqual(record["geometry"]["active_mel_frames"],100)
                    self.assertEqual(set(record["sidecars"]),
                        {"mel","mask","first_raw_logits","first_processed_scores",
                         "first_decoder_state"})
                    self.assertEqual(record["sidecars"]["first_raw_logits"]["shape"],
                                     [51865 if multilingual else 51864])
                    self.assertEqual("task" in model.kwargs,multilingual)
                    self.assertEqual("language" in model.kwargs,multilingual)


if __name__ == "__main__":
    unittest.main()
