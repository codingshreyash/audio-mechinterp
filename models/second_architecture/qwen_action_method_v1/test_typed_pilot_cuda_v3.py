"""No-weight V3 processor-prerequisite, candidate, and GO fault tests."""
from __future__ import annotations

import base64
import hashlib
import io
import json
import sys
import tempfile
import types
import unittest
import zipfile
from pathlib import Path
from unittest import mock

from . import typed_pilot_cuda_admission_v3 as admission
from . import typed_pilot_cuda_backend_v3 as backend
from . import typed_pilot_cuda_parent_v3 as parent


class CudaV3NoWeightTests(unittest.TestCase):
    def test_fixed_fresh_paths_and_absent_go(self) -> None:
        self.assertTrue(str(backend.LOAD_ATTEMPT).endswith("typed_pilot_cuda_v3/load_002"))
        self.assertTrue(str(backend.PILOT_ATTEMPT).endswith("typed_pilot_cuda_v3/pilot_002"))
        self.assertNotEqual(backend.LOAD_ATTEMPT,backend.PILOT_ATTEMPT)
        with tempfile.TemporaryDirectory() as temp:
            attempt=Path(temp)/"load_002"
            with mock.patch.object(backend,"stage_paths",return_value=(attempt,Path(temp)/"missing_go")), \
                 mock.patch.object(backend,"source_identity",return_value={}), \
                 mock.patch.object(backend,"ADMISSION_CANDIDATE",Path(temp)/"missing_admission"):
                with self.assertRaisesRegex(ValueError,"processor admission candidate absent"):
                    parent.launch("load")
            self.assertFalse(attempt.exists())

    def test_fake_processor_preflight_requires_all_four_classes(self) -> None:
        revision="f75b40e3da2003cdd6e1829b1f420ca70797c34e"
        with tempfile.TemporaryDirectory() as temp:
            snapshot=Path(temp)/"snapshots"/revision
            snapshot.mkdir(parents=True)
            class Qwen2_5OmniProcessor:
                @classmethod
                def from_pretrained(cls,path,local_files_only):
                    self.assertEqual(path,str(snapshot))
                    self.assertTrue(local_files_only)
                    obj=cls()
                    obj.image_processor=type("Qwen2VLImageProcessor",(),{})()
                    obj.video_processor=type("Qwen2VLVideoProcessor",(),{})()
                    obj.feature_extractor=type("WhisperFeatureExtractor",(),{})()
                    obj.tokenizer=types.SimpleNamespace(eos_token_id=151645)
                    return obj
            fake=types.ModuleType("transformers")
            fake.Qwen2_5OmniProcessor=Qwen2_5OmniProcessor
            with mock.patch.dict(sys.modules,{"transformers":fake}):
                result=admission.verify_processor(snapshot)
                self.assertEqual(result["classes"],admission.PROCESSOR_CLASSES)
                bad=dict(admission.PROCESSOR_CLASSES,video_processor="WrongVideoProcessor")
                with mock.patch.object(admission,"PROCESSOR_CLASSES",bad):
                    with self.assertRaisesRegex(ValueError,"processor classes changed"):
                        admission.verify_processor(snapshot)

    def test_fake_wheel_and_installed_file_pin_detects_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            pins={}
            wheels={}
            for name,prefix in (("Pillow","PIL/"),("torchvision","torchvision/")):
                payload=(name+"-binary").encode()
                relative=prefix+"_extension.so"
                installed=root/relative
                installed.parent.mkdir(parents=True)
                installed.write_bytes(payload)
                wheel=root/(name.lower()+"-fake.whl")
                encoded=base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).decode().rstrip("=")
                with zipfile.ZipFile(wheel,"w") as archive:
                    archive.writestr(relative,payload)
                    archive.writestr(name+"-1.dist-info/RECORD",
                        f"{relative},sha256={encoded},{len(payload)}\n")
                pins[name]={"version":"1.0", "env":"unused", "wheel":wheel.name,
                            "sha256":admission.base.sha(wheel),"prefixes":(prefix,)}
                wheels[name]=wheel
            def distribution(_name):
                return types.SimpleNamespace(locate_file=lambda relative:root/relative)
            with mock.patch.object(admission,"PREREQUISITES",pins), \
                 mock.patch.object(admission.importlib.metadata,"version",return_value="1.0"), \
                 mock.patch.object(admission.importlib.metadata,"distribution",
                                   side_effect=distribution):
                verified=admission.verify_prerequisites(wheel_paths=wheels)
                self.assertEqual(set(verified),{"Pillow","torchvision"})
                (root/"torchvision/_extension.so").write_text("changed")
                with self.assertRaisesRegex(ValueError,"installed torchvision file differs"):
                    admission.verify_prerequisites(wheel_paths=wheels)

    def test_capture_admission_is_fixed_and_no_weight(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp)
            candidate=base/"admission_candidate_v3.json"
            identity={"processor_preflight":{"classes":admission.PROCESSOR_CLASSES}}
            with mock.patch.object(backend,"ADMISSION_CANDIDATE",candidate), \
                 mock.patch.object(backend,"LOAD_ATTEMPT",base/"load_002"), \
                 mock.patch.object(backend,"PILOT_ATTEMPT",base/"pilot_002"), \
                 mock.patch.object(backend,"source_identity",return_value=identity), \
                 mock.patch.object(backend,"runtime_identity",return_value={"gpu_uuid":"GPU-A"}), \
                 mock.patch.object(admission,"gpu_inventory",return_value={"gpu_used_bytes":0}):
                result=parent.capture_admission()
                self.assertEqual(result["status"],"source_only_no_weights")
                self.assertEqual(json.loads(candidate.read_text())["processor_preflight"],
                                 identity["processor_preflight"])
                with self.assertRaisesRegex(ValueError,"already exists"):
                    parent.capture_admission()

    def test_go_candidate_runtime_mismatch_rejects_before_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp)
            attempt,go,review,candidate=(base/name for name in
                ("load_002","go.json","review.md","candidate.json"))
            review.write_text("fake independent review")
            identity={"processor_preflight":{"classes":admission.PROCESSOR_CLASSES}}
            runtime={"gpu_uuid":"GPU-A"}
            baseline={"gpu_uuid":"GPU-A","gpu_name":"NVIDIA A100-SXM4-80GB",
                "gpu_memory_mib":81920,"mig_mode":"Disabled",
                "compute_processes":[],"gpu_used_bytes":0,
                "idle_gpu_limit_bytes":admission.IDLE_GPU_LIMIT_BYTES}
            candidate.write_text(json.dumps({"schema":"qwen_action_cuda_admission_candidate_v3",
                "status":"source_only_no_weights","source_identity":identity,
                "processor_preflight":identity["processor_preflight"],
                "approved_load_attempt":str(attempt.resolve()),
                "approved_pilot_attempt":str((base/"pilot_002").resolve()),
                "runtime_identity":runtime,"gpu_baseline":baseline}))
            go.write_text(json.dumps({"schema":"qwen_action_typed_pilot_cuda_go_v3",
                "status":"independent_executable_go","stage":"load",
                "approved_attempt":str(attempt.resolve()),
                "scope":"one typed F_L1 pilot only",
                "qa_policy":"typed_F_L1_only_no_audio_QA_exception",
                "qa_amendment_sha256":backend.sha(backend.QA_AMENDMENT),
                "parent_amendment_sha256":backend.sha(backend.PARENT_AMENDMENT),
                "admission_candidate_sha256":backend.sha(candidate),
                "source_identity":identity,"runtime_identity":{"gpu_uuid":"GPU-B"},
                "gpu_baseline":baseline,"review_sha256":backend.sha(review)}))
            with mock.patch.object(backend,"source_identity",return_value=identity), \
                 mock.patch.object(backend,"stage_paths",return_value=(attempt,go)), \
                 mock.patch.object(backend,"ADMISSION_CANDIDATE",candidate), \
                 mock.patch.object(backend,"LOAD_ATTEMPT",attempt), \
                 mock.patch.object(backend,"PILOT_ATTEMPT",base/"pilot_002"), \
                 mock.patch.object(backend,"REVIEW",review), \
                 mock.patch.object(backend,"require_independent_source_review"):
                with self.assertRaisesRegex(ValueError,"GO binding invalid"):
                    backend.validate_go("load",verify_runtime=False)
            self.assertFalse(attempt.exists())

    def test_source_review_must_follow_v3_source_and_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            review=Path(temp)/"review.md"
            candidate=Path(temp)/"candidate.json"
            go=Path(temp)/"go.json"
            def commit(path):
                if path == review:return "review"
                if path == candidate:return "candidate"
                if path == go:return "go"
                return "source"
            with mock.patch.object(backend,"REVIEW",review), \
                 mock.patch.object(backend,"ADMISSION_CANDIDATE",candidate), \
                 mock.patch.object(backend,"_commit_of",side_effect=commit), \
                 mock.patch.object(backend,"_ancestor",return_value=True):
                backend.require_independent_source_review(go)
            def same_commit(path):
                return "source" if path == review else commit(path)
            with mock.patch.object(backend,"REVIEW",review), \
                 mock.patch.object(backend,"ADMISSION_CANDIDATE",candidate), \
                 mock.patch.object(backend,"_commit_of",side_effect=same_commit), \
                 mock.patch.object(backend,"_ancestor",return_value=True):
                with self.assertRaisesRegex(ValueError,"independent source review/GO"):
                    backend.require_independent_source_review(go)


if __name__ == "__main__":
    unittest.main()
