"""No-weight Git-history faults for V3's original-GO provenance gate."""
from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from . import whisper_tail_causal_worker_v3 as worker


def digest(data:bytes)->str:
    return hashlib.sha256(data).hexdigest()


def git(root:Path,*args:str)->str:
    return subprocess.run(("git",*args),cwd=root,text=True,capture_output=True,
                          check=True).stdout.strip()


def commit(root:Path,message:str)->None:
    git(root,"add",".")
    git(root,"-c","user.name=Codex","-c","user.email=codex@openai.com",
        "commit","-m",message)


def fixture(root:Path,*,review_before_admission:bool=False,
            bound_source:bytes=b"source at GO\n") -> tuple[Path,Path,Path,Path,Path,dict]:
    git(root,"init","-q")
    source=root/"source.txt";source.write_bytes(b"source at GO\n")
    plan=root/"plan.json";plan.write_text("{}\n")
    commit(root,"source")
    review=root/"review.md"
    if review_before_admission:
        review.write_text("independent review\n")
        commit(root,"review too early")
    admission=root/"admission.json"
    expected={"source_identity":{"source.txt":digest(bound_source)}}
    admission.write_text(json.dumps(expected,sort_keys=True)+"\n")
    commit(root,"admission")
    if not review_before_admission:
        review.write_text("independent review\n")
        commit(root,"review")
    go=root/"go.json"
    attempt=root/"unused_attempt"
    go.write_text(json.dumps({"schema":"whisper_tail_causal_go_v3",
        "status":"independent_executable_go","model":"whisper_multilingual",
        "attempt":str(attempt),"admission_sha256":worker.sha(admission),
        "plan_sha256":worker.sha(plan),"source_identity":expected["source_identity"],
        "review_sha256":worker.sha(review)},sort_keys=True)+"\n")
    commit(root,"GO")
    return source,plan,admission,review,go,expected


def validate(root:Path,source:Path,plan:Path,admission:Path,
             review:Path,go:Path,expected:dict)->dict:
    with mock.patch.object(worker,"ROOT",root),\
         mock.patch.object(worker,"HERE",root),\
         mock.patch.object(worker,"SOURCES",("source.txt",)),\
         mock.patch.object(worker,"REVIEW",review),\
         mock.patch.object(worker,"ATTEMPTS",{"whisper_multilingual":root/"unused_attempt"}),\
         mock.patch.object(worker,"ADMISSIONS",{"whisper_multilingual":admission}),\
         mock.patch.object(worker,"GOS",{"whisper_multilingual":go}),\
         mock.patch.object(worker.plan_source,"PLAN_PATH",plan),\
         mock.patch.object(worker,"admission",return_value=expected):
        return worker.validate_go("whisper_multilingual",admission,go)


class OriginalGoHistoryTests(unittest.TestCase):
    def test_byte_identical_post_go_restore_keeps_original_lineage(self)->None:
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            source,plan,admission,review,go,expected=fixture(root)
            self.assertEqual(validate(root,source,plan,admission,review,go,expected),expected)
            source.write_bytes(b"temporary edit\n");commit(root,"later edit")
            source.write_bytes(b"source at GO\n");commit(root,"restore exact bytes")
            self.assertEqual(validate(root,source,plan,admission,review,go,expected),expected)

    def test_changed_current_bytes_reject(self)->None:
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            source,plan,admission,review,go,expected=fixture(root)
            source.write_bytes(b"changed current bytes\n")
            with self.assertRaisesRegex(ValueError,"tracked/clean"):
                validate(root,source,plan,admission,review,go,expected)

    def test_go_time_source_mismatch_reject(self)->None:
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            source,plan,admission,review,go,expected=fixture(
                root,bound_source=b"source after GO\n")
            source.write_bytes(b"source after GO\n");commit(root,"late source mutation")
            with self.assertRaisesRegex(ValueError,"original GO tree"):
                validate(root,source,plan,admission,review,go,expected)

    def test_admission_after_review_reject(self)->None:
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            source,plan,admission,review,go,expected=fixture(
                root,review_before_admission=True)
            with self.assertRaisesRegex(ValueError,"source/admission/review/GO order"):
                validate(root,source,plan,admission,review,go,expected)


if __name__=="__main__":unittest.main()
