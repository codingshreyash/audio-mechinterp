"""Frozen Track B fixture, transcript and human-QA admission rules.

No candidate weights are loaded here. The 32 synthetic WAVs are known exactly,
but model inference remains closed until two independent human transcripts per
clip and an independently reviewed QA receipt exist.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import unicodedata
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from . import audit as fixture_audit
from . import render

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RENDERED = HERE / "rendered_v2"
MANIFEST = RENDERED / "manifest.json"
TECHNICAL = RENDERED / "TECHNICAL_BINDING.json"
QUEUE = RENDERED / "listening/QUEUE.json"
PLAN = HERE.parent / "WORD_DISJOINT_TIME_STATE_VALIDATION_PLAN_V2.md"
PLAN_REVIEW = HERE.parent / "WORD_DISJOINT_TIME_STATE_VALIDATION_PLAN_V2_INDEPENDENT_REVIEW.md"
QA_RECEIPT = HERE / "QA_GO.json"
QA_REVIEW = HERE / "QA_GO_INDEPENDENT_REVIEW.md"
HUMAN_QA_AUDIT = RENDERED / "QA_HUMAN_AUDIT_V1.json"
PRIVATE_CASE_MAP = RENDERED / "QA_PRIVATE_CASES.json"
MANIFEST_SHA = "f8fd1f43f38877d6d08c47597c8db9f494e1377a69f0a2451c2f99fbcda71ea2"
TECHNICAL_SHA = "027763d4fa758dac95c552c2f2aef60683f1c780b0c034d82327be8bd59e93b9"
QUEUE_SHA = "9a4113618f23b25d9aaca6846f57f014fbe0bc7d074d0c8f807ff01b32e4dc2c"
ARMS = ("self", "W", "O", "full", "random_W", "random_O")
SITES_W2V = ("feature_projection", "post_block11")
SITES_WHISPER = ("post_block0", "post_norm")


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def committed_clean(path: Path) -> str:
    """Require a tracked, byte-clean artifact and return its source commit."""
    resolved = path.resolve()
    require(resolved.is_relative_to(ROOT) and path.is_file()
            and not path.is_symlink(), "review/freeze path invalid")
    rel = str(resolved.relative_to(ROOT))
    tracked = subprocess.run(["git", "-C", str(ROOT), "ls-files", "--error-unmatch",
                              "--", rel], capture_output=True)
    dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain",
                            "--", rel], capture_output=True, text=True, check=True)
    require(tracked.returncode == 0 and not dirty.stdout.strip(),
            f"review/freeze not committed clean: {rel}")
    commit = subprocess.check_output(["git", "-C", str(ROOT), "log", "-1",
                                      "--format=%H", "--", rel], text=True).strip()
    require(re.fullmatch(r"[0-9a-f]{40}", commit) is not None,
            "review/freeze source commit missing")
    return commit


def source_bindings(paths: dict[str, Path]) -> dict[str, dict[str, str]]:
    """Bind every executable or review-test source to its committed bytes."""
    result = {}
    for role, path in paths.items():
        result[role] = {"path": str(path.resolve().relative_to(ROOT)),
                        "sha256": sha(path), "commit": committed_clean(path)}
    return result


def require_source_go_text(text: str, bindings: dict[str, dict[str, str]]) -> None:
    """Reject an old GO when any reviewed source binding changes."""
    blocks = re.findall(r"```json\s*\n(.*?)\n```", text, flags=re.S)
    bound = []
    for block in blocks:
        try:
            value = json.loads(block)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and value.get("schema") == "word_disjoint_source_go_v1":
            bound.append(value)
    require(len(bound) == 1 and bound[0].get("decision") == "GO"
            and "Decision: GO" in text
            and bound[0].get("source_bindings") == bindings,
            "source GO does not bind exact reviewed sources")


def reviewed_freeze_paths(preflight: Path, freeze: Path, review: Path,
                          bindings: dict[str, dict[str, str]]) -> None:
    """Require exact source blobs to predate a separate GO and frozen receipt."""
    preflight_commit = committed_clean(preflight)
    review_commit = committed_clean(review)
    freeze_commit = committed_clean(freeze)
    require_source_go_text(review.read_text(), bindings)
    for role, item in bindings.items():
        path = ROOT / item["path"]
        require(path.resolve().is_relative_to(ROOT) and path.is_file()
                and not path.is_symlink() and sha(path) == item["sha256"]
                and committed_clean(path) == item["commit"],
                f"{role} reviewed source changed")
        snapshot = subprocess.run(["git", "-C", str(ROOT), "show",
                                   f"{review_commit}:{item['path']}"],
                                  capture_output=True)
        require(snapshot.returncode == 0
                and hashlib.sha256(snapshot.stdout).hexdigest() == item["sha256"],
                f"{role} source blob absent from independent GO commit")
    for earlier, later in (((item["commit"], review_commit)
                            for item in bindings.values())):
        require(earlier != later and subprocess.run(
            ["git", "-C", str(ROOT), "merge-base", "--is-ancestor", earlier, later],
            capture_output=True).returncode == 0,
            "reviewed source commit does not predate independent GO")
    for earlier, later in ((preflight_commit, freeze_commit),
                           (review_commit, freeze_commit)):
        require(earlier != later and subprocess.run(
            ["git", "-C", str(ROOT), "merge-base", "--is-ancestor", earlier, later],
            capture_output=True).returncode == 0,
            "source review/preflight/freeze commit ordering invalid")


def tokens(value: str) -> list[str]:
    return re.sub(r"[\W_]+", " ", unicodedata.normalize("NFKC", value).casefold()).split()


def reference_tokens(word: str) -> list[str]:
    return ["i", "said", word, "and", "then", "stopped"]


@dataclass(frozen=True)
class Case:
    pair_id: str
    engine: str
    word: str
    path: Path
    wav_sha256: str
    pcm_sha256: str
    frames: int
    slot: tuple[int, int]
    opaque_id: str

    @property
    def key(self) -> str:
        return f"{self.pair_id}_{self.engine}_{self.word}"


def load_pcm(case: Case) -> np.ndarray:
    require(sha(case.path) == case.wav_sha256, "candidate input WAV changed")
    with wave.open(str(case.path), "rb") as reader:
        require((reader.getnchannels(), reader.getframerate(), reader.getsampwidth(),
                 reader.getcomptype()) == (1, 16000, 2, "NONE"),
                "candidate input is not mono 16-kHz PCM16")
        raw = reader.readframes(reader.getnframes())
    require(len(raw) == 2 * case.frames
            and hashlib.sha256(raw).hexdigest() == case.pcm_sha256,
            "candidate input PCM hash/frame count differs")
    return np.frombuffer(raw, dtype="<i2").copy()


def fixture_cases() -> tuple[list[Case], dict[str, Any]]:
    """Re-audit every rendered byte; never read candidate outputs or locked WAVs."""
    require(all(p.is_file() for p in (MANIFEST, TECHNICAL, QUEUE, PLAN, PLAN_REVIEW)),
            "Track B fixture, plan, or review is absent")
    require(sha(MANIFEST) == MANIFEST_SHA and sha(TECHNICAL) == TECHNICAL_SHA
            and sha(QUEUE) == QUEUE_SHA, "frozen Track B fixture bytes changed")
    require("Decision: GO" in PLAN_REVIEW.read_text(), "prospective plan GO absent")
    manifest = json.loads(MANIFEST.read_text())
    binding = json.loads(TECHNICAL.read_text())
    fixture_audit.validate_provenance(manifest)
    fixture_audit.verify_saved_report(manifest, binding, RENDERED, MANIFEST_SHA)
    queue = json.loads(QUEUE.read_text())
    require(queue.get("status") == "blinded_listening_queue_no_results"
            and queue.get("planned_clips") == 32
            and queue.get("provenance", {}).get("manifest_sha256") == MANIFEST_SHA
            and queue["provenance"].get("technical_binding_sha256") == TECHNICAL_SHA,
            "blinded queue does not bind technical fixture")
    from .queue import build
    private_expected, public_expected = build(manifest, binding, MANIFEST_SHA, TECHNICAL_SHA)
    require(queue == public_expected, "blinded queue contents differ from frozen fixture")
    lookup = {x["source_path"]: x["opaque_id"] for x in private_expected["cases"]}
    cases: list[Case] = []
    for row in manifest["pairs"]:
        pair_id, engine = row["pair_id"], row["engine"]
        slot = (row["word_slot_start_sample"], row["word_slot_end_sample_exclusive"])
        for word in row["words"]:
            output = row["outputs"][word]
            path = ROOT / output["path"]
            require(path.resolve().is_relative_to(RENDERED.resolve())
                    and path.is_file() and not path.is_symlink()
                    and sha(path) == output["file_sha256"], "clip path/hash invalid")
            cases.append(Case(pair_id, engine, word, path, output["file_sha256"],
                              output["pcm_int16_sha256"], output["frames"], slot,
                              lookup[output["path"]]))
    require(len(cases) == 32 and len({c.key for c in cases}) == 32
            and len({c.opaque_id for c in cases}) == 32, "Track B case count differs")
    return cases, {"manifest_sha256": MANIFEST_SHA,
                   "technical_binding_sha256": TECHNICAL_SHA,
                   "queue_sha256": QUEUE_SHA,
                   "plan_sha256": sha(PLAN),
                   "plan_review_sha256": sha(PLAN_REVIEW)}


def directions(cases: list[Case], modes: tuple[str, ...]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Case]] = {}
    for case in cases:
        grouped.setdefault((case.pair_id, case.engine), []).append(case)
    require(len(grouped) == 16, "paired case population differs")
    out: list[dict[str, Any]] = []
    for (pair_id, engine), pair in sorted(grouped.items()):
        require(len(pair) == 2 and pair[0].slot == pair[1].slot
                and pair[0].frames == pair[1].frames, "paired lengths/slots differ")
        for recipient, donor in ((pair[0], pair[1]), (pair[1], pair[0])):
            for mode in modes:
                out.append({"index": len(out), "pair_id": pair_id, "engine": engine,
                            "mode": mode, "recipient_key": recipient.key,
                            "donor_key": donor.key, "recipient_word": recipient.word,
                            "donor_word": donor.word, "slot": list(recipient.slot),
                            "frames": recipient.frames})
    require(len(out) == 32 * len(modes), "directed case population differs")
    return out


def evaluate_human_reviews(exports: list[dict], cases: list[Case], queue: dict) -> dict:
    """Validate the blinded initial records; never infer what a listener heard."""
    by_id = {c.opaque_id: c for c in cases}
    order = queue["opaque_ids_in_order"]
    require(len(exports) in (2, 3), "need two primary and at most one adjudicator")
    reviewer_ids = [e.get("reviewer_id") for e in exports]
    require(all(isinstance(x, str) and x.strip() for x in reviewer_ids)
            and len(set(reviewer_ids)) == len(reviewer_ids), "reviewers not independent IDs")
    answer_sets = []
    for n, export in enumerate(exports):
        require(export.get("schema") == "word_disjoint_blind_review_v1"
                and export.get("queue_sha256") == QUEUE_SHA
                and export.get("planned_clips") == 32
                and export.get("opaque_ids_in_order") == order,
                "review export provenance/order differs")
        rows = export.get("records")
        require(isinstance(rows, list) and len(rows) == 32
                and [r.get("opaque_id") for r in rows] == order,
                "reviewer did not complete exact 32-clip queue")
        answers = {}
        for row in rows:
            ident = row["opaque_id"]
            case = by_id[ident]
            require(row.get("wav_sha256") == case.wav_sha256
                    and row.get("reviewer_id") == reviewer_ids[n]
                    and isinstance(row.get("submitted_at_utc"), str)
                    and row["submitted_at_utc"].strip()
                    and isinstance(row.get("verbatim_transcript"), str)
                    and isinstance(row.get("middle_word"), str)
                    and row.get("clear_or_unclear") in ("clear", "unclear")
                    and type(row.get("confidence")) is int
                    and 1 <= row["confidence"] <= 5,
                    "review record fields/hash invalid")
            answers[ident] = {
                "tokens": tokens(row["verbatim_transcript"]),
                "middle": tokens(row["middle_word"]),
                "clear": row["clear_or_unclear"] == "clear",
                "raw_transcript": row["verbatim_transcript"],
                "raw_middle": row["middle_word"],
            }
        answer_sets.append(answers)
    # The prospective QA auditor escalates raw answer/clarity disagreements,
    # even if punctuation normalization later yields the same score.
    disagreements = {ident for ident in order
                     if any(answer_sets[0][ident][k] != answer_sets[1][ident][k]
                            for k in ("raw_transcript", "raw_middle", "clear"))}
    require(not disagreements or len(exports) == 3,
            "third review coverage differs from disagreements")
    failures = []
    for ident in order:
        case = by_id[ident]
        accepted = sum(answer[ident]["tokens"] == reference_tokens(case.word)
                       and answer[ident]["middle"] == [case.word]
                       and answer[ident]["clear"]
                       for answer in answer_sets)
        if accepted < 2:
            failures.append(ident)
    return {"quality_confirmed": 32 - len(failures),
            "failed_opaque_ids": failures,
            "reviewer_ids": reviewer_ids,
            "disagreement_count": len(disagreements)}


def qa_gate(cases: list[Case]) -> dict[str, Any]:
    """Require exact human outputs and an independent, hash-bound QA GO."""
    require(QA_RECEIPT.is_file() and QA_REVIEW.is_file()
            and HUMAN_QA_AUDIT.is_file(),
            "32/32 human listening QA receipt and independent review absent")
    receipt = json.loads(QA_RECEIPT.read_text())
    human_audit = json.loads(HUMAN_QA_AUDIT.read_text())
    require(receipt.get("schema") == "word_disjoint_audio_qa_go_v1"
            and receipt.get("status") == "independent_human_qa_32_of_32_pass"
            and receipt.get("manifest_sha256") == MANIFEST_SHA
            and receipt.get("technical_binding_sha256") == TECHNICAL_SHA
            and receipt.get("queue_sha256") == QUEUE_SHA
            and receipt.get("confirmed_clips") == 32
            and receipt.get("human_qa_audit_report_sha256") == sha(HUMAN_QA_AUDIT)
            and human_audit.get("schema") == "word_disjoint_human_qa_audit_v1"
            and human_audit.get("status") == "human_qa_32_of_32_pass"
            and human_audit.get("human_quality_gate_passed") is True
            and human_audit.get("passed_clips") == 32
            and human_audit.get("manifest_sha256") == MANIFEST_SHA
            and human_audit.get("technical_binding_sha256") == TECHNICAL_SHA
            and human_audit.get("public_queue_sha256") == QUEUE_SHA,
            "human listening QA receipt not bound or incomplete")
    queue = json.loads(QUEUE.read_text())
    exports = []
    listed = receipt.get("review_exports")
    require(isinstance(listed, list) and len(listed) in (2, 3),
            "human review export inventory incomplete")
    stages = ("primary", "secondary", "tertiary")[:len(listed)]
    require([item.get("stage") for item in listed] == list(stages),
            "human review export stages/order differ")
    for item in listed:
        relative = item.get("path")
        require(isinstance(relative, str), "human review export path absent")
        raw_path = ROOT / relative
        path = raw_path.resolve()
        require(path.is_relative_to(ROOT) and path.is_file()
                and not raw_path.is_symlink() and sha(path) == item.get("sha256"),
                "human review export bytes differ")
        exports.append(json.loads(path.read_text()))
    result = evaluate_human_reviews(exports, cases, queue)
    require(result["quality_confirmed"] == 32
            and receipt.get("reviewer_ids") == result["reviewer_ids"]
            and human_audit.get("reviewer_ids") ==
            dict(zip(stages, result["reviewer_ids"]))
            and human_audit.get("raw_exports") == dict(zip(stages, exports))
            and receipt.get("clip_sha256_by_id") ==
            {c.opaque_id: c.wav_sha256 for c in cases},
            "32/32 blinded human judgments not established")
    audit_item = receipt.get("qa_auditor_source")
    require(isinstance(audit_item, dict) and isinstance(audit_item.get("path"), str),
            "independent QA auditor source binding absent")
    raw_auditor = ROOT / audit_item["path"]
    auditor_path = raw_auditor.resolve()
    require(auditor_path.is_relative_to(ROOT) and auditor_path.is_file()
            and not raw_auditor.is_symlink()
            and sha(auditor_path) == audit_item.get("sha256")
            and human_audit.get("auditor_source_sha256") == audit_item.get("sha256")
            and PRIVATE_CASE_MAP.is_file()
            and human_audit.get("private_case_map_sha256") == sha(PRIVATE_CASE_MAP)
            and human_audit.get("review_export_sha256") ==
            dict(zip(stages, (item["sha256"] for item in listed))),
            "QA auditor source changed")
    review = QA_REVIEW.read_text()
    receipt_commit = committed_clean(QA_RECEIPT)
    review_commit = committed_clean(QA_REVIEW)
    ancestor = subprocess.run(["git", "-C", str(ROOT), "merge-base",
                               "--is-ancestor", receipt_commit, review_commit],
                              capture_output=True)
    require("Decision: GO" in review and sha(QA_RECEIPT) in review
            and sha(HUMAN_QA_AUDIT) in review
            and MANIFEST_SHA in review and TECHNICAL_SHA in review
            and receipt_commit != review_commit and ancestor.returncode == 0,
            "independent QA review does not bind exact receipt/fixture")
    return {"qa_receipt_sha256": sha(QA_RECEIPT),
            "qa_review_sha256": sha(QA_REVIEW),
            "review_export_sha256": [x["sha256"] for x in listed],
            "qa_result": result}

