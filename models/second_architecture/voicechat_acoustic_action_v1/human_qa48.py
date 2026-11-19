"""Source-only, fail-closed human wording QA for the prospective 48-WAV study.

No human judgment is generated here. The public corpus entry remains HOLD
until an independent variant-byte admission receipt exists. Synthetic fixtures
can exercise the pure builders and auditors without VoiceChat weights.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
SPEC = HERE / "spec.json"
FIRST_TEMPLATE = HERE / "human_qa48_first_template.html"
FIRST_JS = HERE / "human_qa48_first.js"
COMPARE_TEMPLATE = HERE / "human_qa48_compare_template.html"
COMPARE_JS = HERE / "human_qa48_compare.js"
SPEC_SHA256 = "3325f4f693461cf48cf994bfda486e81b2bc6535d0c855a3ac838601644d414d"
PROTOCOL_V1 = HERE.parent / "VOICECHAT_ACOUSTIC_ACTION_PROSPECTIVE_V1.md"
PROTOCOL_V2 = HERE.parent / "VOICECHAT_ACOUSTIC_ACTION_PROSPECTIVE_V2.md"
PROTOCOL_SHAS = (
    "b80d38a10d2bb822c67bbc0cb918dd5e242062d099582428f2dcacd98e53b0a9",
    "ec6aebece37dbf2f8ca744581dc9920bb7069d02ec1ce2fe5f0c2b9ace2db374",
)
ARMS = ("clean", "full_noise", "decision_noise", "displaced_noise")
ID = re.compile(r"[0-9a-f]{24}\Z")
REVIEWER_ID = re.compile(r"[A-Za-z0-9_.-]{3,64}\Z")
HEARD_KEYS = {"date_heard", "event_heard", "authorization_heard"}
FIRST_RECORD_KEYS = {
    "opaque_id", "wav_sha256", "reviewer_id", "submitted_at_utc",
    "verbatim_transcript", "uncertain_spans", "clarity", "confidence", *HEARD_KEYS,
}
COMPARE_KEYS = {
    "opaque_id", "reviewer_id", "compared_at_utc", "date_match",
    "event_match", "authorization_match", "same_critical_meaning_as_clean",
}
CHOICES = {"yes", "no", "unclear"}


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def sha(path: Path) -> str:
    require(path.is_file() and not path.is_symlink(), f"missing or linked file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True).encode("ascii")


def _unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in output, "duplicate JSON key")
        output[key] = value
    return output


def read_json(path: Path) -> Any:
    require(path.is_file() and not path.is_symlink(), f"missing or linked JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique,
                      parse_constant=lambda value: (_ for _ in ()).throw(
                          ValueError(f"nonfinite JSON: {value}")))


def utc(value: Any) -> datetime:
    require(isinstance(value, str) and value.endswith("Z"), "timestamp must be UTC Z")
    try:
        stamp = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise ValueError("invalid timestamp") from error
    require(stamp.utcoffset() is not None and stamp.utcoffset().total_seconds() == 0,
            "timestamp not UTC")
    return stamp


def _plan() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    require(sha(SPEC) == SPEC_SHA256 and sha(PROTOCOL_V1) == PROTOCOL_SHAS[0]
            and sha(PROTOCOL_V2) == PROTOCOL_SHAS[1], "frozen QA specification changed")
    spec = read_json(SPEC)
    source_plan = spec["source_plan"]
    require(len(source_plan) == 12 and len({(r["script_id"], r["voice_id"])
                                          for r in source_plan}) == 12, "12-source plan changed")
    return spec, source_plan


def validate_variants(variant_dir: Path, *, synthetic_test: bool = False) -> dict[str, Any]:
    """Bind all 48 exact WAVs. Real use needs a separate independently pinned receipt."""
    spec, source_plan = _plan()
    manifest_path = variant_dir / "variant_manifest.json"
    manifest = read_json(manifest_path)
    require(manifest.get("schema") == "voicechat_acoustic_action_variants_v1"
            and manifest.get("status") == "complete_48_wavs_no_human_qa_or_model_inference"
            and manifest.get("spec_sha256") == SPEC_SHA256
            and manifest.get("planned_sources") == 12
            and manifest.get("planned_arms") == 48
            and manifest.get("candidate_inference_authorized") is False,
            "variant manifest identity/status differs")
    if synthetic_test:
        require(manifest.get("synthetic_test_only") is True,
                "fake QA must use synthetic variant source")
    else:
        # The future independent variant-byte review/receipt must be bound in a
        # later version. A locally supplied hash cannot authenticate itself.
        raise ValueError("production QA HOLD: independent 48-WAV variant admission absent")
    rows = manifest.get("rows")
    require(isinstance(rows, list) and len(rows) == 48, "48 variant rows required")
    expected = [(r["script_id"], r["voice_id"], arm)
                for r in source_plan for arm in ARMS]
    require([(r.get("script_id"), r.get("voice_id"), r.get("arm"))
             for r in rows] == expected, "variant arm order/population differs")
    names = {r.get("wav_path") for r in rows}
    require(len(names) == 48 and all(isinstance(n, str) and Path(n).name == n
                                    and n.endswith(".wav") for n in names),
            "variant WAV names invalid")
    require({p.name for p in variant_dir.iterdir()} == names | {manifest_path.name},
            "variant directory has missing/extra files")
    for row, (script_id, voice_id, arm) in zip(rows, expected):
        source_id = f"{script_id}__{voice_id}"
        require(row.get("source_id") == source_id
                and row.get("split") == ("development" if script_id.endswith("1") else "locked")
                and row.get("wav_path") == f"{source_id}__{arm}.wav"
                and isinstance(row.get("wav_sha256"), str)
                and re.fullmatch(r"[0-9a-f]{64}", row["wav_sha256"]) is not None
                and sha(variant_dir / row["wav_path"]) == row["wav_sha256"],
                "variant row or WAV hash differs")
    return manifest


def make_private_and_queues(manifest: dict[str, Any], manifest_sha256: str
                            ) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Create no-model fixture mapping; public queues contain no source/arm/reference."""
    spec, _ = _plan()
    require(re.fullmatch(r"[0-9a-f]{64}", manifest_sha256) is not None,
            "variant manifest SHA invalid")
    cases = []
    for row in manifest["rows"]:
        ident = digest(("voicechat-acoustic-qa48-v1|" + manifest_sha256 + "|"
                        + row["source_id"] + "|" + row["arm"] + "|"
                        + row["wav_sha256"]).encode())[:24]
        cases.append({
            "opaque_id": ident, "wav_sha256": row["wav_sha256"],
            "source_id": row["source_id"], "script_id": row["script_id"],
            "voice_id": row["voice_id"], "split": row["split"], "arm": row["arm"],
            "reference_script": spec["scripts"][row["script_id"]],
        })
    require(len({c["opaque_id"] for c in cases}) == 48, "opaque ID collision")
    private = {"schema": "voicechat_acoustic_qa48_private_v1",
               "variant_manifest_sha256": manifest_sha256,
               "spec_sha256": SPEC_SHA256, "planned_sources": 12,
               "planned_arms": 48,
               "synthetic_test_only": manifest.get("synthetic_test_only") is True,
               "cases": cases}
    queues = {}
    for role in ("primary", "secondary"):
        ids = [c["opaque_id"] for c in cases]
        random.Random(int(digest((manifest_sha256 + "|" + role).encode())[:16], 16)).shuffle(ids)
        by_id = {c["opaque_id"]: c for c in cases}
        queues[role] = {
            "schema": "voicechat_acoustic_qa48_blind_queue_v1",
            "variant_manifest_sha256": manifest_sha256,
            "role": role, "planned_clips": 48,
            "clips": [{"opaque_id": ident, "file": ident + ".wav",
                       "wav_sha256": by_id[ident]["wav_sha256"]} for ident in ids],
        }
    return private, queues


def validate_private_queues(private: dict[str, Any],
                            queues: dict[str, dict[str, Any]]) -> None:
    spec, plan = _plan()
    require(isinstance(private, dict)
            and set(private) == {"schema", "variant_manifest_sha256", "spec_sha256",
                                 "planned_sources", "planned_arms",
                                 "synthetic_test_only", "cases"}
            and private["schema"] == "voicechat_acoustic_qa48_private_v1"
            and re.fullmatch(r"[0-9a-f]{64}",
                             private["variant_manifest_sha256"]) is not None
            and private["spec_sha256"] == SPEC_SHA256
            and private["planned_sources"] == 12 and private["planned_arms"] == 48
            and type(private["synthetic_test_only"]) is bool
            and isinstance(private["cases"], list)
            and len(private["cases"]) == 48,
            "private 48-case map invalid")
    expected = [(row["script_id"], row["voice_id"], row["split"], arm)
                for row in plan for arm in ARMS]
    ids = []
    for case, (script, voice, split, arm) in zip(private["cases"], expected):
        require(isinstance(case, dict)
                and set(case) == {"opaque_id", "wav_sha256", "source_id",
                                  "script_id", "voice_id", "split", "arm",
                                  "reference_script"}
                and re.fullmatch(r"[0-9a-f]{64}", case["wav_sha256"]) is not None
                and case["source_id"] == f"{script}__{voice}"
                and (case["script_id"], case["voice_id"], case["split"], case["arm"])
                    == (script, voice, split, arm)
                and case["reference_script"] == spec["scripts"][script],
                "private case or reference differs from frozen source plan")
        derived = digest(("voicechat-acoustic-qa48-v1|"
                          + private["variant_manifest_sha256"] + "|"
                          + case["source_id"] + "|" + arm + "|"
                          + case["wav_sha256"]).encode())[:24]
        require(case["opaque_id"] == derived, "opaque ID derivation differs")
        ids.append(derived)
    require(len(set(ids)) == 48 and set(queues) == {"primary", "secondary"},
            "private/queue population differs")
    by_id = {c["opaque_id"]: c for c in private["cases"]}
    for role in ("primary", "secondary"):
        queue = queues[role]
        require(isinstance(queue, dict)
                and set(queue) == {"schema", "variant_manifest_sha256", "role",
                                   "planned_clips", "clips"}
                and queue["schema"] == "voicechat_acoustic_qa48_blind_queue_v1"
                and queue["variant_manifest_sha256"] ==
                    private["variant_manifest_sha256"]
                and queue["role"] == role and queue["planned_clips"] == 48
                and isinstance(queue["clips"], list) and len(queue["clips"]) == 48,
                "public queue shape/binding differs")
        order = ids.copy()
        seed = int(digest((private["variant_manifest_sha256"] + "|" + role)
                          .encode())[:16], 16)
        random.Random(seed).shuffle(order)
        require(queue["clips"] == [
            {"opaque_id": ident, "file": ident + ".wav",
             "wav_sha256": by_id[ident]["wav_sha256"]} for ident in order],
            "public queue order/WAV map differs")


def validate_first_export(document: Any, queue: dict[str, Any]) -> str:
    require(isinstance(document, dict)
            and set(document) == {"schema", "reviewer_id", "queue_sha256",
                                  "role", "started_at_utc", "exported_at_utc", "records"},
            "first export fields invalid")
    reviewer = document["reviewer_id"]
    require(isinstance(reviewer, str) and REVIEWER_ID.fullmatch(reviewer) is not None,
            "reviewer ID invalid")
    require(document["schema"] == "voicechat_acoustic_qa48_first_v1"
            and document["role"] == queue["role"]
            and document["queue_sha256"] == digest(canonical(queue)),
            "first export queue binding differs")
    clips = queue["clips"]
    records = document["records"]
    require(isinstance(records, list) and len(records) == len(clips),
            "first export incomplete")
    begun, ended = utc(document["started_at_utc"]), utc(document["exported_at_utc"])
    require(begun <= ended, "review export time reversed")
    previous = begun
    for clip, record in zip(clips, records):
        require(isinstance(record, dict) and set(record) == FIRST_RECORD_KEYS,
                "first record fields invalid")
        require(record["opaque_id"] == clip["opaque_id"]
                and record["wav_sha256"] == clip["wav_sha256"]
                and record["reviewer_id"] == reviewer,
                "first record clip/reviewer binding differs")
        moment = utc(record["submitted_at_utc"])
        require(previous <= moment <= ended, "first record time/order differs")
        previous = moment
        require(isinstance(record["verbatim_transcript"], str)
                and bool(record["verbatim_transcript"].strip())
                and isinstance(record["uncertain_spans"], str)
                and bool(record["uncertain_spans"].strip())
                and all(isinstance(record[key], str) and bool(record[key].strip())
                        for key in HEARD_KEYS)
                and record["clarity"] in {"clear", "unclear"}
                and type(record["confidence"]) is int
                and 1 <= record["confidence"] <= 5,
                "first record answer incomplete")
    return reviewer


def comparison_template(private: dict[str, Any], queue: dict[str, Any],
                        first: dict[str, Any]) -> dict[str, Any]:
    """Only call after all first-phase exports for both primary roles are locked."""
    validate_first_export(first, queue)
    cases = {c["opaque_id"]: c for c in private["cases"]}
    clean_by_source = {
        cases[rec["opaque_id"]]["source_id"]: rec["verbatim_transcript"]
        for rec in first["records"] if cases[rec["opaque_id"]]["arm"] == "clean"
    }
    require(queue["role"] == "tertiary" or len(clean_by_source) == 12,
            "first review lacks a clean source recording")
    return {"schema": "voicechat_acoustic_qa48_compare_v1",
            "role": queue["role"], "reviewer_id": first["reviewer_id"],
            "queue_sha256": digest(canonical(queue)),
            "first_export_sha256": digest(canonical(first)),
            "first_exported_at_utc": first["exported_at_utc"],
            "records": [
                {"opaque_id": rec["opaque_id"],
                 "reference_script": cases[rec["opaque_id"]]["reference_script"],
                 "source_id": cases[rec["opaque_id"]]["source_id"],
                 "arm": cases[rec["opaque_id"]]["arm"],
                 "initial_verbatim": rec["verbatim_transcript"],
                 "clean_initial_verbatim":
                     clean_by_source.get(cases[rec["opaque_id"]]["source_id"],
                                         cases[rec["opaque_id"]]["reference_script"]),
                 "clean_comparison_basis":
                     ("first_clean_verbatim"
                      if cases[rec["opaque_id"]]["source_id"] in clean_by_source
                      else "reference_script_for_sparse_third"),
                 "date_match": None, "event_match": None,
                 "authorization_match": None,
                 "same_critical_meaning_as_clean": None,
                 "compared_at_utc": None}
                for rec in first["records"]]}


def validate_comparison(document: Any, template: dict[str, Any]) -> None:
    require(isinstance(document, dict) and set(document) == set(template)
            and all(document[k] == template[k] for k in template if k != "records"),
            "comparison header/first-export binding differs")
    records = document["records"]
    require(isinstance(records, list) and len(records) == len(template["records"]),
            "comparison population incomplete")
    for record, expected in zip(records, template["records"]):
        require(isinstance(record, dict) and set(record) == set(expected)
                and all(record[k] == expected[k] for k in expected
                        if k not in {"date_match", "event_match", "authorization_match",
                                     "same_critical_meaning_as_clean", "compared_at_utc"})
                and all(record[k] in CHOICES for k in
                        ("date_match", "event_match", "authorization_match",
                         "same_critical_meaning_as_clean")),
                "comparison reference or critical marks differ")
        require(utc(record["compared_at_utc"]) >= utc(template["first_exported_at_utc"]),
                "comparison preceded locked first export")


def validate_asr_diagnostics(asr: Any, private: dict[str, Any]) -> str:
    """Require two raw diagnostic streams; their transcripts never score QA."""
    require(isinstance(asr, dict) and set(asr) == {"schema", "records"}
            and asr["schema"] == "voicechat_acoustic_qa48_asr_raw_v1"
            and isinstance(asr["records"], list) and len(asr["records"]) == 96,
            "two raw ASR screens for 48 WAVs required")
    expected = {(case["opaque_id"], case["wav_sha256"], backend)
                for case in private["cases"] for backend in ("whisper", "wav2vec2")}
    seen = set()
    for record in asr["records"]:
        require(isinstance(record, dict)
                and set(record) == {"opaque_id", "wav_sha256", "backend",
                                    "raw_transcript", "status"},
                "ASR diagnostic record fields differ")
        key = (record["opaque_id"], record["wav_sha256"], record["backend"])
        require(key in expected and key not in seen
                and isinstance(record["raw_transcript"], str)
                and record["status"] in {"complete", "failed"},
                "ASR diagnostic identity/result invalid")
        seen.add(key)
    require(seen == expected, "ASR diagnostic population incomplete")
    return digest(canonical(asr))


def evaluate(private: dict[str, Any], queues: dict[str, dict[str, Any]],
             first: dict[str, dict[str, Any]], comparisons: dict[str, dict[str, Any]],
             asr_diagnostics: dict[str, Any],
             *, tertiary_first: dict[str, Any] | None = None,
             tertiary_compare: dict[str, Any] | None = None) -> dict[str, Any]:
    """Pure 48-arm audit; never authorizes candidate model inference."""
    require(set(first) == set(comparisons) == {"primary", "secondary"},
            "both complete initial and post-reference reviews required")
    validate_private_queues(private, queues)
    asr_sha = validate_asr_diagnostics(asr_diagnostics, private)
    reviewers = {}
    for role in ("primary", "secondary"):
        reviewers[role] = validate_first_export(first[role], queues[role])
        validate_comparison(comparisons[role],
                            comparison_template(private, queues[role], first[role]))
    require(reviewers["primary"] != reviewers["secondary"], "reviewers not independent IDs")
    cases = private["cases"]
    by_id = {c["opaque_id"]: c for c in cases}
    role_rows = {}
    for role in ("primary", "secondary"):
        first_map = {r["opaque_id"]: r for r in first[role]["records"]}
        compare_map = {r["opaque_id"]: r for r in comparisons[role]["records"]}
        require(set(first_map) == set(compare_map) == set(by_id),
                "reviewer case population differs")
        role_rows[role] = (first_map, compare_map)
    rows = []
    flags = []
    for case in cases:
        ident = case["opaque_id"]
        a, ac = (role_rows["primary"][0][ident], role_rows["primary"][1][ident])
        b, bc = (role_rows["secondary"][0][ident], role_rows["secondary"][1][ident])
        critical = ("date_match", "event_match", "authorization_match",
                    "same_critical_meaning_as_clean")
        a_ok = a["clarity"] == "clear" and all(ac[k] == "yes" for k in critical)
        b_ok = b["clarity"] == "clear" and all(bc[k] == "yes" for k in critical)
        disagreement = (a["clarity"] != b["clarity"]
                        or any(a[k] != b[k] for k in HEARD_KEYS)
                        or any(ac[k] != bc[k] for k in critical))
        if disagreement:
            flags.append(ident)
        rows.append({"opaque_id": ident, "source_id": case["source_id"],
                     "script_id": case["script_id"], "voice_id": case["voice_id"],
                     "split": case["split"], "arm": case["arm"],
                     "wav_sha256": case["wav_sha256"],
                     "primary_pass": a_ok, "secondary_pass": b_ok,
                     "critical_disagreement": disagreement,
                     "initial_two_listener_pass": a_ok and b_ok})
    require(flags == critical_disagreement_ids(private, queues, first, comparisons),
            "third-review flag reconstruction differs")
    # A third review can explain a disagreement but cannot rescue a clip
    # unless both initial listeners judged its critical meaning unchanged.
    if flags:
        require(tertiary_first is not None and tertiary_compare is not None,
                "third review required for critical disagreement")
        third_queue = third_queue_for_flags(private, flags)
        third_id = validate_first_export(tertiary_first, third_queue)
        require(third_id not in reviewers.values(), "third reviewer ID reused")
        validate_comparison(tertiary_compare,
                            comparison_template(private, third_queue, tertiary_first))
        third_reviewed = True
    else:
        require(tertiary_first is None and tertiary_compare is None,
                "unneeded third review supplied")
        third_reviewed = False
    by_source: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_source.setdefault(row["source_id"], []).append(row)
    require(len(by_source) == 12 and all(len(group) == 4 for group in by_source.values()),
            "source denominator changed")
    sources = [{
        "source_id": source, "split": group[0]["split"],
        "eligible": all(r["initial_two_listener_pass"] for r in group),
        "arms": {r["arm"]: r["initial_two_listener_pass"] for r in group},
    } for source, group in sorted(by_source.items())]
    locked_ab = [s for s in sources if s["split"] == "locked"
                 and s["source_id"].split("__", 1)[0] in {"A2", "B2"}]
    require(len(locked_ab) == 4, "four locked security sources absent")
    return {
        "schema": "voicechat_acoustic_qa48_audit_v1",
        "status": "human_reviews_complete_source_eligibility_recorded",
        "candidate_inference_authorized": False,
        "synthetic_test_only": private["synthetic_test_only"],
        "variant_manifest_sha256": private["variant_manifest_sha256"],
        "private_case_map_sha256": digest(canonical(private)),
        "queue_sha256": {role: digest(canonical(queue))
                          for role, queue in queues.items()},
        "asr_raw_diagnostics_sha256": asr_sha,
        "planned_sources": 12, "planned_arms": 48,
        "reviewed_arms": 48, "eligible_sources": sum(s["eligible"] for s in sources),
        "locked_ab_all_eligible": all(s["eligible"] for s in locked_ab),
        "critical_disagreement_arms": len(flags),
        "third_reviewed": third_reviewed,
        "reviewer_ids": {**reviewers, **({"tertiary": third_id} if flags else {})},
        "case_rows": rows, "sources": sources,
        "first_export_sha256": {k: digest(canonical(v)) for k, v in first.items()},
        "comparison_sha256": {k: digest(canonical(v)) for k, v in comparisons.items()},
        "first_exports": first, "comparisons": comparisons,
        "tertiary_first": tertiary_first, "tertiary_comparison": tertiary_compare,
        "asr_diagnostics": asr_diagnostics,
        "limits": "File checks cannot prove human identity, listening, or blindness.",
    }


def third_queue_for_flags(private: dict[str, Any], flags: list[str]) -> dict[str, Any]:
    require(flags and len(flags) == len(set(flags)), "third review flags invalid")
    by_id = {c["opaque_id"]: c for c in private["cases"]}
    require(set(flags) <= set(by_id), "third review flag not in corpus")
    order = sorted(flags)
    return {
        "schema": "voicechat_acoustic_qa48_blind_queue_v1",
        "variant_manifest_sha256": private["variant_manifest_sha256"],
        "role": "tertiary", "planned_clips": len(order),
        "clips": [{"opaque_id": ident, "file": ident + ".wav",
                   "wav_sha256": by_id[ident]["wav_sha256"]} for ident in order],
    }


def critical_disagreement_ids(private: dict[str, Any],
                              queues: dict[str, dict[str, Any]],
                              first: dict[str, dict[str, Any]],
                              comparisons: dict[str, dict[str, Any]]) -> list[str]:
    """Determine the sparse third-listener population after both comparisons."""
    validate_private_queues(private, queues)
    require(set(first) == set(comparisons) == {"primary", "secondary"},
            "two completed review/comparison pairs required")
    seen = {}
    for role in ("primary", "secondary"):
        validate_first_export(first[role], queues[role])
        validate_comparison(comparisons[role],
                            comparison_template(private, queues[role], first[role]))
        seen[role] = ({r["opaque_id"]: r for r in first[role]["records"]},
                      {r["opaque_id"]: r for r in comparisons[role]["records"]})
    require(first["primary"]["reviewer_id"] != first["secondary"]["reviewer_id"],
            "reviewer IDs reused")
    flags = []
    for case in private["cases"]:
        ident = case["opaque_id"]
        a, ac = seen["primary"][0][ident], seen["primary"][1][ident]
        b, bc = seen["secondary"][0][ident], seen["secondary"][1][ident]
        if (a["clarity"] != b["clarity"]
            or any(a[key] != b[key] for key in HEARD_KEYS)
            or any(ac[key] != bc[key] for key in
                   ("date_match", "event_match", "authorization_match",
                    "same_critical_meaning_as_clean"))):
            flags.append(ident)
    return flags


def build_third_blind_bundle(private: dict[str, Any],
                             queues: dict[str, dict[str, Any]],
                             first: dict[str, dict[str, Any]],
                             comparisons: dict[str, dict[str, Any]],
                             primary_bundle: Path, output_dir: Path) -> dict[str, Any]:
    """Give a third listener only disputed opaque clips and no references."""
    flags = critical_disagreement_ids(private, queues, first, comparisons)
    require(flags and not output_dir.exists(), "no third queue needed or path already used")
    require(primary_bundle.resolve() == (primary_bundle.parent / "primary").resolve(),
            "third source must be verified primary blind bundle")
    verify_blind_bundle(primary_bundle.parent, private, queues)
    queue = third_queue_for_flags(private, flags)
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "QUEUE.json").write_bytes(canonical(queue) + b"\n")
    (output_dir / "FIRST_FORM.html").write_bytes(render_first_form(queue))
    for clip in queue["clips"]:
        shutil.copyfile(primary_bundle / clip["file"], output_dir / clip["file"])
    require({p.name for p in output_dir.iterdir()} ==
            {"QUEUE.json", "FIRST_FORM.html"} |
                {clip["file"] for clip in queue["clips"]},
            "third blind bundle inventory differs")
    require((output_dir / "QUEUE.json").read_bytes() == canonical(queue) + b"\n"
            and (output_dir / "FIRST_FORM.html").read_bytes() ==
                render_first_form(queue)
            and all(sha(output_dir / clip["file"]) == clip["wav_sha256"]
                    for clip in queue["clips"]),
            "third blind bundle bytes differ")
    return queue


def build_fake_blind_bundle(manifest: dict[str, Any], variant_dir: Path,
                            output_dir: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Test-only output. Production builder must first bind an external variant audit."""
    require(validate_variants(variant_dir, synthetic_test=True) == manifest,
            "fake bundle requires verified synthetic input")
    require(not output_dir.exists(), "blind bundle output already used")
    private, queues = make_private_and_queues(manifest, sha(variant_dir / "variant_manifest.json"))
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "PRIVATE_CASES.json").write_bytes(canonical(private) + b"\n")
    for role, queue in queues.items():
        bundle = output_dir / role
        bundle.mkdir()
        (bundle / "QUEUE.json").write_bytes(canonical(queue) + b"\n")
        for case in private["cases"]:
            shutil.copyfile(variant_dir / f'{case["source_id"]}__{case["arm"]}.wav',
                            bundle / f'{case["opaque_id"]}.wav')
        (bundle / "FIRST_FORM.html").write_bytes(render_first_form(queue))
    verify_blind_bundle(output_dir, private, queues)
    return private, queues


def verify_blind_bundle(root: Path, private: dict[str, Any],
                        queues: dict[str, dict[str, Any]]) -> None:
    """Check reference isolation, exact page bytes, queue and all 96 copied WAVs."""
    validate_private_queues(private, queues)
    require(root.is_dir() and not root.is_symlink()
            and {p.name for p in root.iterdir()} ==
                {"PRIVATE_CASES.json", "primary", "secondary"},
            "blind bundle top-level inventory differs")
    require((root / "PRIVATE_CASES.json").read_bytes() == canonical(private) + b"\n",
            "private mapping changed")
    case_by_id = {c["opaque_id"]: c for c in private["cases"]}
    for role in ("primary", "secondary"):
        queue = queues[role]
        bundle = root / role
        require(bundle.is_dir() and not bundle.is_symlink(),
                "blind reviewer directory absent or linked")
        expected = {"QUEUE.json", "FIRST_FORM.html"} | {
            c["opaque_id"] + ".wav" for c in private["cases"]}
        require({p.name for p in bundle.iterdir()} == expected,
                "blind reviewer inventory differs")
        require((bundle / "QUEUE.json").read_bytes() == canonical(queue) + b"\n"
                and (bundle / "FIRST_FORM.html").read_bytes() == render_first_form(queue),
                "blind queue or rendered form bytes differ")
        for clip in queue["clips"]:
            require(ID.fullmatch(clip["opaque_id"]) is not None
                    and clip["wav_sha256"] ==
                        case_by_id[clip["opaque_id"]]["wav_sha256"]
                    and sha(bundle / clip["file"]) == clip["wav_sha256"],
                    "blind WAV hash or opaque mapping differs")


def write_synthetic_receipt(path: Path, report: dict[str, Any]) -> str:
    """Test-only dry run. A real QA GO requires independent file audit."""
    require(report.get("schema") == "voicechat_acoustic_qa48_audit_v1"
            and report.get("synthetic_test_only") is True
            and report.get("candidate_inference_authorized") is False,
            "only synthetic QA report can be written by this source slice")
    require(not path.exists(), "QA receipt path already used")
    path.write_bytes(canonical(report) + b"\n")
    return sha(path)


def _render(template: Path, script: Path, embedded: dict[str, Any]) -> bytes:
    require(template.is_file() and script.is_file()
            and not template.is_symlink() and not script.is_symlink(),
            "QA form source absent or linked")
    shell = template.read_text(encoding="utf-8")
    require(shell.count("__EMBEDDED_DATA__") == 1 and shell.count("__APP_JS__") == 1,
            "QA form slots differ")
    payload = json.dumps(embedded, separators=(",", ":"), ensure_ascii=True)
    payload = payload.replace("<", "\\u003c").replace("&", "\\u0026")
    return shell.replace("__EMBEDDED_DATA__", payload).replace(
        "__APP_JS__", script.read_text(encoding="utf-8")).encode("utf-8")


def render_first_form(queue: dict[str, Any]) -> bytes:
    require(queue.get("schema") == "voicechat_acoustic_qa48_blind_queue_v1"
            and queue.get("role") in {"primary", "secondary", "tertiary"}
            and isinstance(queue.get("clips"), list)
            and len(queue["clips"]) == queue.get("planned_clips"),
            "blind form queue invalid")
    return _render(FIRST_TEMPLATE, FIRST_JS,
                   {"queue": queue, "queue_sha256": digest(canonical(queue))})


def render_compare_form(template: dict[str, Any]) -> bytes:
    count = len(template["records"]) if isinstance(template.get("records"), list) else 0
    require(template.get("schema") == "voicechat_acoustic_qa48_compare_v1"
            and template.get("role") in {"primary", "secondary", "tertiary"}
            and (count == 48 if template.get("role") != "tertiary"
                 else 1 <= count <= 48),
            "comparison form template invalid")
    return _render(COMPARE_TEMPLATE, COMPARE_JS, template)


def build_post_blind_forms(private: dict[str, Any],
                           queues: dict[str, dict[str, Any]],
                           first: dict[str, dict[str, Any]],
                           output_dir: Path) -> None:
    """Reveal references only after both complete independent first exports."""
    require(set(first) == {"primary", "secondary"} and not output_dir.exists(),
            "post-blind forms require two exports and new output directory")
    ids = [validate_first_export(first[role], queues[role])
           for role in ("primary", "secondary")]
    require(ids[0] != ids[1], "first reviewer IDs reused")
    output_dir.mkdir(parents=True, exist_ok=False)
    for role in ("primary", "secondary"):
        template = comparison_template(private, queues[role], first[role])
        (output_dir / f"{role}_COMPARE_FORM.html").write_bytes(
            render_compare_form(template))


def build_third_compare_form(private: dict[str, Any], third_queue: dict[str, Any],
                             third_first: dict[str, Any], output: Path) -> None:
    require(not output.exists(), "third comparison path already used")
    validate_first_export(third_first, third_queue)
    template = comparison_template(private, third_queue, third_first)
    output.write_bytes(render_compare_form(template))
