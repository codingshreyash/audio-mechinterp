"""Recompute an ASR-only interim report from the two immutable raw runs.

This was written after the raw ASR runs. It imports the previously frozen
auditor for source, corpus, run, normalization, and word-error verification.
It cannot authorize candidate-model inference; human listening is pending.
"""

from __future__ import annotations

import importlib.util
import json
import argparse
from pathlib import Path


HERE = Path(__file__).resolve().parent
AUDITOR = HERE / "audit_asr_listening_v1.py"
RUNS = {
    "whisper": HERE / "asr_runs/whisper_20261118T1130Z",
    "wav2vec2": HERE / "asr_runs/wav2vec2_20261118T1132Z",
}
OUTPUT = HERE / "asr_runs/ASR_ONLY_INTERIM_V1.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("frozen_asr_auditor", AUDITOR)
    assert spec is not None and spec.loader is not None
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)

    cases, order, pins = audit.frozen_cases()
    freeze = audit.verify_freeze()
    rows = {
        key: audit.verify_asr_run(path, key, cases, pins, freeze,
                                  audit.sha(path / "complete.json"))
        for key, path in RUNS.items()
    }
    by_id = {case["opaque_id"]: case for case in cases}
    comparison = {
        key: {ident: audit.compare_asr(by_id[ident], data[ident], key)
              for ident in order}
        for key, data in rows.items()
    }
    summary = {
        key: audit.summarize([data[ident] for ident in order])
        for key, data in comparison.items()
    }
    disagreement = sum(
        comparison["whisper"][ident]["normalized_tokens"]
        != comparison["wav2vec2"][ident]["normalized_tokens"]
        for ident in order
    )
    flags = sum(
        any(not comparison[key][ident]["normalized_script_exact"] for key in rows)
        or comparison["whisper"][ident]["normalized_tokens"]
        != comparison["wav2vec2"][ident]["normalized_tokens"]
        for ident in order
    )
    result = {
        "status": "complete_asr_only_human_qa_pending",
        "scope": "posthoc ASR-only summary from frozen auditor functions; no corpus quality go decision",
        "planned_clips": len(order),
        "auditor_sha256": audit.sha(AUDITOR),
        "freeze_sha256": freeze,
        "runs": {
            key: {"complete_sha256": audit.sha(path / "complete.json"),
                  "raw_sha256": audit.sha(path / "raw.jsonl")}
            for key, path in RUNS.items()
        },
        "aggregate": summary,
        "asr_disagreement_count": disagreement,
        "requires_secondary_from_asr_alone": flags,
        "human_primary_responses": 0,
        "candidate_model_inference_authorized": False,
    }
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "output_sha256": audit.sha(args.output)},
                     indent=2))


if __name__ == "__main__":
    main()
