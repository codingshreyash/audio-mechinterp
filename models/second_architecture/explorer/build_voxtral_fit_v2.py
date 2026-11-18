"""Build a read-only, self-contained explorer from the audited Voxtral FIT run.

This is a presentation layer, not an evaluator. It refuses unbound or incomplete
data and never modifies the experiment's source artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE.parent
ROOT = MODEL_DIR.parents[1]
RUN = MODEL_DIR / "runs/voxtral_fit_v2_causal_full/20261118T0928Z"
MASKS = MODEL_DIR / "VOXTRAL_FIT_V2_CAUSAL_MASKS.json"
RUNNER = MODEL_DIR / "voxtral_fit_v2_causal_full.py"
TEMPLATE = HERE / "template.html"
APP_JS = HERE / "app.js"
OUTPUT = HERE / "voxtral_fit_v2.html"
EXPECTED_SHA = {
    "result": "e29a686e30505a69a3e4b26bddde606a02db753a3a5410c38432cdd14ac3baaf",
    "audit": "bdedcc93bdb2a50bdcf832d5877e56c80e725ad1f8e3bc1c60461a8547b75182",
    "meta": "90a46945c2036efd286f9c842b5900bd5fdf5a484c55253da0ff4f9cb18e1d21",
    "runner": "4e0a17f80d70ca1498f2487d8cca81b20e36381bfeb35945807b2587b89373cd",
    "masks": "a634fc0ab8454b7a82802c56a5365dc90e79f446e4d280cee6321adfead4108e",
}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def load_verified() -> dict:
    result_path = RUN / "result.json"
    audit_path = RUN / "audit.json"
    meta_path = RUN / "meta.json"
    require(all(path.is_file() for path in (result_path, audit_path, meta_path, MASKS, RUNNER)),
            "Missing audited run/source artifact")
    for name, path in (("result", result_path), ("audit", audit_path), ("meta", meta_path),
                       ("runner", RUNNER), ("masks", MASKS)):
        require(sha(path) == EXPECTED_SHA[name], f"Pinned {name} SHA mismatch")
    result = json.loads(result_path.read_text())
    audit = json.loads(audit_path.read_text())
    meta = json.loads(meta_path.read_text())
    masks = json.loads(MASKS.read_text())
    require(audit.get("status") == "passed_read_only_full_fit_audit", "Independent audit did not pass")
    require(result.get("status") == "eight_record_fit_complete", "Result is incomplete")
    require(audit.get("result_sha256") == sha(result_path), "Audited result SHA mismatch")
    require(audit.get("meta_sha256") == sha(meta_path) == result.get("meta_sha256"),
            "Metadata SHA mismatch")
    require(meta.get("script_sha256") == sha(RUNNER), "Runner SHA mismatch")
    require(meta.get("masks_sha256") == sha(MASKS), "Mask SHA mismatch")
    require(result.get("summary") == audit.get("summary"), "Audited gate summary mismatch")
    require(result.get("directions") == audit.get("directions") == 16, "Wrong directional population")
    require(result.get("progress_rows") == 864 and result.get("captures") == 64 and
            result.get("generations") == 800 and result.get("invalid_conditions") == 0,
            "Run workload differs from frozen plan")
    require(len(masks.get("records", [])) == 8, "Wrong frozen mask population")

    rows = []
    for ordinal in range(1, 9):
        checkpoint_path = RUN / f"record_{ordinal:02d}.checkpoint.json"
        require(result["checkpoint_sha256"].get(str(ordinal)) == sha(checkpoint_path),
                f"Checkpoint {ordinal} SHA mismatch")
        checkpoint = json.loads(checkpoint_path.read_text())
        attempt_path = RUN / checkpoint["attempt_file"]
        require(checkpoint.get("attempt_sha256") == sha(attempt_path),
                f"Attempt {ordinal} SHA mismatch")
        attempt = [json.loads(line) for line in attempt_path.read_text().splitlines() if line.strip()]
        require(len(attempt) == 108, f"Attempt {ordinal} has wrong slot count")
        require(sum(row["type"] == "capture" for row in attempt) == 8,
                f"Attempt {ordinal} capture count changed")
        require(sum(row["type"] == "generation" for row in attempt) == 100,
                f"Attempt {ordinal} generation count changed")
        rows.extend(attempt)
    require(len(rows) == 864, "Incomplete raw rows")
    generated = [row for row in rows if row["type"] == "generation"]
    case_ids = {f"{r['pair_id']}|{r['engine']}|fit_chosen|{r['recipient']}" for r in generated}
    require(len(case_ids) == 16 and case_ids == set(meta["case_ids"]),
            "Generated rows differ from frozen cases")

    compact_rows = []
    for row in generated:
        edit = row.get("edit") or {}
        compact_rows.append({
            "case_id": f"{row['pair_id']}|{row['engine']}|fit_chosen|{row['recipient']}",
            "pair": row["pair_id"], "voice": row["engine"], "recipient": row["recipient"],
            "opposite": row["opposite"], "site": row["site"], "kind": row["kind"],
            "output": row["raw_decoded"], "classification": row["classification"],
            "token_ids": row["generated_token_ids"], "terminal_eos": row["terminal_eos"],
            "contrast": row["recipient_minus_opposite"],
            "scores": row["canonical_scores"],
            "hook_calls": row["hook_calls_including_two_scores"],
            "edit": {key: edit.get(key) for key in (
                "mode", "rows", "alpha", "unscaled_edit_norm", "target_bf16_norm",
                "effective_bf16_norm", "edited_bf16_sha256", "donor_sha256", "random_seed")
                if key in edit},
        })
    mask_rows = {}
    for record in masks["records"]:
        key = f"{record['pair_id']}|{record['engine']}"
        clips = {}
        for clip in record["clips"]:
            wav = ROOT / clip["wav_path"]
            require(wav.is_file() and sha(wav) == clip["wav_sha256"],
                    f"Pinned input WAV changed: {clip['word']}")
            clips[clip["word"]] = {
                "wav_path": clip["wav_path"], "wav_sha256": clip["wav_sha256"]}
        mask_rows[key] = {
            "word_slot_samples": [record["word_slot_start_sample"],
                                  record["word_slot_end_sample_exclusive"]],
            "word_mel_frames_direct_overlap": record["word_mel_frames_direct_overlap"],
            "site_maps": record["site_maps"],
            "source_clips": clips,
        }
    return {
        "title": "Voxtral v2 FIT causal assay",
        "scope": "Synthetic FIT audio and constructed mel/hidden-state edits. Not a waveform attack, held-out result, or security finding.",
        "summary": result["summary"],
        "rows": compact_rows,
        "masks": mask_rows,
        "case_ids": meta["case_ids"],
        "provenance": {
            "run": str(RUN.relative_to(MODEL_DIR)),
            "result_sha256": sha(result_path), "audit_sha256": sha(audit_path),
            "meta_sha256": sha(meta_path), "runner_sha256": sha(RUNNER),
            "masks_sha256": sha(MASKS),
            "model_id": meta["snapshot"]["model_id"],
            "model_revision": meta["snapshot"]["revision"],
            "progress_rows": 864, "generations": 800, "captures": 64,
            "teacher_forced_forwards": 1600,
            "cumulative_seconds": audit["last_checkpoint_cumulative_seconds"],
            "peak_mps_bytes": audit["peak_checkpoint_mps_driver_bytes"],
        },
    }


def build(output: Path) -> dict:
    data = load_verified()
    template = TEMPLATE.read_text()
    require(template.count("__EMBEDDED_JSON__") == 1, "Template payload slot missing/duplicated")
    require(template.count("__APP_JS__") == 1, "Template application slot missing/duplicated")
    encoded = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    encoded = encoded.replace("<", "\\u003c").replace("&", "\\u0026")
    html = template.replace("__EMBEDDED_JSON__", encoded).replace("__APP_JS__", APP_JS.read_text())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html)
    return {"output": str(output), "output_sha256": sha(output),
            "embedded_cases": len(data["case_ids"]), "embedded_generations": len(data["rows"]),
            "source_result_sha256": data["provenance"]["result_sha256"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    print(json.dumps(build(args.output), indent=2))
