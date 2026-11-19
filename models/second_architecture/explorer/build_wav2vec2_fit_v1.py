"""Build an offline explorer from the frozen Wav2Vec2 FIT evidence.

The builder reads only committed receipts and locally retained audited tensors.
It refuses incomplete, changed, or unreviewed source data. It is a presentation
layer: no model weights are loaded and no scientific outcome is recomputed here.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE.parent
ROOT = MODEL_DIR.parents[1]
BASELINE = MODEL_DIR / "runs/wav2vec2_fit_baseline_stage_capture_v1/20261118T2352Z"
CAUSAL = MODEL_DIR / "runs/wav2vec2_fit_causal_patch_v1/20261119T0013Z"
COMPLEMENT = MODEL_DIR / "runs/wav2vec2_fit_complement_patch_v1/20261119T0035Z"
CORPUS = ROOT / "eval/audio_jlens/corpus_v1/rendered/manifest.json"
BASELINE_INVENTORY = MODEL_DIR / "WAV2VEC2_FIT_BASELINE_V2_SIDECAR_MANIFEST_V1.json"
CAUSAL_INVENTORY = MODEL_DIR / "WAV2VEC2_FIT_CAUSAL_SIDECAR_MANIFEST_V1.json"
CAUSAL_REVIEW = MODEL_DIR / "WAV2VEC2_FIT_CAUSAL_RESULT_INDEPENDENT_REVIEW_V1.md"
COMPLEMENT_INVENTORY = MODEL_DIR / "WAV2VEC2_FIT_COMPLEMENT_SIDECAR_MANIFEST_V1.json"
COMPLEMENT_REVIEW = MODEL_DIR / "WAV2VEC2_FIT_COMPLEMENT_RESULT_INDEPENDENT_REVIEW_V1.md"
TEMPLATE = HERE / "wav2vec2_fit_v1_template.html"
APP = HERE / "wav2vec2_fit_v1.js"
OUTPUT = HERE / "wav2vec2_fit_v1.html"

EXPECTED_SHA = {
    "baseline_records": "c9865f4ec306e57baf7dbfc7bd05037ec769034d95db70440972904fcab2507c",
    "baseline_audit": "c84c17cdefbebe1007f106cb955f910e7a4ce0431745e59a4528fc2aae41bc49",
    "baseline_inventory": "2f0484205d21b596795fbfbf30c1e65a63451dafc8740e5860c95be20ace2b53",
    "causal_records": "bacfc2bcdfe7239b960200735758c708389a7762a65761eb185c5caaa6e60285",
    "causal_audit": "fcfa865d56e0ebb1b719e8a890edd77fdb5aa39f2ae08aef782ff1b0d6eeb052",
    "causal_inventory": "1785a336a1016e7ec689c9a670532238b29f117e074e54bd45d0078ef8c570b6",
    "causal_review": "fbfbef6a201a0e9978ea86e9c41be6300f04ebcadc36b8190e80a181c8df5d11",
    "complement_records": "3eff05935744803e5eb3022a7417a4c3304e2465c148ad84613df53ba0392766",
    "complement_audit": "606ef0aae4af76b7903dcac72a8d51b8be7bd58e99bdf7b0481d22dcd91a4dbe",
    "complement_inventory": "92dc6d42ca718b98764d9bfba38c98ac71d2b7fe903d18325c34e0d95da0db6a",
    "complement_review": "5a37977c6e84a9b93afffe5f41951c18f3b63e4b99955973e930729e48f45e06",
}
STAGES = ("input_values", "first_conv_raw", "first_conv_norm",
          "feature_projection", "post_block11")
ARMS = ("W", "P", "Q", "self_W", "full_donor", "R1", "R2")
SITES = ("feature_projection", "post_block11")


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    raw = path.read_bytes()
    require(raw.endswith(b"\n"), f"truncated JSONL: {path}")
    return [json.loads(line) for line in raw.decode("utf-8").splitlines()]


def max_pool(values: np.ndarray, bins: int = 480) -> list[float]:
    """Preserve the maximum difference in each display bin, including spikes."""
    require(values.ndim == 1 and values.size > 0 and np.isfinite(values).all()
            and (values >= 0).all(), "invalid nonnegative trace")
    n = int(values.size)
    result = []
    for i in range(min(bins, n)):
        left = i * n // min(bins, n)
        right = (i + 1) * n // min(bins, n)
        result.append(float(values[left:right].max()))
    return result


def verified_file(path: Path, expected: str) -> None:
    require(path.is_file() and not path.is_symlink() and sha(path) == expected,
            f"missing or changed source file: {path}")


def verified_complement() -> list[dict]:
    for key, path in (
        ("complement_records", COMPLEMENT / "records.jsonl"),
        ("complement_audit", COMPLEMENT / "audit.json"),
        ("complement_inventory", COMPLEMENT_INVENTORY),
        ("complement_review", COMPLEMENT_REVIEW),
    ):
        verified_file(path, EXPECTED_SHA[key])
    require("Decision: GO" in COMPLEMENT_REVIEW.read_text(),
            "independent complement result review lacks GO")
    audit = json.loads((COMPLEMENT / "audit.json").read_text())
    require(audit["status"] == "independent_saved_data_audit_pass"
            and audit["control_gate_pass"] is True
            and audit["records_sha256"] == EXPECTED_SHA["complement_records"]
            and audit["patch_rows"] == 192 and audit["sidecar_count"] == 384,
            "complement audit is not complete")
    rows = read_jsonl(COMPLEMENT / "records.jsonl")
    require(len(rows) == 194 and rows[0]["type"] == "preflight"
            and rows[-1]["type"] == "complete", "complement bookends differ")
    patches = [row for row in rows if row["type"] == "patch"]
    require(len(patches) == 192
            and len({(r["direction_index"], r["site"], r["arm"]) for r in patches}) == 192
            and {r["site"] for r in patches} == set(SITES)
            and {r["arm"] for r in patches} == {"W", "C", "self_C", "full_donor", "R1_C", "R2_C"},
            "complement patch matrix differs")
    manifest = json.loads(COMPLEMENT_INVENTORY.read_text())
    require(manifest["records_sha256"] == EXPECTED_SHA["complement_records"]
            and manifest["audit_sha256"] == EXPECTED_SHA["complement_audit"]
            and len(manifest["files"]) == 384,
            "complement sidecar inventory differs")
    by_path = {item["path"]: item for item in manifest["files"]}
    require(len(by_path) == 384, "complement sidecar path duplicated")
    expected = set()
    for row in patches:
        for kind in ("patch", "logits"):
            rel = row[f"{kind}_path"]
            require(rel.startswith("captures/") and ".." not in Path(rel).parts
                    and rel not in expected and rel in by_path,
                    "complement sidecar path invalid or duplicated")
            expected.add(rel)
            item = by_path[rel]
            require(item["sha256"] == row[f"{kind}_sha256"]
                    and (COMPLEMENT / rel).stat().st_size == item["size_bytes"],
                    "complement sidecar metadata differs")
            verified_file(COMPLEMENT / rel, item["sha256"])
    actual = {p.relative_to(COMPLEMENT).as_posix() for p in COMPLEMENT.rglob("*.npy")}
    require(actual == expected == set(by_path), "complement sidecar population differs")
    return patches


def load_verified() -> dict:
    complement_patches = verified_complement()
    for key, path in (
        ("baseline_records", BASELINE / "records.jsonl"),
        ("baseline_audit", BASELINE / "audit.json"),
        ("baseline_inventory", BASELINE_INVENTORY),
        ("causal_records", CAUSAL / "records.jsonl"),
        ("causal_audit", CAUSAL / "audit.json"),
        ("causal_inventory", CAUSAL_INVENTORY),
    ):
        verified_file(path, EXPECTED_SHA[key])
    require(bool(EXPECTED_SHA["causal_review"]), "independent result review SHA not pinned")
    verified_file(CAUSAL_REVIEW, EXPECTED_SHA["causal_review"])
    review_text = CAUSAL_REVIEW.read_text()
    require("Decision: GO" in review_text, "independent causal result review lacks GO")

    baseline_audit = json.loads((BASELINE / "audit.json").read_text())
    causal_audit = json.loads((CAUSAL / "audit.json").read_text())
    require(baseline_audit["status"] == "independent_saved_data_audit_pass"
            and baseline_audit["records_sha256"] == EXPECTED_SHA["baseline_records"]
            and baseline_audit["baseline_count"] == 16,
            "baseline audit is not complete")
    require(causal_audit["status"] == "independent_saved_data_audit_pass"
            and causal_audit["records_sha256"] == EXPECTED_SHA["causal_records"]
            and causal_audit["patch_rows"] == 224
            and causal_audit["sidecar_count"] == 448
            and causal_audit["control_gate_pass"] is True,
            "causal audit is not complete or controls failed")

    base_rows = read_jsonl(BASELINE / "records.jsonl")
    causal_rows = read_jsonl(CAUSAL / "records.jsonl")
    require(len(base_rows) == 26 and len(causal_rows) == 226,
            "frozen run row counts differ")
    require(base_rows[0]["type"] == causal_rows[0]["type"] == "preflight"
            and base_rows[-1]["type"] == causal_rows[-1]["type"] == "complete",
            "run bookends differ")
    base_cases = {x["case_key"]: x for x in base_rows if x["type"] == "baseline" and x["mode"] == "S"}
    base_traces = {(x["pair_id"], x["engine"], x["mode"]): x
                   for x in base_rows if x["type"] == "trace"}
    patches = [x for x in causal_rows if x["type"] == "patch"]
    require(len(base_cases) == 8 and len(base_traces) == 8 and len(patches) == 224,
            "baseline/patch population differs")
    require(len({(x["direction_index"], x["site"], x["arm"]) for x in patches}) == 224,
            "duplicate causal matrix cell")
    require(set(x["site"] for x in patches) == set(SITES)
            and set(x["arm"] for x in patches) == set(ARMS),
            "causal site/arm matrix differs")

    # Rehash all baseline sidecars, not just the plotted traces, and reject extras.
    baseline_manifest = json.loads(BASELINE_INVENTORY.read_text())
    baseline_files = baseline_manifest["files"]
    require(len(baseline_files) == 320 and baseline_manifest["sidecar_count"] == 320,
            "baseline sidecar inventory count differs")
    expected_baseline = set()
    for item in baseline_files:
        rel = item["path"]
        require(rel.startswith(("captures/", "traces/"))
                and ".." not in Path(rel).parts and rel not in expected_baseline,
                "baseline inventory path invalid or duplicated")
        expected_baseline.add(rel)
        path = BASELINE / rel
        require(path.stat().st_size == item["size_bytes"], f"size changed: {rel}")
        verified_file(path, item["sha256"])
    actual_baseline = {p.relative_to(BASELINE).as_posix() for p in BASELINE.rglob("*.npy")}
    require(actual_baseline == expected_baseline, "baseline sidecar inventory differs")
    trace_hashes = {item["path"]: item["sha256"] for item in baseline_files}

    # The independent reviewer inventoried every patch/logit sidecar. Match both
    # that inventory and each run record before displaying any result.
    causal_manifest = json.loads(CAUSAL_INVENTORY.read_text())
    require(causal_manifest["records_sha256"] == EXPECTED_SHA["causal_records"]
            and causal_manifest["audit_sha256"] == EXPECTED_SHA["causal_audit"]
            and causal_manifest["sidecar_count"] == 448,
            "causal independent inventory binding differs")
    causal_manifest_files = {item["path"]: item for item in causal_manifest["files"]}
    require(len(causal_manifest_files) == 448, "causal inventory duplicate/missing path")
    expected_causal = set()
    for row in patches:
        for field in ("patch", "logits"):
            rel = row[f"{field}_path"]
            require(rel.startswith("captures/") and ".." not in Path(rel).parts
                    and rel not in expected_causal, "causal sidecar path invalid or duplicated")
            expected_causal.add(rel)
            item = causal_manifest_files[rel]
            require(item["sha256"] == row[f"{field}_sha256"]
                    and (CAUSAL / rel).stat().st_size == item["size_bytes"],
                    "causal manifest/record sidecar differs")
            verified_file(CAUSAL / rel, row[f"{field}_sha256"])
    actual_causal = {p.relative_to(CAUSAL).as_posix() for p in CAUSAL.rglob("*.npy")}
    require(len(expected_causal) == 448 and actual_causal == expected_causal
            == set(causal_manifest_files),
            "causal sidecar inventory differs")

    corpus = json.loads(CORPUS.read_text())
    require(sha(CORPUS) == base_rows[0]["receipt"]["manifest_sha256"],
            "corpus manifest changed")
    clips: dict[str, dict] = {}
    for case in base_rows[0]["receipt"]["cases"]:
        matches = [p for p in corpus["pairs"] if p["pair_id"] == case["pair_id"]
                   and p["engine"] == case["engine"] and p["phrasing_id"] == "fit_chosen"]
        require(len(matches) == 1, "source clip pair missing/duplicated")
        info = matches[0]["outputs"][case["word"]]
        require(info["frames"] == case["frames"]
                and info["file_sha256"] == case["wav_sha256"]
                and [matches[0]["word_slot_start_sample"],
                     matches[0]["word_slot_end_sample_exclusive"]] == case["slot_samples"],
                "source clip metadata differs")
        wav = ROOT / info["path"]
        verified_file(wav, info["file_sha256"])
        clips[case["key"]] = {"path": "../../../" + info["path"],
                              "sha256": info["file_sha256"],
                              "word": case["word"], "frames": case["frames"],
                              "slot_samples": case["slot_samples"]}
    require(len(clips) == 8, "source clip count differs")

    traces: dict[str, dict] = {}
    for (pair, engine, mode), row in sorted(base_traces.items()):
        stages = {}
        for stage in STAGES:
            info = row["stages"][stage]
            rel = f"traces/{mode}_{pair}_{engine}_{stage}_maxabs.npy"
            require(trace_hashes[rel] == info["maxabs_sha256"],
                    "trace row/manifest hash differs")
            values = np.load(BASELINE / rel, allow_pickle=False)
            require(values.ndim == 1 and values.dtype == np.float32
                    and math.isclose(float(values.max()), max(max_pool(values)),
                                     rel_tol=1e-6, abs_tol=1e-6),
                    "trace extrema/dtype differs")
            stages[stage] = {"points": max_pool(values), "length": int(values.size),
                             "nominal_window": info["nominal_window"],
                             "first_nonzero_index": info["first_nonzero_index"],
                             "offslot_maxabs": info["offslot_maxabs"],
                             "full_maxabs": float(values.max())}
        traces[f"{pair}|{engine}|{mode}"] = stages

    compact_patches = []
    for row in patches:
        compact_patches.append({k: row[k] for k in (
            "direction_index", "pair_id", "engine", "mode", "recipient_key",
            "donor_key", "site", "arm", "transcript", "primary_eligible",
            "target_only_transfer", "control_pass", "windows", "delta_l2",
            "changed_frames_vs_recipient", "logits_l2_vs_recipient",
            "logits_l2_vs_donor", "seed")})
    causal_by_key = {(r["direction_index"], r["site"], r["arm"]): r for r in patches}
    compact_complement = []
    for row in complement_patches:
        reference = causal_by_key[(row["direction_index"], row["site"], "W")]
        require(all(row[key] == reference[key] for key in
                    ("pair_id", "engine", "mode", "recipient_key", "donor_key", "site"))
                and (row["arm"] != "W" or row["transcript"] == reference["transcript"]),
                "complement and prior causal cohort differ")
        compact_complement.append({k: row[k] for k in (
            "direction_index", "pair_id", "engine", "mode", "recipient_key",
            "donor_key", "site", "arm", "transcript", "primary_eligible",
            "target_only_transfer", "control_pass", "delta_l2",
            "natural_c_l2", "natural_w_l2")})
    baselines = []
    for row in base_rows:
        if row["type"] == "baseline":
            baselines.append({k: row[k] for k in (
                "mode", "case_key", "pair_id", "engine", "word", "transcript",
                "eligible", "normalized_words")})
    return {
        "scope": "Synthetic FIT audio, one English CTC ASR model, internal patches. Shared normalization is a constructed counterfactual. This is neither held-out validation nor an attacker-controlled audio/security finding.",
        "model": base_rows[0]["receipt"]["model_id"],
        "revision": base_rows[0]["receipt"]["model_revision"],
        "clips": clips, "traces": traces, "baselines": baselines,
        "patches": compact_patches, "complement_patches": compact_complement,
        "provenance": {"baseline_records_sha256": EXPECTED_SHA["baseline_records"],
                       "causal_records_sha256": EXPECTED_SHA["causal_records"],
                       "baseline_audit_sha256": EXPECTED_SHA["baseline_audit"],
                       "causal_audit_sha256": EXPECTED_SHA["causal_audit"],
                       "causal_sidecar_manifest_sha256": EXPECTED_SHA["causal_inventory"],
                       "causal_review_sha256": EXPECTED_SHA["causal_review"],
                       "complement_records_sha256": EXPECTED_SHA["complement_records"],
                       "complement_audit_sha256": EXPECTED_SHA["complement_audit"],
                       "complement_review_sha256": EXPECTED_SHA["complement_review"],
                       "baseline_sidecars": 320, "causal_sidecars": 448,
                       "planned_patch_rows": 224, "primary_directions_per_site": 4},
    }


def build(output: Path = OUTPUT) -> dict:
    data = load_verified()
    template = TEMPLATE.read_text()
    require(template.count("__EMBEDDED_JSON__") == 1
            and template.count("__APP_JS__") == 1,
            "template payload slots differ")
    encoded = json.dumps(data, ensure_ascii=False, separators=(", ", ":"), allow_nan=False)
    encoded = encoded.replace("<", "\\u003c").replace("&", "\\u0026")
    html = template.replace("__EMBEDDED_JSON__", encoded).replace("__APP_JS__", APP.read_text())
    output.write_text(html)
    return {"output": str(output), "output_sha256": sha(output),
            "patches": len(data["patches"]), "clips": len(data["clips"]),
            "traces": len(data["traces"])}


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))
