"""Independent saved-data audit of the frozen Whisper FIT causal assay.

This module does not import the runner or load model weights. It checks the
planned case/arm matrix, captured states, transcript decoding and controls.
It cannot prove that the recorded hook was applied by an unmodified model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import wave
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "eval/audio_jlens/corpus_v1/rendered/manifest.json"
MANIFEST_SHA = "adf8cd02d24a43ef576bcb993dc54bd2527440a62798e12fc788def3c063971f"
RUNNER = ROOT / "models/second_architecture/whisper_fit_word_window_causal_v1.py"
MODEL_ID = "openai/whisper-tiny.en"
MODEL_REVISION = "87c7102498dcde7456f24cfd30239ca606ed9063"
WEIGHTS_SHA = "db59695928ded6043adaef491a53ef4e12da9611184d77c53baa691a60b958ad"
WINDOWS = {"piper": ((90, 161), (10, 81)), "kokoro": ((101, 184), (10, 93))}
SLOTS = {"piper": (30527, 49847), "kokoro": (33920, 57120)}
WORDS = ("silver", "teal")
SITES = ("post_conv_pos", "post_block1", "post_norm")
CAPTURES = ("post_conv_pos", "post_block0", "post_block1", "post_block2", "post_block3", "post_norm")
ASSETS = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
          "generation_config.json", "preprocessor_config.json")
SEEDS = (20261118, 20261119)
EOS_ID = 50256
STATE_SHAPE = (1, 1500, 384)
MAX_RSS = 4 * 1024**3
MAX_SECONDS = 1800
HEX = re.compile(r"[0-9a-f]{64}\Z")


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def no_duplicate_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_json(raw: str) -> Any:
    return json.loads(raw, object_pairs_hook=no_duplicate_object,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"nonfinite JSON: {value}")))


def words(text: str) -> list[str]:
    return re.findall(r"[^\W_]+(?:['’][^\W_]+)?", text.casefold())


def edit_distance(a: list[str], b: list[str]) -> int:
    row = list(range(len(b) + 1))
    for i, left in enumerate(a, 1):
        next_row = [i]
        for j, right in enumerate(b, 1):
            next_row.append(min(next_row[-1] + 1, row[j] + 1,
                                row[j - 1] + (left != right)))
        row = next_row
    return row[-1]


def expected_matrix() -> set[tuple[str, str, str, str, int | None]]:
    matrix = set()
    for engine in WINDOWS:
        for recipient_word, donor_word in (("silver", "teal"), ("teal", "silver")):
            recipient, donor = f"{engine}_{recipient_word}", f"{engine}_{donor_word}"
            for site in SITES:
                for condition in ("word", "prefix", "self"):
                    matrix.add((recipient, donor, site, condition, None))
            matrix.add((recipient, donor, "post_norm", "full_donor", None))
            for seed in SEEDS:
                matrix.add((recipient, donor, "post_norm", "random", seed))
    require(len(matrix) == 48, "internal arm matrix count")
    return matrix


def trusted_inputs() -> dict[str, dict[str, Any]]:
    require(sha(MANIFEST) == MANIFEST_SHA, "manifest bytes changed")
    manifest = parse_json(MANIFEST.read_text())
    selected = [p for p in manifest["pairs"] if p.get("pair_id") == "color_05"
                and p.get("phrasing_id") == "fit_chosen"]
    require(len(selected) == 2 and {p["engine"] for p in selected} == set(WINDOWS),
            "FIT pair coverage differs")
    cases = {}
    for pair in selected:
        engine = pair["engine"]
        require(pair.get("split") == "fit_pool" and pair.get("map_fit") is True
                and pair.get("words") == list(WORDS), "non-FIT or altered word pair")
        require(pair.get("outside_slot_pcm_identical") is True
                and pair.get("inside_slot_pcm_differs") is True, "manifest PCM assertion absent")
        slot = (pair["word_slot_start_sample"], pair["word_slot_end_sample_exclusive"])
        require(slot == SLOTS[engine], "word sample slot changed")
        pcm = {}
        for word in WORDS:
            record = pair["outputs"][word]
            path = ROOT / record["path"]
            require(path.is_file() and not path.is_symlink() and sha(path) == record["file_sha256"],
                    f"WAV bytes changed: {engine}/{word}")
            with wave.open(str(path), "rb") as wav:
                require((wav.getnchannels(), wav.getsampwidth(), wav.getframerate(),
                         wav.getnframes()) == (1, 2, 16000, record["frames"]),
                        f"WAV format changed: {engine}/{word}")
                pcm[word] = wav.readframes(wav.getnframes())
            require(hashlib.sha256(pcm[word]).hexdigest() == record["pcm_int16_sha256"],
                    f"PCM bytes changed: {engine}/{word}")
            cases[f"{engine}_{word}"] = {"wav_sha256": record["file_sha256"],
                                        "pcm_sha256": record["pcm_int16_sha256"],
                                        "frames": record["frames"]}
        start, end = slot
        require(len(pcm["silver"]) == len(pcm["teal"])
                and pcm["silver"][:2 * start] == pcm["teal"][:2 * start]
                and pcm["silver"][2 * end:] == pcm["teal"][2 * end:]
                and pcm["silver"][2 * start:2 * end] != pcm["teal"][2 * start:2 * end],
                f"actual PCM slot contrast invalid: {engine}")
    return cases


def read_records(path: Path) -> list[dict[str, Any]]:
    require(path.is_file() and not path.is_symlink(), "records.jsonl absent or symlink")
    data = path.read_bytes()
    require(data.endswith(b"\n"), "records JSONL has truncated final line")
    rows = []
    for number, raw in enumerate(data.decode("utf-8").splitlines(), 1):
        row = parse_json(raw)
        require(isinstance(row, dict), f"line {number} is not a JSON object")
        rows.append(row)
    return rows


def audit_preflight(row: dict[str, Any], expected_path: Path,
                    expected_sha: str, cases: dict[str, dict[str, Any]]) -> dict[str, Any]:
    require(HEX.fullmatch(expected_sha) is not None and sha(expected_path) == expected_sha,
            "external preflight SHA mismatch")
    expected = parse_json(expected_path.read_text())
    receipt = row.get("receipt")
    require(row.get("type") == "preflight" and receipt == expected
            and row.get("expected_receipt_sha256") == expected_sha,
            "preflight record differs from external receipt")
    require(receipt.get("schema") == "whisper_fit_word_window_preflight_v1"
            and receipt.get("manifest_sha256") == MANIFEST_SHA
            and receipt.get("model_id") == MODEL_ID
            and receipt.get("model_revision") == MODEL_REVISION
            and receipt.get("checkpoint_sha256") == WEIGHTS_SHA
            and receipt.get("device") == "cpu" and receipt.get("threads") == 4,
            "preflight model or execution identity changed")
    require(sha(RUNNER) == receipt.get("script_sha256"), "runner bytes differ from preflight")
    require(receipt.get("generation") == {"do_sample": False, "num_beams": 1,
            "max_new_tokens": 128, "output_scores": True,
            "return_dict_in_generate": True}
            and receipt.get("capture_sites") == list(CAPTURES)
            and receipt.get("intervention_sites") == list(SITES),
            "preflight generation or hook sites changed")
    snapshot = (Path.home() / ".cache/huggingface/hub"
                / "models--openai--whisper-tiny.en/snapshots" / MODEL_REVISION)
    require(sha(snapshot / "model.safetensors") == WEIGHTS_SHA
            and sha(snapshot / "config.json") == receipt.get("config_sha256"),
            "cached model/config identity differs")
    asset_hashes = receipt.get("asset_sha256")
    require(isinstance(asset_hashes, dict) and set(asset_hashes) == set(ASSETS),
            "preflight asset hash inventory absent")
    for asset in ASSETS:
        require(sha(snapshot / asset) == asset_hashes[asset], f"cached asset changed: {asset}")
    inventory = {path.name: sha(path) for path in snapshot.iterdir() if path.is_file()}
    require(receipt.get("snapshot_files_sha256") == inventory,
            "cached snapshot inventory changed")
    import inspect
    from transformers.models.whisper import (feature_extraction_whisper, generation_whisper,
                                             modeling_whisper, processing_whisper,
                                             tokenization_whisper)

    modules = (feature_extraction_whisper, generation_whisper, modeling_whisper,
               processing_whisper, tokenization_whisper)
    source_hashes = {module.__name__: sha(Path(inspect.getfile(module))) for module in modules}
    require(receipt.get("source_hashes") == source_hashes,
            "installed Whisper source differs from preflight")
    recorded_cases = {row["key"]: row for row in receipt.get("cases", [])}
    require(set(recorded_cases) == set(cases), "preflight case coverage differs")
    for key, case in cases.items():
        require(recorded_cases[key].get("file_sha256") == case["wav_sha256"]
                and recorded_cases[key].get("pcm_sha256") == case["pcm_sha256"]
                and recorded_cases[key].get("frames") == case["frames"],
                f"preflight WAV identity differs: {key}")
    for engine, (word_window, prefix_window) in WINDOWS.items():
        entry = receipt.get("windows", {}).get(engine, {})
        require(entry.get("word_window") == list(word_window)
                and entry.get("prefix_window") == list(prefix_window)
                and entry.get("slot_samples") == list(SLOTS[engine]),
                f"preflight window changed: {engine}")
    return receipt


def checked_result(value: dict[str, Any], tokenizer: Any | None = None) -> dict[str, Any]:
    require(isinstance(value, dict), "result is not object")
    ids = value.get("token_ids")
    require(isinstance(ids, list) and 2 <= len(ids) <= 132
            and all(type(token) is int and 0 <= token < 51864 for token in ids),
            "token IDs invalid")
    transcript = value.get("transcript")
    require(isinstance(transcript, str) and len(transcript) <= 2000, "transcript invalid")
    if tokenizer is not None:
        decoded = tokenizer.batch_decode([ids], skip_special_tokens=True)[0].strip()
        require(decoded == transcript, "saved transcript does not decode from token IDs")
    eos_seen = EOS_ID in ids
    require(value.get("eos_seen") is eos_seen and value.get("censored") is (not eos_seen),
            "EOS/censor flags differ from IDs")
    scores = value.get("step_scores")
    require(isinstance(scores, list) and len(scores) <= 128
            and len(scores) <= len(ids), "step-score length invalid")
    for i, score in enumerate(scores):
        require(score.get("token_id") == ids[len(ids) - len(scores) + i]
                and len(score.get("top2_ids", [])) == 2
                and len(score.get("top2_scores", [])) == 2,
                "step-score token alignment invalid")
        for number in [score.get("selected_score"), *score["top2_scores"]]:
            require(isinstance(number, (int, float)) and math.isfinite(number),
                    "nonfinite step score")
    require(isinstance(value.get("elapsed_seconds"), (int, float))
            and 0 <= value["elapsed_seconds"] <= MAX_SECONDS
            and type(value.get("rss_bytes")) is int
            and 0 < value["rss_bytes"] <= MAX_RSS,
            "per-decode time or RSS invalid")
    return {"token_ids": ids, "transcript": transcript, "words": words(transcript),
            "eos_seen": eos_seen, "censored": not eos_seen}


def load_capture(path: Path, expected_sha: str) -> np.ndarray:
    require(path.is_file() and not path.is_symlink() and sha(path) == expected_sha,
            f"capture hash mismatch: {path.name}")
    array = np.load(path, allow_pickle=False)
    require(array.shape == STATE_SHAPE and array.dtype == np.float32
            and bool(np.isfinite(array).all()), f"capture shape/dtype/nonfinite: {path.name}")
    return array


def load_mel(path: Path, expected_sha: str) -> np.ndarray:
    require(path.is_file() and not path.is_symlink() and sha(path) == expected_sha,
            f"mel capture hash mismatch: {path.name}")
    array = np.load(path, allow_pickle=False)
    require(array.shape == (1, 80, 3000) and array.dtype == np.float32
            and bool(np.isfinite(array).all()), f"mel shape/dtype/nonfinite: {path.name}")
    return array


def audit_traces(trace_rows: list[dict[str, Any]], output: Path,
                 captures: dict[str, dict[str, np.ndarray]]) -> dict[str, Any]:
    result = {}
    for row in trace_rows:
        engine = row.get("engine")
        require(engine in WINDOWS and engine not in result and row.get("words") == list(WORDS),
                "trace engine/words duplicated or changed")
        recorded = row.get("traces")
        require(isinstance(recorded, dict) and set(recorded) == {"mel", *CAPTURES},
                "trace site inventory differs")
        result[engine] = {}
        silver, teal = captures[f"{engine}_silver"], captures[f"{engine}_teal"]
        for site, saved in recorded.items():
            expected_name = f"captures/{engine}_{site}_silver_teal_l2.npy"
            require(saved.get("path") == expected_name, "trace path changed")
            path = output / expected_name
            require(path.is_file() and not path.is_symlink()
                    and sha(path) == saved.get("sha256"), "trace SHA differs")
            actual = np.load(path, allow_pickle=False)
            require(actual.dtype == np.float32 and actual.ndim == 1
                    and bool(np.isfinite(actual).all()), "trace dtype/finite invalid")
            if site == "mel":
                expected = np.linalg.norm(silver[site] - teal[site], axis=1)[0]
                word_window = tuple(value * 2 for value in WINDOWS[engine][0])
                prefix = tuple(value * 2 for value in WINDOWS[engine][1])
            else:
                expected = np.linalg.norm(silver[site] - teal[site], axis=-1)[0]
                word_window, prefix = WINDOWS[engine]
            require(actual.shape == expected.shape and saved.get("shape") == list(expected.shape)
                    and np.allclose(actual, expected, rtol=1e-5, atol=1e-6),
                    "trace differs from captured donor-recipient states")
            require(saved.get("word_window") == list(word_window)
                    and saved.get("prefix_window") == list(prefix), "trace window differs")
            outside = np.ones(len(expected), dtype=bool)
            outside[slice(*word_window)] = False
            outside[slice(*prefix)] = False
            for field, measured in (
                ("word_l2_sum", float(expected[slice(*word_window)].sum())),
                ("prefix_l2_sum", float(expected[slice(*prefix)].sum())),
                ("remaining_l2_sum", float(expected[outside].sum())),
            ):
                reported = saved.get(field)
                require(isinstance(reported, (int, float)) and math.isfinite(reported)
                        and math.isclose(reported, measured, rel_tol=1e-5, abs_tol=1e-5),
                        f"trace {field} differs from captures")
            first = int(np.flatnonzero(expected > 0)[0]) if np.any(expected > 0) else None
            require(saved.get("first_nonzero_frame") == first,
                    "trace first-nonzero frame differs")
            result[engine][site] = {"first_nonzero_frame": first,
                                    "word_l2_sum": float(expected[slice(*word_window)].sum()),
                                    "prefix_l2_sum": float(expected[slice(*prefix)].sum()),
                                    "remaining_l2_sum": float(expected[outside].sum())}
    require(set(result) == set(WINDOWS), "trace engine coverage incomplete")
    return result


def recompute_outcome(transcript: str, recipient_word: str,
                      donor_word: str, baseline: str) -> dict[str, Any]:
    observed, base = words(transcript), words(baseline)
    expected = [donor_word if token == recipient_word else token for token in base]
    return {
        "target_replacement": observed == expected and base.count(recipient_word) == 1
                              and donor_word not in base,
        "donor_once_recipient_absent": observed.count(donor_word) == 1
                                      and observed.count(recipient_word) == 0,
        "recipient_transcript_match": observed == base,
        "normalized_words": observed,
    }


def audit_random_patch(row: dict[str, Any], recipient_state: np.ndarray,
                       donor_state: np.ndarray, word_window: tuple[int, int]) -> None:
    import torch

    start, end = word_window
    recipient = torch.from_numpy(recipient_state[:, start:end, :].copy())
    donor = torch.from_numpy(donor_state[:, start:end, :].copy())
    target = torch.linalg.vector_norm(donor - recipient)
    require(bool(torch.isfinite(target)) and float(target) > 0,
            "random target norm is zero/nonfinite")
    seed = row.get("seed")
    require(seed in SEEDS, "random seed differs from plan")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(tuple(recipient.shape), generator=generator, dtype=torch.float32)
    noise *= target / torch.linalg.vector_norm(noise)
    edit_sha = hashlib.sha256(noise.contiguous().numpy().tobytes()).hexdigest()
    patch = row.get("patch_audit", {})
    require(patch.get("random_edit_sha256") == edit_sha
            and math.isclose(patch.get("random_target_norm", -1), float(target), rel_tol=1e-5, abs_tol=1e-6)
            and math.isclose(patch.get("random_edit_norm", -1), float(torch.linalg.vector_norm(noise)),
                             rel_tol=1e-5, abs_tol=1e-6),
            "random norm or edit hash differs from saved states")


def audit_rows(rows: list[dict[str, Any]], output: Path, cases: dict[str, dict[str, Any]],
               tokenizer: Any | None = None) -> dict[str, Any]:
    require(len(rows) == 56 and rows[0].get("type") == "preflight"
            and rows[-1].get("type") == "complete", "planned result length/order differs")
    require(all(row.get("type") == "baseline" for row in rows[1:5])
            and all(row.get("type") == "trace" for row in rows[5:7])
            and all(row.get("type") == "intervention" for row in rows[7:-1]),
            "baseline/trace/intervention ordering differs")
    baselines = {}
    captures = {}
    for row in rows[1:5]:
        key = row.get("case")
        require(key in cases and key not in baselines, "baseline case duplicate/unknown")
        baselines[key] = checked_result(row.get("result"), tokenizer)
        word = key.rsplit("_", 1)[1]
        alternative = WORDS[1 - WORDS.index(word)]
        require(baselines[key]["words"].count(word) == 1
                and alternative not in baselines[key]["words"]
                and baselines[key]["eos_seen"] and row.get("eligible") is True,
                f"clean baseline ineligible: {key}")
        require(row.get("capture_complete") is True,
                "baseline capture incomplete")
        require(row.get("capture_hook_counts") == dict.fromkeys(CAPTURES, 1)
                and row["result"].get("hook_count") is None,
                "baseline capture hook count differs")
        hashes = row.get("capture_sha256")
        require(isinstance(hashes, dict) and set(hashes) == {"mel", *CAPTURES},
                "capture hash inventory invalid")
        captures[key] = {site: load_capture(output / "captures" / f"{key}_{site}.npy", hashes[site])
                         for site in CAPTURES}
        captures[key]["mel"] = load_mel(output / "captures" / f"{key}_mel.npy", hashes["mel"])
    require(set(baselines) == set(cases), "baseline coverage incomplete")

    trace = audit_traces(rows[5:7], output, captures)

    actual: set[tuple[str, str, str, str, int | None]] = set()
    outcomes: list[dict[str, Any]] = []
    for row in rows[7:-1]:
        recipient, donor, site, condition = (row.get(key) for key in
                                              ("recipient", "donor", "site", "condition"))
        seed = row.get("seed")
        key = (recipient, donor, site, condition, seed)
        require(key in expected_matrix() and key not in actual,
                "intervention missing, duplicate, or not prespecified")
        actual.add(key)
        engine, rec_word = recipient.rsplit("_", 1)
        donor_engine, donor_word = donor.rsplit("_", 1)
        require(engine == donor_engine and rec_word != donor_word,
                "cross-engine or same-word donor")
        word_window, prefix = WINDOWS[engine]
        expected_window = ((0, 1500) if condition == "full_donor" else
                           prefix if condition == "prefix" else word_window)
        require(row.get("window") == list(expected_window), "intervention window changed")
        require(row.get("hook_count") == 1, "intervention hook did not fire once")
        require(row.get("valid_hook") is True, "intervention hook validity absent/false")
        result = checked_result(row.get("result"), tokenizer)
        require(row["result"].get("hook_count") == 1, "result hook count differs")
        source = recipient if condition == "self" else donor
        rec_hashes = next(item["capture_sha256"] for item in rows[1:5] if item["case"] == recipient)
        source_hashes = next(item["capture_sha256"] for item in rows[1:5] if item["case"] == source)
        require(row.get("recipient_state_sha256") == rec_hashes[site]
                and row.get("source_state_sha256") == source_hashes[site],
                "intervention source/recipient capture hash differs")
        baseline = baselines[recipient]
        independent = recompute_outcome(result["transcript"], rec_word, donor_word,
                                        baseline["transcript"])
        independent["target_replacement"] = independent["target_replacement"] and result["eos_seen"]
        independent["token_difference_count"] = (sum(left != right for left, right in zip(
            result["token_ids"], baseline["token_ids"]))
            + abs(len(result["token_ids"]) - len(baseline["token_ids"])))
        require(row.get("outcome") == independent, "stored outcome differs from raw transcript")
        if condition == "self":
            require(row.get("control_pass") is True,
                    "self-patch control pass flag absent/false")
            require(result["token_ids"] == baseline["token_ids"],
                    "self-patch changed complete token sequence")
        elif condition == "full_donor":
            require(row.get("control_pass") is True,
                    "full-donor control pass flag absent/false")
            require(result["token_ids"] == baselines[donor]["token_ids"],
                    "full donor did not reproduce donor tokens")
        else:
            require(row.get("control_pass") is None,
                    "noncontrol arm has control pass flag")
        if condition == "random":
            audit_random_patch(row, captures[recipient]["post_norm"],
                               captures[donor]["post_norm"], word_window)
        elif condition == "full_donor":
            require(row.get("patch_audit") == {}, "full donor has unexpected patch audit")
        outcomes.append({"recipient": recipient, "donor": donor, "site": site,
                         "condition": condition, "seed": seed,
                         "target_replacement": independent["target_replacement"],
                         "donor_word_appears": donor_word in result["words"],
                         "wer_vs_recipient": edit_distance(baseline["words"], result["words"])
                                             / max(1, len(baseline["words"])),
                         "eos_seen": result["eos_seen"]})
    require(actual == expected_matrix(), "intervention matrix incomplete")

    complete = rows[-1]
    require(isinstance(complete.get("elapsed_seconds"), (int, float))
            and 0 <= complete["elapsed_seconds"] <= MAX_SECONDS
            and type(complete.get("rss_bytes")) is int
            and 0 < complete["rss_bytes"] <= MAX_RSS, "campaign cap exceeded")
    counts = Counter((row["site"], row["condition"], row["target_replacement"])
                     for row in outcomes)
    return {"status": "independent_saved_data_audit_pass", "records_sha256": sha(output / "records.jsonl"),
            "input_manifest_sha256": MANIFEST_SHA, "baseline_count": len(baselines),
            "intervention_count": len(outcomes), "outcome_counts": [
                {"site": site, "condition": condition, "target_replacement": replacement, "count": count}
                for (site, condition, replacement), count in sorted(counts.items())],
            "outcomes": outcomes, "trace": trace,
            "limits": "Saved-data consistency only; hook/model execution cannot be independently proven from JSON and captures."}


def audit_failed_attempt(rows: list[dict[str, Any]], output: Path,
                         tokenizer: Any | None = None) -> dict[str, Any]:
    """Preserve a stopped attempt without awarding any scientific result."""
    require(rows and rows[-1].get("type") == "failure"
            and sum(row.get("type") == "failure" for row in rows) == 1
            and not any(row.get("type") == "complete" for row in rows),
            "failure record is not terminal and unique")
    failure = rows[-1]
    require(isinstance(failure.get("error_type"), str)
            and isinstance(failure.get("error"), str)
            and isinstance(failure.get("elapsed_seconds"), (int, float))
            and math.isfinite(failure["elapsed_seconds"]), "failure detail invalid")
    require(all(row.get("type") in {"preflight", "baseline", "trace", "intervention"}
                for row in rows[:-1]), "failed attempt contains unexpected row type")
    require(len(rows) <= 56, "failed attempt exceeds planned row count")
    baseline_rows = {row["case"]: row for row in rows if row.get("type") == "baseline"}
    arm_rows = [row for row in rows if row.get("type") == "intervention"]
    explicit_control_failures = []
    for row in arm_rows:
        if row.get("condition") not in {"self", "full_donor"}:
            continue
        if row.get("control_pass") is not False:
            continue
        result = checked_result(row.get("result"), tokenizer)
        reference = row.get("recipient") if row["condition"] == "self" else row.get("donor")
        require(reference in baseline_rows, "failed control lacks baseline")
        baseline = checked_result(baseline_rows[reference].get("result"), tokenizer)
        require(result["token_ids"] != baseline["token_ids"],
                "false control flag contradicts raw token IDs")
        explicit_control_failures.append({"recipient": row.get("recipient"),
                                          "condition": row["condition"],
                                          "site": row.get("site")})
    ineligible_baselines = [row["case"] for row in baseline_rows.values()
                            if row.get("eligible") is False]
    return {"status": "invalid_or_incomplete_attempt", "records_sha256": sha(output / "records.jsonl"),
            "baseline_rows": len(baseline_rows), "intervention_rows": len(arm_rows),
            "failure": {"error_type": failure["error_type"], "error": failure["error"]},
            "explicit_control_failures": explicit_control_failures,
            "ineligible_baselines": ineligible_baselines,
            "scientific_outcome_available": False}


def audit(output: Path, expected_preflight: Path, expected_preflight_sha256: str) -> dict[str, Any]:
    cases = trusted_inputs()
    rows = read_records(output / "records.jsonl")
    require(rows and rows[0].get("type") == "preflight", "preflight first row absent")
    receipt = audit_preflight(rows[0], expected_preflight, expected_preflight_sha256, cases)
    # Tokenizer is loaded offline, without the checkpoint weights, to verify that
    # each saved text is actually decoded from its saved token sequence.
    from transformers import WhisperProcessor
    processor = WhisperProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION,
                                                 local_files_only=True)
    tokenizer = processor.tokenizer
    if rows[-1].get("type") == "failure":
        return audit_failed_attempt(rows, output, tokenizer)
    return audit_rows(rows, output, cases, tokenizer)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-preflight", type=Path, required=True)
    parser.add_argument("--expected-preflight-sha256", required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    args = parser.parse_args()
    if args.audit_output.exists():
        raise FileExistsError("audit output exists; preserve prior attempt")
    report = audit(args.output, args.expected_preflight, args.expected_preflight_sha256)
    args.audit_output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(args.audit_output)


if __name__ == "__main__":
    main()
