"""Independent saved-data audit of the Whisper FIT spread-versus-use assay.

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
import subprocess
import wave
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "eval/audio_jlens/corpus_v1/rendered/manifest.json"
MANIFEST_SHA = "adf8cd02d24a43ef576bcb993dc54bd2527440a62798e12fc788def3c063971f"
ASR_SCREEN = ROOT / "eval/audio_jlens/corpus_v1/asr_audit.json"
ASR_SCREEN_SHA = "41da6ee4dd67a2f7b06ece24354fdb0ca6911f7017ec514e170f55f61d38a5e7"
RUNNER = ROOT / "models/second_architecture/whisper_fit_spread_causal_v1.py"
MODEL_ID = "openai/whisper-tiny.en"
MODEL_REVISION = "87c7102498dcde7456f24cfd30239ca606ed9063"
WEIGHTS_SHA = "db59695928ded6043adaef491a53ef4e12da9611184d77c53baa691a60b958ad"
PAIRS = {
    "color_05": {"words": ("silver", "teal"),
                 "piper": {"W": (90, 161), "P": (10, 81), "Q": (171, 242), "N": 272,
                           "slot": (30527, 49847)},
                 "kokoro": {"W": (101, 184), "P": (10, 93), "Q": (194, 277), "N": 287,
                            "slot": (33920, 57120)}},
    "object_02": {"words": ("plate", "cup"),
                  "piper": {"W": (90, 139), "P": (10, 59), "Q": (149, 198), "N": 250,
                            "slot": (30527, 42788)},
                  "kokoro": {"W": (101, 175), "P": (10, 84), "Q": (185, 259), "N": 279,
                             "slot": (33920, 54320)}}}
ENGINES = ("piper", "kokoro")
SITES = ("post_block0", "post_norm")
CAPTURES = ("post_conv_pos", "post_block0", "post_block1", "post_block2", "post_block3", "post_norm")
ASSETS = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
          "generation_config.json", "preprocessor_config.json")
SEEDS = (20261118, 20261119)
SEED_PREFIX = "whisper-fit-spread-v1"
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


def case_key(pair_id: str, engine: str, word: str) -> str:
    return f"{pair_id}_{engine}_{word}"


def region_spans(pair_id: str, engine: str, condition: str) -> list[list[int]]:
    region = PAIRS[pair_id][engine]
    if condition in {"word", "self"}:
        return [list(region["W"])]
    if condition == "prefix":
        return [list(region["P"])]
    if condition == "postword":
        return [list(region["Q"])]
    if condition in {"complement", "random"}:
        return [[0, region["W"][0]], [region["W"][1], region["N"]]]
    if condition == "full_donor":
        return [[0, 1500]]
    raise ValueError(f"unknown region: {condition}")


def derived_arm_seed(pair_id: str, engine: str, recipient_word: str,
                     donor_word: str, site: str, base_seed: int) -> int:
    require(pair_id in PAIRS and engine in ENGINES and site in SITES
            and (recipient_word, donor_word) in (tuple(PAIRS[pair_id]["words"]),
                                                   tuple(reversed(PAIRS[pair_id]["words"])))
            and type(base_seed) is int and base_seed in SEEDS,
            "random seed derivation input invalid")
    key = f"{SEED_PREFIX}|{pair_id}|{engine}|{recipient_word}|{donor_word}|{site}|{base_seed}"
    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big") & ((1 << 63) - 1)


def expected_matrix() -> set[tuple[str, str, str, str, str, int | None]]:
    matrix = set()
    for pair_id, pair in PAIRS.items():
        for engine in ENGINES:
            a, b = pair["words"]
            for recipient_word, donor_word in ((a, b), (b, a)):
                recipient = case_key(pair_id, engine, recipient_word)
                donor = case_key(pair_id, engine, donor_word)
                for site in SITES:
                    for condition in ("word", "prefix", "postword", "complement", "self"):
                        matrix.add((pair_id, recipient, donor, site, condition, None))
                    for seed in SEEDS:
                        matrix.add((pair_id, recipient, donor, site, "random", seed))
                matrix.add((pair_id, recipient, donor, "post_norm", "full_donor", None))
    require(len(matrix) == 120, "internal arm matrix count")
    return matrix


def trusted_inputs() -> dict[str, dict[str, Any]]:
    require(sha(MANIFEST) == MANIFEST_SHA, "manifest bytes changed")
    require(sha(ASR_SCREEN) == ASR_SCREEN_SHA, "prior ASR screen bytes changed")
    manifest = parse_json(MANIFEST.read_text())
    asr = parse_json(ASR_SCREEN.read_text())
    require(asr.get("corpus_manifest_sha256") == MANIFEST_SHA,
            "prior ASR screen manifest identity differs")
    screened = [row for row in asr["records"] if row.get("pair_id") in PAIRS
                and row.get("phrasing_id") == "fit_chosen"
                and row.get("engine") in ENGINES]
    require(len(screened) == 8
            and {(row["pair_id"], row["engine"], row["word"]) for row in screened}
            == {(pair_id, engine, word) for pair_id, pair in PAIRS.items()
                for engine in ENGINES for word in pair["words"]}
            and all(row.get("normalized_exact") is True for row in screened),
            "prior ASR screen coverage or eligibility differs")
    selected = [p for p in manifest["pairs"] if p.get("pair_id") in PAIRS
                and p.get("phrasing_id") == "fit_chosen"]
    require(len(selected) == 4 and {(p["pair_id"], p["engine"]) for p in selected}
            == {(pair_id, engine) for pair_id in PAIRS for engine in ENGINES},
            "FIT pair coverage differs")
    cases = {}
    for pair in selected:
        pair_id, engine = pair["pair_id"], pair["engine"]
        spec = PAIRS[pair_id][engine]
        pair_words = PAIRS[pair_id]["words"]
        require(pair.get("split") == "fit_pool" and pair.get("map_fit") is True
                and pair.get("words") == list(pair_words), "non-FIT or altered word pair")
        require(pair.get("outside_slot_pcm_identical") is True
                and pair.get("inside_slot_pcm_differs") is True, "manifest PCM assertion absent")
        slot = (pair["word_slot_start_sample"], pair["word_slot_end_sample_exclusive"])
        require(slot == spec["slot"], "word sample slot changed")
        pcm = {}
        for word in pair_words:
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
            key = case_key(pair_id, engine, word)
            cases[key] = {"wav_sha256": record["file_sha256"],
                          "pcm_sha256": record["pcm_int16_sha256"],
                          "frames": record["frames"], "pair_id": pair_id, "engine": engine,
                          "word": word}
        start, end = slot
        a, b = (pcm[word] for word in pair_words)
        require(len(a) == len(b) and a[:2 * start] == b[:2 * start]
                and a[2 * end:] == b[2 * end:]
                and a[2 * start:2 * end] != b[2 * start:2 * end],
                f"actual PCM slot contrast invalid: {pair_id}/{engine}")
        require((len(a) // 2 + 319) // 320 == spec["N"], "active encoder row count differs")
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
    require(receipt.get("schema") == "whisper_fit_spread_causal_preflight_v1"
            and receipt.get("manifest_sha256") == MANIFEST_SHA
            and receipt.get("prior_asr_screen_sha256") == ASR_SCREEN_SHA
            and receipt.get("model_id") == MODEL_ID
            and receipt.get("model_revision") == MODEL_REVISION
            and receipt.get("checkpoint_sha256") == WEIGHTS_SHA
            and receipt.get("device") == "cpu" and receipt.get("threads") == 4,
            "preflight model or execution identity changed")
    seed_rows = receipt.get("random_arm_seeds")
    require(isinstance(seed_rows, list) and len(seed_rows) == 32,
            "preflight random arm seed map absent")
    observed_seed_map = {}
    for item in seed_rows:
        require(isinstance(item, dict) and set(item) ==
                {"pair_id", "engine", "recipient_word", "donor_word", "site",
                 "base_seed", "arm_seed"},
                "preflight random arm seed record invalid")
        key = (item["pair_id"], item["engine"], item["recipient_word"],
               item["donor_word"], item["site"], item["base_seed"])
        require(key not in observed_seed_map,
                "preflight duplicate random arm seed record")
        observed_seed_map[key] = item["arm_seed"]
    expected_seed_map = {
        (pair_id, engine, recipient_word, donor_word, site, base_seed):
        derived_arm_seed(pair_id, engine, recipient_word, donor_word, site, base_seed)
        for pair_id, pair in PAIRS.items()
        for engine in ENGINES
        for recipient_word, donor_word in (tuple(pair["words"]), tuple(reversed(pair["words"])))
        for site in SITES
        for base_seed in SEEDS
    }
    require(observed_seed_map == expected_seed_map
            and len(set(observed_seed_map.values())) == 32,
            "preflight random arm seed map differs from plan")
    require(sha(RUNNER) == receipt.get("script_sha256"), "runner bytes differ from preflight")
    check_runner_commit(receipt)
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
    for pair_id, pair in PAIRS.items():
        for engine in ENGINES:
            spec = pair[engine]
            entry = receipt.get("regions", {}).get(pair_id, {}).get(engine, {})
            require(entry.get("W") == list(spec["W"])
                    and entry.get("P") == list(spec["P"])
                    and entry.get("Q") == list(spec["Q"])
                    and entry.get("N") == spec["N"]
                    and entry.get("C") == region_spans(pair_id, engine, "complement")
                    and entry.get("slot_samples") == list(spec["slot"]),
                    f"preflight region changed: {pair_id}/{engine}")
    return receipt


def check_runner_commit(receipt: dict[str, Any]) -> None:
    commit = receipt.get("script_source_git_commit")
    require(isinstance(commit, str) and re.fullmatch(r"[0-9a-f]{40}", commit) is not None,
            "runner source commit ID invalid")
    relative = RUNNER.relative_to(ROOT).as_posix()
    shown = subprocess.run(["git", "show", f"{commit}:{relative}"], cwd=ROOT,
                           capture_output=True, check=False)
    require(shown.returncode == 0
            and hashlib.sha256(shown.stdout).hexdigest() == receipt.get("script_sha256"),
            "runner source commit does not contain pinned runner bytes")


def checked_result(value: dict[str, Any], pair_id: str,
                   tokenizer: Any | None = None) -> dict[str, Any]:
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
    # Generation must terminate at its sole EOS. An early or duplicate EOS is
    # retained as censored evidence, never as a complete answer.
    eos_terminated = ids.count(EOS_ID) == 1 and ids[-1] == EOS_ID
    require(value.get("eos_seen") is eos_seen
            and value.get("eos_terminated") is eos_terminated
            and value.get("censored") is (not eos_terminated),
            "EOS termination/censor flags differ from IDs")
    scores = value.get("step_scores")
    require(isinstance(scores, list) and 1 <= len(scores) <= 128
            and 1 <= len(ids) - len(scores) <= 4,
            "step-score coverage/length invalid")
    for i, score in enumerate(scores):
        require(score.get("token_id") == ids[len(ids) - len(scores) + i]
                and len(score.get("top2_ids", [])) == 2
                and len(score.get("top2_scores", [])) == 2,
                "step-score token alignment invalid")
        for number in [score.get("selected_score"), *score["top2_scores"]]:
            require(isinstance(number, (int, float)) and math.isfinite(number),
                    "nonfinite step score")
        a, b = PAIRS[pair_id]["words"]
        margin_field = f"generation_processed_{a}_minus_{b}_first_token_score"
        require(margin_field in score
                and [name for name in score if name.startswith("generation_processed_")
                     and name.endswith("_first_token_score")] == [margin_field],
                "alias score field differs from pair")
        margin = score[margin_field]
        status = score.get("alias_margin_status")
        require(status in {"finite", "multi_token_alias", "nonfinite_suppressed_alias"}
                and ((status == "finite" and isinstance(margin, (int, float))
                      and math.isfinite(margin)) or
                     (status != "finite" and margin is None)),
                "alias score/status mismatch")
    require(isinstance(value.get("elapsed_seconds"), (int, float))
            and 0 <= value["elapsed_seconds"] <= MAX_SECONDS
            and type(value.get("rss_bytes")) is int
            and 0 < value["rss_bytes"] <= MAX_RSS,
            "per-decode time or RSS invalid")
    return {"token_ids": ids, "transcript": transcript, "words": words(transcript),
            "eos_seen": eos_seen, "eos_terminated": eos_terminated,
            "censored": not eos_terminated}


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
    result: dict[str, Any] = {}
    for row in trace_rows:
        pair_id, engine = row.get("pair_id"), row.get("engine")
        key = (pair_id, engine)
        require(pair_id in PAIRS and engine in ENGINES and key not in result
                and row.get("words") == list(PAIRS[pair_id]["words"]),
                "trace pair/engine/words duplicated or changed")
        spec = PAIRS[pair_id][engine]
        expected_regions = {"W": list(spec["W"]), "P": list(spec["P"]),
                            "Q": list(spec["Q"]), "N": spec["N"],
                            "C": region_spans(pair_id, engine, "complement"),
                            "slot_samples": list(spec["slot"])}
        require(row.get("regions") == expected_regions, "trace region specification differs")
        a, b = PAIRS[pair_id]["words"]
        left = captures[case_key(pair_id, engine, a)]
        right = captures[case_key(pair_id, engine, b)]
        pre_left, pre_right = left["post_conv_pos"], right["post_conv_pos"]
        complement = region_spans(pair_id, engine, "complement")
        outside_equal = all(np.array_equal(pre_left[:, start:end, :],
                                           pre_right[:, start:end, :])
                            for start, end in complement)
        word_nonzero = not np.array_equal(pre_left[:, slice(*spec["W"]), :],
                                          pre_right[:, slice(*spec["W"]), :])
        require(outside_equal and word_nonzero
                and row.get("outside_preconv_equal") is True
                and row.get("word_preconv_differs") is True
                and row.get("input_locality_pass") is True,
                "preconv input-locality gate differs")
        recorded = row.get("traces")
        require(isinstance(recorded, dict) and set(recorded) == {"mel", *CAPTURES},
                "trace site inventory differs")
        result[key] = {}
        for site, saved in recorded.items():
            require(isinstance(saved, dict), "trace site detail invalid")
            prefix = f"captures/{pair_id}_{engine}_{site}_{a}_{b}"
            l2_path = output / f"{prefix}_l2.npy"
            maxabs_path = output / f"{prefix}_maxabs.npy"
            require(saved.get("path") == f"{prefix}_l2.npy"
                    and saved.get("maxabs_path") == f"{prefix}_maxabs.npy",
                    "trace path changed")
            require(l2_path.is_file() and not l2_path.is_symlink()
                    and maxabs_path.is_file() and not maxabs_path.is_symlink()
                    and sha(l2_path) == saved.get("sha256")
                    and sha(maxabs_path) == saved.get("maxabs_sha256"),
                    "trace sidecar SHA differs")
            left_state, right_state = left[site], right[site]
            axis = 1 if site == "mel" else -1
            diff = left_state - right_state
            expected_l2 = np.linalg.norm(diff, axis=axis)[0]
            expected_maxabs = np.max(np.abs(diff), axis=axis)[0]
            actual_l2 = np.load(l2_path, allow_pickle=False)
            actual_maxabs = np.load(maxabs_path, allow_pickle=False)
            require(actual_l2.dtype == np.float32 and actual_maxabs.dtype == np.float32
                    and actual_l2.shape == expected_l2.shape
                    and actual_maxabs.shape == expected_maxabs.shape
                    and saved.get("shape") == list(expected_l2.shape)
                    and bool(np.isfinite(actual_l2).all())
                    and bool(np.isfinite(actual_maxabs).all())
                    and np.allclose(actual_l2, expected_l2, rtol=1e-5, atol=1e-6)
                    and np.allclose(actual_maxabs, expected_maxabs, rtol=1e-5, atol=1e-6),
                    "trace sidecar differs from captured states")
            scale = 2 if site == "mel" else 1
            region = {name: [value * scale for value in spec[name]]
                      for name in ("W", "P", "Q")}
            c_spans = [[start * scale, end * scale] for start, end in complement]
            active = spec["N"] * scale
            require(saved.get("word_window") == region["W"]
                    and saved.get("prefix_window") == region["P"]
                    and saved.get("postword_window") == region["Q"]
                    and saved.get("complement_spans") == c_spans
                    and saved.get("active_rows") == active,
                    "trace site region changed")
            sums = {
                "word_l2_sum": float(expected_l2[slice(*region["W"])].sum()),
                "prefix_l2_sum": float(expected_l2[slice(*region["P"])].sum()),
                "postword_l2_sum": float(expected_l2[slice(*region["Q"])].sum()),
                "complement_l2_sum": float(sum(
                    expected_l2[start:end].sum() for start, end in c_spans)),
            }
            for field, expected in (*sums.items(),
                                    ("global_max_abs_difference", float(expected_maxabs.max()))):
                observed = saved.get(field)
                require(isinstance(observed, (int, float)) and math.isfinite(observed)
                        and math.isclose(observed, expected, rel_tol=1e-5, abs_tol=1e-5),
                        f"trace {field} differs from captures")
            first = int(np.flatnonzero(expected_l2 > 0)[0]) if np.any(expected_l2 > 0) else None
            require(saved.get("first_nonzero_frame") == first,
                    "trace first-nonzero frame differs")
            result[key][site] = {"first_nonzero_frame": first, **sums}
    require(set(result) == {(pair_id, engine) for pair_id in PAIRS for engine in ENGINES},
            "trace pair/engine coverage incomplete")
    return {f"{pair_id}/{engine}": details for (pair_id, engine), details in result.items()}


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
        "wer_to_recipient": edit_distance(observed, base) / max(1, len(base)),
        "wer_to_single_word_donor_counterfactual": edit_distance(observed, expected)
                                                     / max(1, len(expected)),
        "normalized_words": observed,
    }


def audit_random_patch(row: dict[str, Any], recipient_state: np.ndarray,
                       donor_state: np.ndarray) -> None:
    import torch

    pair_id, engine, site = row["pair_id"], row["engine"], row["site"]
    spans = region_spans(pair_id, engine, "complement")
    recipient = torch.from_numpy(np.concatenate(
        [recipient_state[:, start:end, :] for start, end in spans], axis=1).copy())
    donor = torch.from_numpy(np.concatenate(
        [donor_state[:, start:end, :] for start, end in spans], axis=1).copy())
    target = torch.linalg.vector_norm(donor - recipient)
    require(bool(torch.isfinite(target)) and float(target) > 0,
            "random target norm is zero/nonfinite")
    recipient_word = row["recipient"].removeprefix(f"{pair_id}_{engine}_")
    donor_word = row["donor"].removeprefix(f"{pair_id}_{engine}_")
    base_seed = row.get("base_seed")
    arm_seed = derived_arm_seed(pair_id, engine, recipient_word, donor_word,
                                site, base_seed)
    require(row.get("arm_seed") == arm_seed, "random arm seed differs from plan")
    generator = torch.Generator(device="cpu").manual_seed(arm_seed)
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
    require(len(rows) == 134 and rows[0].get("type") == "preflight"
            and rows[-1].get("type") == "complete", "planned result length/order differs")
    require(all(row.get("type") == "baseline" for row in rows[1:9])
            and all(row.get("type") == "trace" for row in rows[9:13])
            and all(row.get("type") == "intervention" for row in rows[13:-1]),
            "baseline/trace/intervention ordering differs")
    baselines: dict[str, dict[str, Any]] = {}
    captures: dict[str, dict[str, np.ndarray]] = {}
    capture_hashes: dict[str, dict[str, str]] = {}
    for row in rows[1:9]:
        key = row.get("case")
        require(key in cases and key not in baselines, "baseline case duplicate/unknown")
        case = cases[key]
        pair_id, engine, word = case["pair_id"], case["engine"], case["word"]
        require(row.get("pair_id") == pair_id and row.get("engine") == engine,
                "baseline pair/engine identity differs")
        baseline = checked_result(row.get("result"), pair_id, tokenizer)
        alternative = next(w for w in PAIRS[pair_id]["words"] if w != word)
        require(baseline["words"].count(word) == 1
                and alternative not in baseline["words"]
                and baseline["eos_terminated"] and row.get("eligible") is True,
                f"clean baseline ineligible: {key}")
        require(row.get("capture_complete") is True
                and row.get("capture_hook_counts") == dict.fromkeys(CAPTURES, 1)
                and row["result"].get("hook_count") is None,
                "baseline capture incomplete")
        hashes = row.get("capture_sha256")
        require(isinstance(hashes, dict) and set(hashes) == {"mel", *CAPTURES},
                "capture hash inventory invalid")
        captures[key] = {site: load_capture(output / "captures" / f"{key}_{site}.npy",
                                            hashes[site]) for site in CAPTURES}
        captures[key]["mel"] = load_mel(output / "captures" / f"{key}_mel.npy",
                                          hashes["mel"])
        baselines[key], capture_hashes[key] = baseline, hashes
    require(set(baselines) == set(cases), "baseline coverage incomplete")
    trace = audit_traces(rows[9:13], output, captures)

    actual: set[tuple[str, str, str, str, str, int | None]] = set()
    random_arm_seeds: set[int] = set()
    random_edit_hashes: set[str] = set()
    outcomes: list[dict[str, Any]] = []
    planned = expected_matrix()
    for row in rows[13:-1]:
        pair_id, engine = row.get("pair_id"), row.get("engine")
        recipient, donor, site, condition = (row.get(name) for name in
                                              ("recipient", "donor", "site", "condition"))
        base_seed = row.get("base_seed")
        key = (pair_id, recipient, donor, site, condition, base_seed)
        require(key in planned and key not in actual,
                "intervention missing, duplicate, or not prespecified")
        actual.add(key)
        require(engine in ENGINES and recipient in cases and donor in cases
                and cases[recipient]["pair_id"] == pair_id
                and cases[donor]["pair_id"] == pair_id
                and cases[recipient]["engine"] == engine
                and cases[donor]["engine"] == engine
                and recipient != donor,
                "intervention pair/engine/direction invalid")
        rec_word, donor_word = cases[recipient]["word"], cases[donor]["word"]
        require(row.get("region_spans") == region_spans(pair_id, engine, condition),
                "intervention region spans changed")
        require(row.get("hook_count") == 1 and row.get("valid_hook") is True
                and row.get("state_shape") == list(STATE_SHAPE)
                and row.get("state_dtype") == "float32"
                and row.get("finite_checked") is True,
                "intervention hook/state metadata invalid")
        result = checked_result(row.get("result"), pair_id, tokenizer)
        require(row["result"].get("hook_count") == 1, "result hook count differs")
        source = recipient if condition == "self" else donor
        require(row.get("recipient_state_sha256") == capture_hashes[recipient][site]
                and row.get("source_state_sha256") == capture_hashes[source][site],
                "intervention source/recipient capture hash differs")
        baseline = baselines[recipient]
        independent = recompute_outcome(result["transcript"], rec_word, donor_word,
                                        baseline["transcript"])
        independent["target_replacement"] = (independent["target_replacement"]
                                             and result["eos_terminated"])
        independent["token_difference_count"] = (
            sum(left != right for left, right in zip(result["token_ids"],
                                                     baseline["token_ids"]))
            + abs(len(result["token_ids"]) - len(baseline["token_ids"])))
        require(row.get("outcome") == independent,
                "stored outcome differs from raw transcript")
        if condition == "self":
            require(row.get("control_pass") is True
                    and result["token_ids"] == baseline["token_ids"],
                    "self-patch changed complete token sequence")
        elif condition == "full_donor":
            require(row.get("control_pass") is True
                    and result["token_ids"] == baselines[donor]["token_ids"],
                    "full-donor control failed to reproduce donor tokens")
        else:
            require(row.get("control_pass") is None,
                    "noncontrol arm has control pass flag")
        if condition == "random":
            require("seed" not in row and base_seed in SEEDS,
                    "random base seed missing or ambiguous")
            audit_random_patch(row, captures[recipient][site], captures[donor][site])
            arm_seed = row["arm_seed"]
            edit_hash = row["patch_audit"]["random_edit_sha256"]
            require(arm_seed not in random_arm_seeds
                    and edit_hash not in random_edit_hashes,
                    "random arm seed or edit tensor repeated")
            random_arm_seeds.add(arm_seed)
            random_edit_hashes.add(edit_hash)
        elif condition == "full_donor":
            require(row.get("base_seed") is None and row.get("arm_seed") is None
                    and row.get("patch_audit") == {} and "seed" not in row,
                    "full donor has unexpected random metadata")
        else:
            require(row.get("base_seed") is None and row.get("arm_seed") is None
                    and row.get("patch_audit") == {} and "seed" not in row,
                    "nonrandom arm has unexpected random metadata")
        outcomes.append({"pair_id": pair_id, "engine": engine,
                         "recipient": recipient, "donor": donor, "site": site,
                         "condition": condition, "base_seed": base_seed,
                         "arm_seed": row.get("arm_seed"),
                         "target_replacement": independent["target_replacement"],
                         "donor_word_appears": donor_word in result["words"],
                         "recipient_transcript_match": independent["recipient_transcript_match"],
                         "wer_vs_recipient": independent["wer_to_recipient"],
                         "eos_terminated": result["eos_terminated"]})
    require(actual == planned, "intervention matrix incomplete")
    require(len(random_arm_seeds) == 32 and len(random_edit_hashes) == 32,
            "32 independent random edits absent")
    complete = rows[-1]
    require(isinstance(complete.get("elapsed_seconds"), (int, float))
            and 0 <= complete["elapsed_seconds"] <= MAX_SECONDS
            and type(complete.get("rss_bytes")) is int
            and 0 < complete["rss_bytes"] <= MAX_RSS, "campaign cap exceeded")
    counts = Counter((row["pair_id"], row["site"], row["condition"],
                      row["target_replacement"]) for row in outcomes)
    return {"status": "independent_saved_data_audit_pass",
            "records_sha256": sha(output / "records.jsonl"),
            "input_manifest_sha256": MANIFEST_SHA,
            "baseline_count": len(baselines),
            "intervention_count": len(outcomes),
            "random_unique_seed_count": len(random_arm_seeds),
            "random_unique_edit_count": len(random_edit_hashes),
            "outcome_counts": [
                {"pair_id": pair_id, "site": site, "condition": condition,
                 "target_replacement": replacement, "count": count}
                for (pair_id, site, condition, replacement), count in sorted(counts.items())],
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
    require(len(rows) <= 134, "failed attempt exceeds planned row count")
    baseline_rows = {row["case"]: row for row in rows if row.get("type") == "baseline"}
    arm_rows = [row for row in rows if row.get("type") == "intervention"]
    explicit_control_failures = []
    for row in arm_rows:
        if row.get("condition") not in {"self", "full_donor"}:
            continue
        if row.get("control_pass") is not False:
            continue
        pair_id = row.get("pair_id")
        require(pair_id in PAIRS, "failed control pair identity invalid")
        result = checked_result(row.get("result"), pair_id, tokenizer)
        reference = row.get("recipient") if row["condition"] == "self" else row.get("donor")
        require(reference in baseline_rows, "failed control lacks baseline")
        baseline = checked_result(baseline_rows[reference].get("result"), pair_id,
                                  tokenizer)
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
