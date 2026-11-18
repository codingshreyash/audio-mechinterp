"""Read-only CPU audit of a completed Voxtral v2 FIT first-record benchmark.

Never imports or loads model weights, never runs candidate inference, and never
opens development or locked-test WAVs. Audit only after the run has finished.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXPECTED = {
    "runner": "64d799bbb40b58b128b049877d18bde7774c2dc16717af97e692aa4531e907b5",
    "design": "cd2cbd762af3568f85d43c2dec47824666a95d689a3dc89d9cbdf6471ef54ab0",
    "masks": "a634fc0ab8454b7a82802c56a5365dc90e79f446e4d280cee6321adfead4108e",
    "selection": "f897d360dbcec9c09f1813f9c070294753fbabc25c6213ba6bdac34b1d79b56b",
    "index": "bc739cf1f0d265426cc760ddd6a42223554b1097cab6ba87c2c8b81129d1b971",
    "pilot": "6cc84dc1a4ced9ce17380eab3836079c462c9974f31337f146416730b5536ccf",
    "gate": "79622d883e954a5076afac2e97b3809833e7fff5ea89b4ae5ccbd137b819429f",
}
FILES = {
    "runner": HERE / "voxtral_fit_v2_causal_first_record.py",
    "design": HERE / "VOXTRAL_FIT_V2_CAUSAL_DESIGN.md",
    "masks": HERE / "VOXTRAL_FIT_V2_CAUSAL_MASKS.json",
    "selection": HERE / "VOXTRAL_FIT_V2_SELECTION.json",
    "index": HERE / "voxtral_corpus_v1_index.json",
    "pilot": HERE / "runs/voxtral_corpus_v1_fit_pilot_v2/20261118T084011020385Z/result.json",
    "gate": HERE / "runs/voxtral_corpus_v1_fit_pilot_v2/20261118T084011020385Z/v2_gate.json",
}
SITES = ("encoder31", "projector")
ARMS = ("restore", "loss")
PATCHES = ("whole", "word_0", "word_1", "word_2", "word_3", "pre", "post", "padding", "self", "random")
MEMORY_CAP = 30 * 1024**3
TIME_CAP = 45 * 60


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def close(a: float, b: float, *, tol: float = 1e-5) -> bool:
    return math.isfinite(a) and math.isfinite(b) and abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def classify(ids: list[int], visible: str, special: set[int], recipient: str, opposite: str) -> str:
    if any(t in special and not (i == len(ids) - 1 and t == 2) for i, t in enumerate(ids)):
        return "malformed_special"
    if not ids or ids[-1] != 2:
        return "missing_eos"
    stripped = visible.strip()
    if not stripped or not stripped.isascii() or not stripped.isalpha():
        return "malformed_surface"
    answer = stripped.casefold()
    return "recipient" if answer == recipient.casefold() else "opposite" if answer == opposite.casefold() else "other"


def condition_key(row: dict) -> tuple:
    typ = row["type"]
    if typ == "capture":
        return ("capture", row["site"], row["word"], row["kind"])
    if typ == "invalid_condition":
        return ("patch", row["site"], row["recipient"], row["arm"], row["kind"])
    require(typ == "generation", f"unknown progress type {typ}")
    kind = row["kind"]
    if kind.startswith("patch:"):
        _, arm, name = kind.split(":", 2)
        return ("patch", row["site"], row["recipient"], arm, name)
    return ("generation", row["site"], row["recipient"], kind)


def expected_keys(words: list[str]) -> set[tuple]:
    a, b = words
    keys = set()
    for site in SITES:
        for word in words:
            for kind in ("clean", "blank"):
                keys.add(("capture", site, word, kind))
            for kind in ("passive:clean", "passive:blank", "opposite_word_control"):
                keys.add(("generation", site, word, kind))
            for arm in ARMS:
                for name in PATCHES:
                    keys.add(("patch", site, word, arm, name))
    for word in words:
        keys.add(("generation", None, word, "blank_baseline"))
    for name, recipient in ((f"native_{a}", a), (f"native_{b}", b),
                            (f"word_{a}_outside_{b}", a), (f"{a}_with_{b}_padding", a),
                            (f"word_{b}_outside_{a}", b), (f"{b}_with_{a}_padding", b)):
        keys.add(("generation", None, recipient, f"feature:{name}"))
    require(len(keys) == 108, "internal planned-key count wrong")
    return keys


def audit(run_dir: Path) -> dict:
    import torch
    from transformers import AutoProcessor
    import voxtral_fit_pilot_capability_v1 as base

    require(run_dir.is_dir(), "run directory missing")
    for name, path in FILES.items():
        require(sha(path) == EXPECTED[name], f"frozen {name} SHA mismatch")
    meta = json.loads((run_dir / "meta.json").read_text())
    masks = json.loads(FILES["masks"].read_text())
    pilot = json.loads(FILES["pilot"].read_text())
    selection = json.loads(FILES["selection"].read_text())
    first = masks["records"][0]
    words = list(first["words"])
    require((first["pair_id"], first["engine"], first["phrasing_id"], words) ==
            ("animal_08", "piper", "fit_chosen", ["beaver", "monkey"]), "wrong first FIT record")
    require(meta["script_sha256"] == EXPECTED["runner"], "run used different runner")
    for name in ("design", "masks", "selection", "index", "pilot", "gate"):
        key = {"design": "design_sha256", "masks": "masks_sha256", "selection": "selection_sha256",
               "index": "index_sha256", "pilot": "pilot_sha256", "gate": "v2_gate_sha256"}[name]
        require(meta[key] == EXPECTED[name], f"run {name} binding mismatch")
    require(meta["record"] == {k: first[k] for k in ("pair_id", "engine", "phrasing_id", "words")},
            "run population changed")
    require(meta["planned"] == {"generations": 100, "captures": 8, "teacher_forced_forwards": 200},
            "planned workload changed")
    require(meta["model_weights_loaded_during_preflight"] is False, "preflight status changed")
    require(sha(Path(base.__file__)) == "1d4d65a528d558e5bdda3b733eaab6e7e849ab9456d6842d5ee178d3efdfb8d8",
            "reused scoring/provenance code changed")
    require(meta["snapshot"] == base.validate_snapshot(json.loads(FILES["index"].read_text())),
            "saved checkpoint and source-file hashes mismatch")
    require(meta["installed_sources"] == base.validate_transformers_source(),
            "installed Transformers source hashes mismatch")
    require(meta["prompt"] == base.PROMPT and meta["generation"] ==
            {"do_sample": False, "max_new_tokens": 32, "eos": 2}, "prompt/generation policy changed")
    selected = {(r["pair_id"], r["engine"], r["phrasing_id"], r["word"]): r for r in selection["clips"]}
    for clip in first["clips"]:
        key = (first["pair_id"], first["engine"], first["phrasing_id"], clip["word"])
        require(clip["wav_sha256"] == selected[key]["wav_sha256"], "selection clip mismatch")
        require(sha(ROOT / clip["wav_path"]) == clip["wav_sha256"], "FIT WAV bytes changed")

    progress_path = run_dir / "progress.jsonl"
    progress = [json.loads(line) for line in progress_path.read_text().splitlines() if line.strip()]
    keys = [condition_key(r) for r in progress]
    require(len(keys) == len(set(keys)), "duplicate progress condition")
    planned = expected_keys(words)
    require(set(keys).issubset(planned), "unplanned progress condition")
    captures = [r for r in progress if r["type"] == "capture"]
    generated = [r for r in progress if r["type"] == "generation"]
    invalid = [r for r in progress if r["type"] == "invalid_condition"]
    for row in captures:
        site = row["site"]
        expected_raw = [1, 1500, 1280] if site == "encoder31" else [375, 3072]
        expected_stored = expected_raw[1:] if site == "encoder31" else expected_raw
        require(row["hook_calls"] == 1 and row["raw_hook_shape"] == expected_raw and
                row["stored_state_shape"] == expected_stored, "capture hook shape/count mismatch")
        require(row["raw_hook_dtype"] == "torch.bfloat16" and row["stored_state_dtype"] == "torch.bfloat16",
                "capture dtype mismatch")
        require(len(row["sha256"]) == 64, "capture state SHA missing")

    # Reconstruct only the two selected FIT processor inputs; no candidate-model inference.
    processor = AutoProcessor.from_pretrained(str(base.SNAPSHOT), local_files_only=True)
    require(processor.tokenizer.eos_token_id == 2 and int(processor.audio_token_id) == 24,
            "processor token IDs changed")
    features = {}
    for clip in first["clips"]:
        conversation = [{"role": "user", "content": [
            {"type": "audio", "path": str((ROOT / clip["wav_path"]).resolve())},
            {"type": "text", "text": base.PROMPT},
        ]}]
        batch = processor.apply_chat_template(conversation, tokenize=True, return_dict=True, return_tensors="pt")
        features[clip["word"]] = batch["input_features"]
    def tensor_sha(x):
        return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
    for word, native in features.items():
        require(tensor_sha(native) == meta["feature_hashes"][word]["native"], "native feature SHA mismatch")
        blank = native.clone()
        blank[:, :, first["word_mel_frames_direct_overlap"]] = native[:, :, 2999:3000]
        require(tensor_sha(blank) == meta["feature_hashes"][word]["blank"], "blank feature SHA mismatch")
    a, b = words
    fa, fb = features[a], features[b]
    is_word = torch.zeros(3000, dtype=torch.bool)
    is_word[first["word_mel_frames_direct_overlap"]] = True
    first_padding = min(i for i in range(3000) if 160 * i - 200 >= first["clips"][0]["frames"])
    expected_features = {f"native_{a}": fa, f"native_{b}": fb}
    for donor, outside, fd, fo in ((a, b, fa, fb), (b, a, fb, fa)):
        hybrid = fo.clone(); hybrid[:, :, is_word] = fd[:, :, is_word]
        padded = fd.clone(); padded[:, :, first_padding:] = fo[:, :, first_padding:]
        expected_features[f"word_{donor}_outside_{outside}"] = hybrid
        expected_features[f"{donor}_with_{outside}_padding"] = padded
    require({k: tensor_sha(v) for k, v in expected_features.items()} == meta["feature_matrix_hashes"],
            "six-condition feature matrix SHA mismatch")

    special = set(processor.tokenizer.all_special_ids)
    bykey = {condition_key(r): r for r in progress}
    pilot_ids = {r["expected_word"]: r["generated_token_ids"] for r in pilot["records"]
                 if (r["pair_id"], r["engine"], r["phrasing_id"]) ==
                    (first["pair_id"], first["engine"], first["phrasing_id"])}
    for row in generated:
        ids = row["generated_token_ids"]
        visible = processor.batch_decode([ids], skip_special_tokens=True)[0]
        all_tokens = processor.batch_decode([ids], skip_special_tokens=False)[0]
        require(row["raw_decoded"] == visible and row["decoded_with_special_tokens"] == all_tokens,
                "saved decoding mismatch")
        require(row["classification"] == classify(ids, visible, special, row["recipient"], row["opposite"]),
                "classification mismatch")
        require(row["terminal_eos"] == bool(ids and ids[-1] == 2), "terminal EOS field mismatch")
        score = row["canonical_scores"]
        require(set(score) == set(words) and all(math.isfinite(float(x)) for x in score.values()),
                "candidate score keys/nonfinite value")
        require(close(row["recipient_minus_opposite"], score[row["recipient"]] - score[row["opposite"]]),
                "candidate score contrast arithmetic mismatch")
        if row["site"] is None:
            require(row["hook_calls_including_two_scores"] == 0 and not row["hook_events"],
                    "unhooked condition has hook events")
        else:
            shape = [1, 1500, 1280] if row["site"] == "encoder31" else [375, 3072]
            require(row["hook_calls_including_two_scores"] == 3 and len(row["hook_events"]) == 3 and
                    all(h == {"shape": shape, "dtype": "torch.bfloat16"} for h in row["hook_events"]),
                    "hook events mismatch")
        require(row["mps_driver_bytes_after"] < MEMORY_CAP, "per-condition MPS cap reached")
        require(row["seconds_including_scores"] >= 0, "negative condition time")
        kind = row["kind"]
        recipient, opposite, site = row["recipient"], row["opposite"], row["site"]
        base_kind = "blank" if kind.startswith("patch:restore:") else "clean"
        if kind == "blank_baseline":
            continue
        if kind == f"feature:native_{recipient}":
            require(ids == pilot_ids[recipient], "native feature replay mismatch")
        if kind.startswith("passive:"):
            target = pilot_ids[recipient] if kind == "passive:clean" else bykey[("generation", None, recipient, "blank_baseline")]["generated_token_ids"]
            require(ids == target, "passive baseline parity mismatch")
        if kind.endswith(":self"):
            target = pilot_ids[recipient] if base_kind == "clean" else bykey[("generation", None, recipient, "blank_baseline")]["generated_token_ids"]
            require(ids == target, "self-patch baseline parity mismatch")
        if kind.startswith("patch:"):
            _, arm, name = kind.split(":", 2)
            edit = row["edit"]
            smap = first["site_maps"][site]
            expected_rows = (smap["whole_word_rows"] if name in ("whole", "self", "random") else
                             smap["padding_control_rows"] if name == "padding" else smap["bins"][name])
            require(edit["mode"] == name and edit["rows"] == expected_rows and edit["valid"],
                    "patch mode/rows/validity mismatch")
            if name in ("pre", "post", "padding", "random"):
                target = float(edit["target_bf16_norm"])
                measured = float(edit["effective_bf16_norm"])
                require(target > 0 and abs(measured - target) <= max(.05 * target, 1e-6) + 1e-5,
                        "BF16 control norm outside frozen tolerance")
                require(close(float(edit["alpha"]), target / float(edit["unscaled_edit_norm"])),
                        "control alpha arithmetic mismatch")
            if name == "random":
                label = f"voxtral-v2-causal-v1|{first['pair_id']}|{first['engine']}|fit_chosen|{recipient}|{site}|{arm}|random_0"
                seed = int.from_bytes(hashlib.sha256(label.encode()).digest(), "big") & ((1 << 63) - 1)
                require(edit["random_seed_label"] == label and edit["random_seed"] == seed,
                        "random seed derivation mismatch")
            if name == "self":
                require(edit["alpha"] == 0 and edit["effective_bf16_norm"] == 0,
                        "self patch was not zero edit")
            require(len(edit["edited_bf16_sha256"]) == 64, "edited rows SHA missing")
        if kind == "opposite_word_control":
            clean_donor = bykey[("capture", site, opposite, "clean")]
            require(row["edit"]["donor_sha256"] == clean_donor["sha256"], "opposite donor hash mismatch")

    result_path = run_dir / "result.json"
    error_path = run_dir / "error.json"
    require(result_path.exists() != error_path.exists(), "expected exactly one terminal result or error")
    if result_path.exists():
        result = json.loads(result_path.read_text())
        require(result["status"] == "first_record_complete", "unexpected result status")
        require(result["meta_sha256"] == sha(run_dir / "meta.json") and
                result["progress_sha256"] == sha(progress_path), "result content hash mismatch")
        require(result["progress_records"] == len(progress) == 108 and set(keys) == planned,
                "completed planned condition set mismatch")
        require(result["capture_records"] == len(captures) == 8 and
                result["generation_records"] == len(generated) and
                result["invalid_condition_records"] == len(invalid) and
                result["teacher_forced_forward_records"] == 2 * len(generated),
                "reported workload counts mismatch")
        resource = result["resource_projection"]
        full = len(captures) == 8 and len(generated) == 100 and not invalid
        require(resource["full_workload_observed"] == full, "resource full-workload gate mismatch")
        if full:
            measured = float(resource["one_time_fixed_setup_seconds"]) + 8 * float(resource["first_record_variable_seconds"])
            require(close(resource["projected_eight_record_seconds"], measured), "time projection arithmetic mismatch")
            require(resource["passes"] == (measured <= TIME_CAP and resource["peak_mps_driver_bytes"] < MEMORY_CAP),
                    "time/memory gate arithmetic mismatch")
        else:
            require(resource["projected_eight_record_seconds"] is None and not resource["passes"],
                    "incomplete workload must fail closed")
        status = "complete_full_workload" if full else "complete_with_invalid_conditions_projection_blocked"
    else:
        error = json.loads(error_path.read_text())
        require(error["status"] == "failed" and error["progress_sha256"] == sha(progress_path),
                "failed run hash/status mismatch")
        require(error["progress_records"] == len(progress), "failed run progress count mismatch")
        status = "failed_partial_evidence"
    outcomes = {}
    if result_path.exists() and set(keys) == planned:
        for site in SITES:
            outcomes[site] = {}
            for arm in ARMS:
                rows = [bykey[("patch", site, word, arm, "whole")] for word in words]
                eligible = [bykey[("generation", None, word, "blank_baseline")]["classification"] != "recipient"
                            for word in words]
                success = [int((arm == "restore" and eligible[i] and rows[i]["classification"] == "recipient") or
                               (arm == "loss" and rows[i]["classification"] in ("opposite", "other")))
                           for i in range(2)]
                outcomes[site][arm] = {"planned_directions": 2, "eligible_blank_directions": sum(eligible),
                                       "whole_word_complete_answer_successes": sum(success),
                                       "by_recipient": {words[i]: {"eligible_blank": eligible[i],
                                                                    "whole_word_classification": rows[i]["classification"],
                                                                    "whole_word_success": bool(success[i])}
                                                        for i in range(2)}}
    return {"status": status, "run_dir": str(run_dir), "progress_sha256": sha(progress_path),
            "captures": len(captures), "generations": len(generated), "invalid_conditions": len(invalid),
            "planned_slots_present": len(keys), "planned_slots_total": 108,
            "teacher_forced_forwards_recorded": 2 * len(generated),
            "whole_word_outcomes": outcomes,
            "scope": "one FIT pair and two word directions; no 16-direction causal threshold"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.run_dir), indent=2))


if __name__ == "__main__":
    main()
