"""Independent saved-output audit for the Track B Whisper two-site assay.

No Whisper model weights are loaded. Clean mel features and all activation
patches are reconstructed; complete generated ID sequences are re-decoded.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

import wav2vec2_dev_word_disjoint_audit_v1 as dev_audit
import whisper_fit_word_window_causal_v1 as fit_whisper

from . import candidate_common as common
from . import run_whisper as runner


def patch_independent(recipient: np.ndarray, donor: np.ndarray, n: int,
                      word_window: tuple[int, int], arm: str,
                      seed: int | None) -> np.ndarray:
    common.require(recipient.shape == donor.shape == (1, 1500, 384)
                   and recipient.dtype == donor.dtype == np.float32,
                   "Whisper clean state shape invalid")
    a, b = word_window
    common.require(0 < a < b < n <= 1500, "Whisper audit window invalid")
    result = np.array(recipient, copy=True)
    if arm == "self":
        pass
    elif arm == "W":
        result[:, a:b] = donor[:, a:b]
    elif arm == "O":
        result[:, :a] = donor[:, :a]
        result[:, b:n] = donor[:, b:n]
    elif arm == "full":
        result[:, :n] = donor[:, :n]
    elif arm in ("random_W", "random_O"):
        common.require(type(seed) is int, "Whisper random seed invalid")
        if arm == "random_W":
            current, target = recipient[:, a:b], donor[:, a:b]
        else:
            current = np.concatenate((recipient[:, :a], recipient[:, b:n]), axis=1)
            target = np.concatenate((donor[:, :a], donor[:, b:n]), axis=1)
        radius = float(np.linalg.norm(target.astype(np.float64)-current.astype(np.float64)))
        z = np.random.Generator(np.random.PCG64(seed)).standard_normal(current.shape)
        z_norm = float(np.linalg.norm(z))
        common.require(radius > 0 and z_norm > 0 and math.isfinite(radius)
                       and math.isfinite(z_norm), "Whisper random edit invalid")
        modified = current + (z * (radius/z_norm)).astype(np.float32)
        if arm == "random_W":
            result[:, a:b] = modified
        else:
            result[:, :a] = modified[:, :a]
            result[:, b:n] = modified[:, a:]
    else:
        raise ValueError("Whisper arm invalid")
    common.require(np.array_equal(result[:, n:], recipient[:, n:]),
                   "Whisper patch changed padded rows")
    return result


def audit(output: Path, preflight_path: Path, freeze_path: Path) -> dict[str, Any]:
    from transformers import WhisperProcessor
    from huggingface_hub import snapshot_download
    current = runner.preflight(human_qa=True)
    common.require(fit_whisper.canonical(json.loads(preflight_path.read_text())) ==
                   fit_whisper.canonical(current), "Whisper preflight drift")
    freeze_sha = runner.verify_freeze(preflight_path, freeze_path, current)
    records = output / "records.jsonl"
    common.require(records.is_file() and not records.is_symlink(), "Whisper records absent")
    raw = records.read_bytes()
    common.require(raw.endswith(b"\n"), "truncated Whisper records")
    rows = [dev_audit.parse(line) for line in raw.decode("utf-8").splitlines()]
    common.require(len(rows) == 418 and rows[0].get("type") == "preflight"
                   and rows[-1].get("type") == "complete", "Whisper matrix incomplete")
    common.require(rows[0].get("receipt") == current
                   and rows[0].get("preflight_sha256") == common.sha(preflight_path)
                   and rows[0].get("freeze_sha256") == freeze_sha
                   and rows[-1].get("clean_rows") == 32
                   and rows[-1].get("patch_rows") == 384
                   and not any(r.get("type") == "failure" for r in rows),
                   "Whisper run provenance incomplete")
    cases, _ = common.fixture_cases()
    case_by_key = {c.key: c for c in cases}
    clean_rows = rows[1:33]
    common.require([x.get("case_key") for x in clean_rows] == [c.key for c in cases],
                   "Whisper clean row coverage/order differs")
    processor = WhisperProcessor.from_pretrained(
        fit_whisper.MODEL_ID, revision=fit_whisper.MODEL_REVISION, local_files_only=True)
    snapshot = Path(snapshot_download(fit_whisper.MODEL_ID,
                    revision=fit_whisper.MODEL_REVISION, local_files_only=True))
    generation = json.loads((snapshot / "generation_config.json").read_text())
    eos = generation["eos_token_id"]
    eos_set = set(eos if isinstance(eos, list) else [eos])
    expected_files = {"records.jsonl"}
    clean: dict[str, tuple[dict, dict[str, np.ndarray]]] = {}
    for row in clean_rows:
        case = case_by_key[row["case_key"]]
        common.require(row.get("type") == "clean" and row.get("schema") == runner.SCHEMA
                       and row.get("pair_id") == case.pair_id
                       and row.get("engine") == case.engine
                       and row.get("word") == case.word
                       and row.get("hook_counts") == {site: 1 for site in common.SITES_WHISPER},
                       "Whisper clean identity/hook differs")
        shapes = {"mel": (1, 80, 3000),
                  "post_block0": (1, 1500, 384),
                  "post_norm": (1, 1500, 384)}
        arrays = {}
        for name, shape in shapes.items():
            rel = f"captures/{case.key}_{name}.npy"
            common.require(row["paths"].get(name) == rel
                           and row["shape"].get(name) == list(shape),
                           "Whisper clean sidecar receipt differs")
            arrays[name] = dev_audit.array(output, rel, row["sha256"][name], shape)
            expected_files.add(rel)
        samples = common.load_pcm(case).astype(np.float32) / np.float32(32768)
        batch = processor(samples, sampling_rate=16000,
                          return_tensors="pt", return_attention_mask=True)
        common.require(np.array_equal(arrays["mel"], batch["input_features"].numpy()),
                       "Whisper clean mel differs from frozen waveform")
        mask = batch["attention_mask"].detach().cpu().contiguous().numpy()
        common.require(row.get("attention_mask_shape") == list(mask.shape)
                       and row.get("attention_mask_sha256") == dev_audit.tensor_sha(mask),
                       "Whisper attention mask differs")
        ids = row.get("ids")
        common.require(isinstance(ids, list) and ids and all(type(x) is int for x in ids)
                       and len(ids) <= runner.MAX_NEW_TOKENS + 8,
                       "Whisper clean token IDs invalid")
        transcript = processor.batch_decode([ids], skip_special_tokens=True)[0].strip()
        words = dev_audit.token_list(transcript)
        censored = ids[-1] not in eos_set
        ref = common.reference_tokens(case.word)
        common.require(row.get("transcript") == transcript
                       and row.get("tokens") == words
                       and row.get("censored") is censored
                       and row.get("reference_tokens") == ref
                       and row.get("eligible") is (not censored and words == ref),
                       "Whisper clean full-output eligibility differs")
        clean[case.key] = (row, arrays)
    patch_rows = rows[33:-1]
    expected = [(d["index"], site, arm) for d in current["directions"]
                for site in common.SITES_WHISPER for arm in common.ARMS]
    common.require([(x.get("direction"), x.get("site"), x.get("arm"))
                    for x in patch_rows] == expected, "Whisper patch coverage/order differs")
    counts: Counter[str] = Counter()
    control_failures = []
    for row in patch_rows:
        d = current["directions"][row["direction"]]
        site, arm = row["site"], row["arm"]
        rec, ra = clean[d["recipient_key"]]
        donor, da = clean[d["donor_key"]]
        case = case_by_key[d["recipient_key"]]
        n = math.ceil(case.frames/320)
        s, e = d["slot"]
        w = (max(0, math.floor(s/320)-5), min(n, math.ceil(e/320)+5))
        common.require(0 < w[0] < w[1] < n
                       and row.get("window") == list(w)
                       and row.get("active_frames") == n,
                       "Whisper patch geometry differs")
        seed = (runner.SEED_BASE + 4*d["index"] +
                2*common.SITES_WHISPER.index(site) + (0 if arm == "random_W" else 1)
                if arm in ("random_W", "random_O") else None)
        common.require(row.get("type") == "patch"
                       and row.get("schema") == runner.SCHEMA
                       and row.get("recipient_key") == d["recipient_key"]
                       and row.get("donor_key") == d["donor_key"]
                       and row.get("seed") == seed
                       and row.get("hook_counts") ==
                       {s: int(s == site) for s in common.SITES_WHISPER},
                       "Whisper patch identity/hook/seed differs")
        patch = patch_independent(ra[site], da[site], n, w, arm, seed)
        common.require(row.get("patch_tensor_sha256") == dev_audit.tensor_sha(patch),
                       "Whisper patch hash differs")
        a, b = w
        rw, dw = ra[site][:, a:b], da[site][:, a:b]
        ro = np.concatenate((ra[site][:, :a], ra[site][:, b:n]), axis=1)
        do = np.concatenate((da[site][:, :a], da[site][:, b:n]), axis=1)
        norms = {"natural_w_l2": float(np.linalg.norm(dw.astype(np.float64)-rw.astype(np.float64))),
                 "natural_o_l2": float(np.linalg.norm(do.astype(np.float64)-ro.astype(np.float64))),
                 "realized_l2": float(np.linalg.norm(patch.astype(np.float64)-ra[site].astype(np.float64)))}
        common.require(all(math.isclose(row["norms"].get(k), value,
                                        rel_tol=1e-10, abs_tol=1e-10)
                           for k, value in norms.items()), "Whisper edit norm differs")
        ids = row.get("ids")
        common.require(isinstance(ids, list) and ids and all(type(x) is int for x in ids)
                       and len(ids) <= runner.MAX_NEW_TOKENS + 8,
                       "Whisper patched token IDs invalid")
        transcript = processor.batch_decode([ids], skip_special_tokens=True)[0].strip()
        words = dev_audit.token_list(transcript)
        censored = ids[-1] not in eos_set
        strict = rec["eligible"] and donor["eligible"]
        control = (ids == rec["ids"] if arm == "self" else
                   ids == donor["ids"] if arm == "full" else None)
        common.require(row.get("transcript") == transcript
                       and row.get("tokens") == words
                       and row.get("censored") is censored
                       and row.get("strict_eligible") is strict
                       and row.get("matches_donor_transcript") is
                       (transcript == donor["transcript"])
                       and row.get("matches_recipient_transcript") is
                       (transcript == rec["transcript"])
                       and row.get("intended_word_only") is
                       (strict and not censored and words == donor["reference_tokens"])
                       and row.get("control_pass") == control,
                       "Whisper patched full-output/control result differs")
        if control is False:
            control_failures.append((d["index"], site, arm))
        counts[f"{site}/{arm}"] += 1
    common.require(all(counts[f"{site}/{arm}"] == 32
                       for site in common.SITES_WHISPER for arm in common.ARMS),
                   "Whisper patch arm population differs")
    actual_files = {p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file()}
    allowed = expected_files | ({"audit.json"} if (output / "audit.json").exists() else set())
    common.require(actual_files == allowed, "Whisper output files extra/missing")
    bytes_used = sum((output / name).stat().st_size for name in actual_files)
    common.require(bytes_used <= runner.MAX_OUTPUT_BYTES,
                   "Whisper output cap exceeded")
    return {"schema": runner.SCHEMA + "_audit",
            "status": "independent_saved_data_audit_pass",
            "records_sha256": common.sha(records),
            "clean_rows": 32, "patch_rows": 384,
            "arm_counts": dict(counts), "control_failures": control_failures,
            "mechanism_gate_pass": not control_failures,
            "output_bytes": bytes_used,
            "byte_ledger_bound": current["byte_ledger"]["total_bound"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.output, args.preflight, args.freeze)
    target = args.output / "audit.json"
    common.require(not target.exists(), "refusing to overwrite Whisper audit")
    target.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
