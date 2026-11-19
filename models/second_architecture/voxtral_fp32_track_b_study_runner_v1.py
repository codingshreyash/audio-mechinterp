"""Source-only 16-context Voxtral Track B orchestration; no live admission yet.

The CLI has only --validate-only.  A later reviewed pilot receipt, independent
GO, and external freeze must be wired into a separate live entry point before
any checkpoint construction.  The injectable functions below are exercised by
fake CPU models; this module never loads weights by itself.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch

import voxtral_fp32_track_b_preflight_v1 as preflight
import voxtral_track_b_answer_link_core_v1 as core

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUNS = HERE / "runs/voxtral_fp32_track_b_answer_link_v1"
SOURCE_PINS = {
    HERE / "voxtral_fp32_track_b_preflight_v1.py":
        "b4365b7f37d9d624bde17d0d3f46b5baadea247ab1451fa2e096a833ba74cb67",
    HERE / "voxtral_track_b_answer_link_core_v1.py":
        "8551fd6176cf31a03920e562e838d9c3cfe20d6fd8af01ac27b995b931b04be6",
    preflight.V3: preflight.SOURCE_HASHES[preflight.V3],
    preflight.V4: preflight.SOURCE_HASHES[preflight.V4],
    preflight.V5: preflight.SOURCE_HASHES[preflight.V5],
    preflight.V6: preflight.SOURCE_HASHES[preflight.V6],
    preflight.V6_DESIGN_GO: preflight.SOURCE_HASHES[preflight.V6_DESIGN_GO],
}
GENERATIONS = ("clean", "zero", "plus_0_1", "minus_0_1", "random_plus_0_1",
               "plus_1_0", "full_donor")
SCORED = ("clean", "zero", "plus_0_1", "minus_0_1", "random_plus_0_1", "plus_1_0")
CONTEXT_COUNT = 16
GENERATIONS_PER_CONTEXT = 7
MAX_NO_GRAD_SCORES_PER_CONTEXT = 72
MAX_GRAD_SCORES_PER_CONTEXT = 12
CONTEXT_SECONDS = 240
CAMPAIGN_SECONDS = 3840
MPS_BYTES = 30 * 1024**3
OUTPUT_BYTES = 2 * 1024**3
FREE_DISK_BYTES = 12 * 1024**3


class CallLedger:
    """Append-only in-memory call schedule; a live supervisor must persist it.

    This guard checks bounds between calls. It cannot interrupt one hung forward,
    so it is not by itself a live 240-second or 30-GiB admission mechanism.
    """

    def __init__(self, *, context_id: str, now: Callable[[], float] = time.monotonic,
                 memory_bytes: Callable[[], int] | None = None):
        self.context_id = context_id
        self.now = now
        self.memory_bytes = memory_bytes
        self.started = now()
        self.active: tuple[str, float] | None = None
        self.calls: list[dict] = []

    def guard(self, event: str) -> None:
        phase, sep, label = event.partition("|")
        require(sep and phase in ("before", "after") and label,
                "Call event must be labeled before|... or after|...")
        at = self.now()
        require(at - self.started <= CONTEXT_SECONDS,
                "Context model-work cap exceeded between calls")
        mps = self.memory_bytes() if self.memory_bytes is not None else None
        require(mps is None or 0 <= mps <= MPS_BYTES,
                "MPS driver allocation cap exceeded between calls")
        if phase == "before":
            require(self.active is None and label not in {r["label"] for r in self.calls},
                    "Duplicate or overlapping model call")
            self.active = (label, at)
            return
        require(self.active is not None and self.active[0] == label,
                "Model call ended without matching begin")
        duration = at - self.active[1]
        require(math.isfinite(duration) and duration >= 0, "Invalid call duration")
        self.calls.append({"ordinal": len(self.calls), "context_id": self.context_id,
                           "label": label, "elapsed_seconds": duration,
                           "mps_driver_bytes_after": mps})
        self.active = None

    def seal(self, *, expected_generation: int, expected_no_grad: int,
             expected_grad: int, expected_labels: Sequence[str] | None = None) -> dict:
        require(self.active is None, "Unfinished model call")
        classes = {"generation": 0, "no_grad": 0, "grad": 0}
        for call in self.calls:
            kind = call["label"].split("|", 1)[0]
            require(kind in classes, "Unknown call class")
            classes[kind] += 1
        require(classes == {"generation": expected_generation,
                            "no_grad": expected_no_grad, "grad": expected_grad},
                "Executed call classes differ from frozen context ledger")
        if expected_labels is not None:
            require([call["label"] for call in self.calls] == list(expected_labels),
                    "Executed call order/identity differs from frozen context ledger")
        return {"schema": "voxtral_track_b_call_ledger_v1", "context_id": self.context_id,
                "calls": list(self.calls), "counts": classes,
                "calls_sha256": json_sha(self.calls)}


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def sha(path: Path) -> str:
    return preflight.file_sha(path)


def json_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def verify_sources() -> dict[str, str]:
    found = {str(path.relative_to(ROOT)): sha(path) for path in SOURCE_PINS}
    require(all(found[str(path.relative_to(ROOT))] == expected
                for path, expected in SOURCE_PINS.items()), "Frozen Track B source changed")
    require(core.PROMPT == preflight.PROMPT and core.EOS == preflight.EOS and
            core.MAX_NEW_TOKENS == 32, "Core/preflight prompt or EOS differs")
    return found


def plan_from_preflight(receipt: dict) -> list[dict]:
    """Enumerate all fixed directions without looking at model answers."""
    require(receipt.get("schema") == "voxtral_fp32_track_b_no_weight_preflight_v1" and
            receipt.get("candidate_weights_loaded") is False and
            receipt.get("candidate_forward_count") == 0, "No-weight preflight receipt invalid")
    wavs = receipt.get("selected_wavs")
    proc = receipt.get("processor", {})
    batches = proc.get("batches")
    aliases = proc.get("aliases")
    require(isinstance(wavs, list) and len(wavs) == 16 and
            isinstance(batches, list) and len(batches) == 16 and
            isinstance(aliases, dict) and set(aliases) ==
            {w for pair in preflight.PAIR_ORDER for w in pair} and
            proc.get("alias_table_sha256") == preflight.ALIAS_TABLE_SHA,
            "Frozen selected WAV/batch/alias population invalid")
    wav_index = {(row["pair_id"], row["engine"], row["word"]): row for row in wavs}
    batch_index = {(row["pair_id"], row["engine"], row["word"]): row for row in batches}
    require(len(wav_index) == len(batch_index) == 16 and set(wav_index) == set(batch_index),
            "Duplicate or missing selected WAV/batch row")
    plan = []
    for pair_no, pair in enumerate(preflight.PAIR_ORDER):
        pair_id = f"pair_{pair_no:02d}"
        for engine in preflight.ENGINES:
            for recipient, donor in (pair, tuple(reversed(pair))):
                key = (pair_id, engine, recipient)
                donor_key = (pair_id, engine, donor)
                w, b, dw, db = wav_index[key], batch_index[key], wav_index[donor_key], batch_index[donor_key]
                require(w["wav_sha256"] == preflight.EXPECTED_WAV_SHAS[key] and
                        dw["wav_sha256"] == preflight.EXPECTED_WAV_SHAS[donor_key] and
                        w["frames"] == dw["frames"] and w["word_slot"] == dw["word_slot"] and
                        b["input_ids"] == db["input_ids"] and
                        b["attention_mask_sha256"] == db["attention_mask_sha256"] and
                        b["audio_placeholder_count"] == db["audio_placeholder_count"] and
                        b["projector_word_rows"] == db["projector_word_rows"],
                        "Matched donor/recipient geometry or prompt differs")
                rows = tuple(b["projector_word_rows"])
                require(rows and rows == tuple(sorted(set(rows))) and
                        all(0 <= r < b["audio_placeholder_count"] and
                            160 * (8 * r + 3) < w["frames"] for r in rows),
                        "Frozen projector mask invalid or padded")
                context_id = f"{pair_id}|{engine}|{recipient}->{donor}"
                plan.append({"index": len(plan), "context_id": context_id,
                             "pair_id": pair_id, "engine": engine,
                             "recipient": recipient, "donor": donor,
                             "recipient_key": list(key), "donor_key": list(donor_key),
                             "wav_sha256": w["wav_sha256"],
                             "input_features_sha256": b["input_features_sha256"],
                             "input_ids_sha256": b["input_ids_sha256"],
                             "attention_mask_sha256": b["attention_mask_sha256"],
                             "audio_placeholder_count": b["audio_placeholder_count"],
                             "projector_word_rows": list(rows),
                             "recipient_aliases": [item["token_ids_with_eos"] for item in aliases[recipient]],
                             "donor_aliases": [item["token_ids_with_eos"] for item in aliases[donor]],
                             "planned_generations": list(GENERATIONS),
                             "planned_score_conditions": list(SCORED)})
    require(len(plan) == CONTEXT_COUNT and len({row["context_id"] for row in plan}) == 16,
            "Sixteen fixed directed contexts not enumerated")
    return plan


def expected_context_call_labels(context: Mapping[str, Any]) -> list[str]:
    """Exact V5 singleton schedule including the natural clean capture."""
    labels = [f"generation|{arm}" for arm in GENERATIONS]
    for word in ("recipient", "donor"):
        for ordinal, _alias in enumerate(context[f"{word}_aliases"]):
            labels.append(f"grad|clean|{word}|alias_{ordinal}")
    for arm in SCORED:
        for word in ("recipient", "donor"):
            for ordinal, _alias in enumerate(context[f"{word}_aliases"]):
                labels.append(f"no_grad|{arm}|{word}|alias_{ordinal}")
    require(len(labels) <= 7 + MAX_NO_GRAD_SCORES_PER_CONTEXT +
            MAX_GRAD_SCORES_PER_CONTEXT and len(set(labels)) == len(labels),
            "Frozen context call schedule exceeds V5 bounds")
    return labels


def make_batch(processor: Any, context: dict, wav_path: Path) -> dict[str, torch.Tensor]:
    """Reprocess one pinned WAV and bind all tensor bytes before model use."""
    require(wav_path.is_file() and sha(wav_path) == context["wav_sha256"],
            "Track B recipient WAV changed")
    conversation = [{"role": "user", "content": [
        {"type": "audio", "path": str(wav_path)}, {"type": "text", "text": preflight.PROMPT}]}]
    batch = processor.apply_chat_template(conversation, tokenize=True,
                                          return_dict=True, return_tensors="pt")
    require(set(batch) == {"input_ids", "attention_mask", "input_features"} and
            preflight.tensor_sha(batch["input_ids"]) == context["input_ids_sha256"] and
            preflight.tensor_sha(batch["attention_mask"]) == context["attention_mask_sha256"] and
            preflight.tensor_sha(batch["input_features"]) == context["input_features_sha256"] and
            int((batch["input_ids"] == preflight.AUDIO_ID).sum()) ==
            context["audio_placeholder_count"], "Track B processor input bytes changed")
    return dict(batch)


def _decoded_answer(processor: Any, suffix: list[int], recipient: str, donor: str,
                    recipient_aliases: Sequence[Sequence[int]],
                    donor_aliases: Sequence[Sequence[int]]) -> dict:
    decoded = processor.batch_decode([suffix], skip_special_tokens=True)[0]
    with_special = processor.batch_decode([suffix], skip_special_tokens=False)[0]
    return {"suffix_ids": suffix, "decoded": decoded, "decoded_with_special": with_special,
            "class": core.classify(suffix, decoded, set(processor.tokenizer.all_special_ids),
                                   core.EOS, recipient, donor),
            "terminal_eos": bool(suffix and suffix[-1] == core.EOS),
            "recipient_alias_covered": tuple(suffix) in {tuple(a) for a in recipient_aliases},
            "donor_alias_covered": tuple(suffix) in {tuple(a) for a in donor_aliases}}


def clean_generation_with_capture(model: Any, processor: Any, batch: Mapping[str, torch.Tensor],
                                  context: dict, pad: int,
                                  call_guard: Callable[[str], None] | None = None) -> tuple[dict, torch.Tensor]:
    """One natural generation, capturing its pre-LM projector state in prefill."""
    found = []
    t = context["audio_placeholder_count"]
    def hook(_module, _args, output):
        require(isinstance(output, torch.Tensor) and output.shape == (t, core.PROJECTOR_WIDTH)
                and output.dtype == torch.float32 and bool(torch.isfinite(output).all()),
                "Natural projector capture shape/dtype/finiteness invalid")
        found.append(output.detach().clone())
        require(len(found) == 1, "Natural generation projector called more than once")
        return output
    handle = model.model.multi_modal_projector.register_forward_hook(hook)
    try:
        if call_guard is not None:
            call_guard("before|generation|clean")
        with torch.no_grad():
            output = model.generate(**batch, do_sample=False, use_cache=True,
                                    max_new_tokens=32, eos_token_id=core.EOS, pad_token_id=pad)
        if call_guard is not None:
            call_guard("after|generation|clean")
    finally:
        handle.remove()
    require(len(found) == 1 and output.ndim == 2 and output.shape[0] == 1 and
            1 <= output.shape[1] - batch["input_ids"].shape[1] <= 32,
            "Natural generation suffix or prefill hook invalid")
    suffix = output[0, batch["input_ids"].shape[1]:].tolist()
    answer = _decoded_answer(processor, suffix, context["recipient"], context["donor"],
                             context["recipient_aliases"], context["donor_aliases"])
    answer["hook"] = {"hook_calls": 1, "source_sha256": core.tensor_sha(found[0]),
                      "source_shape": list(found[0].shape), "source_dtype": str(found[0].dtype),
                      "captured_during_generation": True}
    return answer, found[0]


def score_class_singletons(model: Any, batch: Mapping[str, torch.Tensor],
                           aliases: Sequence[Sequence[int]], source: torch.Tensor,
                           edit: core.Edit | None, *, pad: int, rows: Sequence[int],
                           grad: bool, call_guard: Callable[[str], None],
                           label: str) -> dict:
    """Use the reviewed single-alias core, checking bounds after every backward."""
    require(1 <= len(aliases) <= 6 and len({tuple(a) for a in aliases}) == len(aliases),
            "Frozen alias class incomplete or duplicated")
    entries = []
    values = []
    gradients = []
    for ordinal, alias in enumerate(aliases):
        call_id = ("grad" if grad else "no_grad") + "|" + label + f"|alias_{ordinal}"
        call_guard("before|" + call_id)
        got = core.score_alias_class(model, batch, (alias,), source, edit, pad=pad,
                                     grad=grad, gradient_rows=rows if grad else None)
        call_guard("after|" + call_id)
        require(got["forward_count"] == 1 and got["aliases_with_eos"] == [list(alias)] and
                len(got["hooks"]) == 1 and got["hooks"][0]["hook_calls"] == 1,
                "Singleton score forward ledger differs")
        q = float(got["alias_logprobs"][0])
        require(math.isfinite(q), "Nonfinite alias score")
        values.append(q)
        if grad:
            require(got["gradient"] is not None and
                    got["gradient"].shape == source.shape and
                    len(got["per_alias_selected_gradients"]) == 1,
                    "Singleton gradient evidence missing")
            gradients.append(got["gradient"].detach())
        entries.append({"ordinal": ordinal, "token_ids_with_eos": list(alias),
                        "token_logprobs_including_eos": got["token_logprobs_including_eos"][0],
                        "q": q, "hook": got["hooks"][0],
                        "selected_gradient": got["per_alias_selected_gradients"][0]
                        if grad else None})
    v = torch.tensor(values, dtype=torch.float64)
    mass = torch.logsumexp(v, dim=0)
    weights = torch.softmax(v, dim=0)
    aggregate = None
    if grad:
        aggregate = sum((float(w) * g for w, g in zip(weights, gradients)),
                        torch.zeros_like(source))
        require(bool(torch.isfinite(aggregate).all()), "Class weighted source gradient nonfinite")
    return {"log_mass": float(mass), "entries": entries,
            "alias_weights": weights.tolist(), "gradient": aggregate,
            "forward_count": len(entries), "gradient_accumulation_dtype":
            str(aggregate.dtype) if aggregate is not None else None}


def _class_parity(grad: dict, clean: dict, zero: dict) -> list[str]:
    problems = []
    for label, reference in (("clean", clean), ("zero", zero)):
        if len(grad["entries"]) != len(reference["entries"]):
            problems.append(f"{label}_alias_count")
            continue
        for a, b in zip(grad["entries"], reference["entries"]):
            if (a["token_ids_with_eos"] != b["token_ids_with_eos"] or
                    abs(a["q"] - b["q"]) > 1e-4 or
                    a["hook"]["pre_hashes"] != b["hook"]["pre_hashes"] or
                    a["hook"]["post_hashes"] != b["hook"]["post_hashes"]):
                problems.append(f"{label}_alias_{a['ordinal']}_parity")
    return problems


def summarize_decision(context: dict, generations: dict, scores: dict,
                       grad_recipient: dict, grad_donor: dict,
                       direction: torch.Tensor, rows: Sequence[int]) -> dict:
    """Frozen V3+V5 endpoints; all planned contexts stay in denominator."""
    s0_grad = grad_donor["log_mass"] - grad_recipient["log_mass"]
    grad_s = grad_donor["gradient"] - grad_recipient["gradient"]
    require(grad_s is not None and bool(torch.isfinite(grad_s).all()),
            "Missing/nonfinite answer Jacobian")
    j = float((grad_s[list(rows)] * direction.to(grad_s.device)).sum())
    require(math.isfinite(j), "Nonfinite answer Jacobian contraction")
    margins = {arm: score["donor"]["log_mass"] - score["recipient"]["log_mass"]
               for arm, score in scores.items()}
    parity = (_class_parity(grad_recipient, scores["clean"]["recipient"], scores["zero"]["recipient"]) +
              _class_parity(grad_donor, scores["clean"]["donor"], scores["zero"]["donor"]))
    if abs(s0_grad - margins["zero"]) > 1e-4:
        parity.append("grad_clean_vs_nograd_zero_margin")
    if abs(margins["clean"] - margins["zero"]) > 1e-4:
        parity.append("clean_vs_zero_margin")
    if generations["clean"]["suffix_ids"] != generations["zero"]["suffix_ids"]:
        parity.append("clean_vs_zero_suffix")
    if generations["zero"]["hook"]["pre_hashes"] != generations["zero"]["hook"]["post_hashes"]:
        parity.append("zero_source_bits")
    if generations["full_donor"]["suffix_ids"] != generations["natural_donor"]["suffix_ids"]:
        parity.append("full_donor_suffix")
    local = {}
    for arm, alpha in (("plus_0_1", .1), ("minus_0_1", -.1)):
        predicted = alpha * j
        measured = margins[arm] - margins["zero"]
        local[arm] = {"alpha": alpha, "predicted_delta": predicted,
                      "measured_delta": measured,
                      "error": abs(measured - predicted),
                      "predicted_sign_matches": (measured > 0) == (predicted > 0) if measured != 0 and predicted != 0 else False,
                      "error_within_rule": abs(measured - predicted) <= max(.05, .25 * abs(predicted)),
                      "switch_clauses": core.predicted_switch_clauses(
                          generations["clean"], generations["zero"], generations[arm],
                          s0=margins["zero"], j=j, alpha=alpha, measured_s=margins[arm])}
    resolved = abs(.1 * j) >= .02
    score_pass = resolved and not parity and all(x["predicted_sign_matches"] and
                                                 x["error_within_rule"] for x in local.values())
    raw_switches = {arm: bool(generations["clean"]["class"] == "recipient" and
                              generations["zero"]["class"] == "recipient" and
                              generations[arm]["class"] == "opposite" and
                              generations[arm]["terminal_eos"])
                    for arm in ("plus_0_1", "minus_0_1", "random_plus_0_1", "plus_1_0")}
    return {"s0_grad": s0_grad, "s0_nograd_zero": margins["zero"], "j": j,
            "margins": margins, "local_signed": local, "local_resolved": resolved,
            "local_score_pass": score_pass, "parity_failures": parity,
            "raw_donor_switches": raw_switches,
            "predicted_switch_count": sum(x["switch_clauses"]["predicted_complete_answer_switch"]
                                          for x in local.values()) if not parity else 0,
            "random_switch": raw_switches["random_plus_0_1"],
            "direction_gradient_sha256": core.tensor_sha(grad_s)}


def _score_evidence(value: dict) -> dict:
    """Keep the exact alias scores and gradient hashes without JSON tensors."""
    return {"log_mass": value["log_mass"], "alias_weights": value["alias_weights"],
            "forward_count": value["forward_count"],
            "gradient_accumulation_dtype": value["gradient_accumulation_dtype"],
            "entries": [{k: v for k, v in item.items() if k != "selected_gradient"}
                        | ({"selected_gradient_sha256": core.tensor_sha(item["selected_gradient"]),
                            "selected_gradient_shape": list(item["selected_gradient"].shape)}
                           if item["selected_gradient"] is not None else {})
                        for item in value["entries"]]}


def run_one_context(model: Any, processor: Any, context: dict,
                    batch: Mapping[str, torch.Tensor], source: torch.Tensor,
                    donor_source: torch.Tensor, clean: dict, donor_clean: dict,
                    *, pad: int, call_guard: Callable[[str], None]) -> tuple[dict, dict]:
    """Execute the fixed six remaining generations and 84 singleton scores."""
    rows = tuple(context["projector_word_rows"])
    require(source.shape == donor_source.shape and source.ndim == 2 and
            source.shape[1] == core.PROJECTOR_WIDTH and
            source.dtype == donor_source.dtype == torch.float32 and
            bool(torch.isfinite(source).all()) and bool(torch.isfinite(donor_source).all()),
            "Matched FP32 natural source geometry differs")
    direction = core.mean_direction(source, donor_source, rows)
    random, random_info = core.random_control(direction, context["context_id"])
    edits = {"zero": core.Edit("zero"),
             "plus_0_1": core.Edit("vector", rows, direction, .1),
             "minus_0_1": core.Edit("vector", rows, direction, -.1),
             "random_plus_0_1": core.Edit("vector", rows, random, .1),
             "plus_1_0": core.Edit("vector", rows, direction, 1.0),
             "full_donor": core.Edit("full", donor=donor_source)}
    r_aliases = tuple(tuple(a) for a in context["recipient_aliases"])
    d_aliases = tuple(tuple(a) for a in context["donor_aliases"])
    generations = {"clean": clean, "natural_donor": donor_clean}
    for arm in GENERATIONS[1:]:
        call_guard("before|generation|" + arm)
        generations[arm] = core.generate_answer(
            model, processor, batch, context["recipient"], context["donor"],
            source=source, edit=edits[arm], pad=pad,
            recipient_aliases=r_aliases, donor_aliases=d_aliases)
        call_guard("after|generation|" + arm)
    # The clean differentiable score uses the same bit-preserving zero hook as
    # the matched no-grad zero score; no separate clean audio source forward.
    grad = {}
    for word, aliases in (("recipient", r_aliases), ("donor", d_aliases)):
        grad[word] = score_class_singletons(model, batch, aliases, source,
                                            core.Edit("zero"), pad=pad, rows=rows,
                                            grad=True, call_guard=call_guard,
                                            label=f"clean|{word}")
    scores = {}
    for arm in SCORED:
        scores[arm] = {}
        for word, aliases in (("recipient", r_aliases), ("donor", d_aliases)):
            scores[arm][word] = score_class_singletons(
                model, batch, aliases, source, edits.get(arm), pad=pad, rows=rows,
                grad=False, call_guard=call_guard, label=f"{arm}|{word}")
    decision = summarize_decision(context, generations, scores, grad["recipient"],
                                  grad["donor"], direction, rows)
    evidence = {"schema": "voxtral_fp32_track_b_context_v1",
                "index": context["index"], "context_id": context["context_id"],
                "status": "completed_with_controls" if not decision["parity_failures"] else "completed_controls_failed",
                "context": context, "source_sha256": core.tensor_sha(source),
                "donor_source_sha256": core.tensor_sha(donor_source),
                "direction_sha256": core.tensor_sha(direction),
                "direction_l2": float(torch.linalg.vector_norm(direction)),
                "random_control": random_info,
                "generations": generations,
                "scores": {arm: {word: _score_evidence(value) for word, value in pair.items()}
                           for arm, pair in scores.items()},
                "grad_scores": {word: _score_evidence(value) for word, value in grad.items()},
                "decision": decision,
                "actual_counts": {"generations": 7,
                                  "no_grad_singleton_score_forwards": sum(
                                      v["forward_count"] for pair in scores.values() for v in pair.values()),
                                  "grad_singleton_score_forwards": sum(
                                      v["forward_count"] for v in grad.values())}}
    require(evidence["actual_counts"]["no_grad_singleton_score_forwards"] <=
            MAX_NO_GRAD_SCORES_PER_CONTEXT and
            evidence["actual_counts"]["grad_singleton_score_forwards"] <=
            MAX_GRAD_SCORES_PER_CONTEXT,
            "Context exceeded frozen V5 singleton score count")
    selected_gradients = {f"{word}_{item['ordinal']}": item["selected_gradient"]
                          for word, value in grad.items() for item in value["entries"]}
    return evidence, selected_gradients


def validate_only() -> dict:
    """Run the no-weight gate and freeze the full study ledger; always HOLD."""
    sources = verify_sources()
    receipt = preflight.run(check_host=False)
    plan = plan_from_preflight(receipt)
    return {"schema": "voxtral_fp32_track_b_study_source_validation_v1",
            "status": "HOLD_no_candidate_model_inference",
            "source_sha256": sha(Path(__file__)), "source_pins": sources,
            "preflight_source_sha256": sha(Path(preflight.__file__)),
            "preflight_alias_sha256": receipt["processor"]["alias_table_sha256"],
            "preflight_zero_geometry_sha256": json_sha(receipt["v6_synthetic_zero_geometry"]),
            "planned_contexts": len(plan), "plan_sha256": json_sha(plan),
            "planned_generations": 16 * 7,
            "max_no_grad_singleton_scores": 16 * 72,
            "max_grad_singleton_scores": 16 * 12,
            "hold_reasons": receipt["hold_reasons"] +
                            ["pilot_measurement_and_independent_go_schema_not_yet_frozen"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if not args.validate_only:
        parser.error("Live Track B study is unavailable until the pilot receipt and independent GO are frozen")
    print(json.dumps(validate_only(), sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
