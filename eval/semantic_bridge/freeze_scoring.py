"""Freeze exact Omni token candidates, exclusions, and semantic-token mask.

Loads only the pinned tokenizer and source manifest on CPU. No model weights,
activations, audio, or accelerator are accessed.
"""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

from transformers import AutoTokenizer, __version__ as transformers_version

from scoring import label_token_surfaces, semantic_token_ids


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
SNAPSHOT = (
    PROJECT / "models/qwen_omni/.hf_cache/hub/models--Qwen--Qwen2.5-Omni-3B/snapshots"
    / "f75b40e3da2003cdd6e1829b1f420ca70797c34e"
)
MANIFEST_SHA256 = "035a12e0388f0fb1c29dc2762c8c070d8052f64202c05c5198b6230d25d2ccf3"
SMOKE_SHA256 = "12f4f6ab4d80542e075422ed17a5c3edbc2b7fb52d13852e96374be76fdb4c96"
TOKENIZER_SHA256 = "8441917e39ae0244e06d704b95b3124795cec478e297f9afac39ba670d7e9d99"
MODEL_REVISION = "f75b40e3da2003cdd6e1829b1f420ca70797c34e"
HEAD_VOCAB_SIZE = 151936
FACTUAL_EXCLUSIONS = {
    "multihop:super-populous-capital": "outdated_premise_india_overtook_china_2023",
}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_bytes(obj) -> bytes:
    return (json.dumps(obj, sort_keys=True, ensure_ascii=False, indent=2) + "\n").encode()


def jsonl_bytes(rows) -> bytes:
    return b"".join((json.dumps(x, sort_keys=True, ensure_ascii=False) + "\n").encode()
                     for x in rows)


def write_or_check(path: Path, data: bytes) -> None:
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"Frozen artifact differs: {path}; create a versioned amendment")
    else:
        path.write_bytes(data)


def main() -> None:
    source = (ROOT / "manifest.jsonl").read_bytes()
    if sha(source) != MANIFEST_SHA256:
        raise ValueError("Source manifest changed")
    if sha((SNAPSHOT / "tokenizer.json").read_bytes()) != TOKENIZER_SHA256:
        raise ValueError("Pinned tokenizer bytes changed")
    tok = AutoTokenizer.from_pretrained(SNAPSHOT, local_files_only=True, use_fast=True)
    if not tok.is_fast or tok.vocab_size != 151643:
        raise ValueError("Unexpected tokenizer implementation/vocabulary")

    manifest = [json.loads(line) for line in source.splitlines()]
    scored, excluded_items, excluded_labels = [], [], []
    for item in manifest:
        entry = {"id": item["id"], "category": item["category"],
                 "split": item["split"], "template_group": item["template_group"],
                 "source_prompt_sha256": item["source_prompt_sha256"],
                 "source_labels": item["expected_bridge_labels"],
                 "scorable_labels": []}
        for label in item["expected_bridge_labels"]:
            surfaces = label_token_surfaces(tok, label, HEAD_VOCAB_SIZE)
            if not surfaces:
                excluded_labels.append({"id": item["id"], "label": label,
                                        "reason": "no_single_semantic_token_surface"})
            else:
                entry["scorable_labels"].append({"label": label, "token_surfaces": surfaces})
        reason = FACTUAL_EXCLUSIONS.get(item["id"])
        if reason is None and not entry["scorable_labels"]:
            reason = "no_scorable_bridge_label"
        entry["scoring_status"] = "excluded" if reason else "scorable"
        entry["exclusion_reason"] = reason
        if reason:
            excluded_items.append({"id": item["id"], "split": item["split"],
                                   "category": item["category"], "reason": reason})
        else:
            scored.append(entry)

    # A semantic-token subset is a secondary diagnostic only; primary metric
    # retains the entire 151,936-row text LM head competition set.
    allowed = semantic_token_ids(tok, HEAD_VOCAB_SIZE)
    packed = struct.pack(f"<{len(allowed)}I", *allowed)
    all_bytes = jsonl_bytes(scored)
    exclusions = {"items": excluded_items, "labels": excluded_labels}
    metadata = {
        "schema_version": 1,
        "source_manifest_sha256": MANIFEST_SHA256,
        "model_id": "Qwen/Qwen2.5-Omni-3B",
        "model_revision": MODEL_REVISION,
        "head_vocab_size": HEAD_VOCAB_SIZE,
        "tokenizer_class": type(tok).__name__,
        "tokenizer_fast": tok.is_fast,
        "tokenizer_vocab_size": tok.vocab_size,
        "tokenizer_length_with_added": len(tok),
        "tokenizer_json_sha256": TOKENIZER_SHA256,
        "tokenizer_config_sha256": sha((SNAPSHOT / "tokenizer_config.json").read_bytes()),
        "chat_template_sha256": sha((SNAPSHOT / "chat_template.json").read_bytes()),
        "transformers_version": transformers_version,
        "surface_rule": "original/lower/capitalize, each bare and one-leading-space; keep only one-token semantic non-special/non-added surfaces; no synonyms",
        "full_vocab_rank_rule": "1 + count(logit greater than best candidate logit), over all 151936 LM-head rows",
        "secondary_semantic_rank_rule": "same rank after excluding special, added, blank/padding and tokens with no alphanumeric decoded character",
        "semantic_token_count": len(allowed),
        "semantic_ids_uint32le_sha256": sha(packed),
        "scorable_item_count": len(scored),
        "excluded_item_count": len(excluded_items),
        "excluded_label_count": len(excluded_labels),
        "scorable_manifest_sha256": sha(all_bytes),
        "exclusions_sha256": sha(json_bytes(exclusions)),
        "scorable_ids": [x["id"] for x in scored],
    }
    smoke_bytes = (ROOT / "smoke.jsonl").read_bytes()
    if sha(smoke_bytes) != SMOKE_SHA256:
        raise ValueError("Fit-only smoke manifest changed")
    smoke = [json.loads(line) for line in smoke_bytes.splitlines()]
    if len(smoke) != 12 or any(item["split"] != "fit" for item in smoke):
        raise ValueError("Framing gate must be exactly 12 fit-only items")
    framing_gate = {
        "schema_version": 1,
        "status": "specified_not_run",
        "source_smoke_sha256": SMOKE_SHA256,
        "model_revision": MODEL_REVISION,
        "route": "Qwen2.5-Omni-3B Thinker text output, separately for typed-chat and audio-chat user content",
        "system_prompt": "Respond to the user's message briefly and directly.",
        "generation": {"do_sample": False, "max_new_tokens": 128,
                       "stop": "model_eos", "repetitions": 2},
        "judging": "For each item and modality, retain full decoded outputs. Three independent reviewers who did not see the model's activations judge whether the answer conveys the source association concept, or gives the source multihop target answer. Majority 2/3 passes; if repeated outputs disagree, the item fails.",
        "gate": "At least 5 of 6 fit-smoke items pass in each category and each modality before interpreting held-out bridge ranks as a valid Omni task. If a category fails, report ranks only as exploratory diagnostic and define/version a new explicit-question prompt before a new held-out test.",
        "items": [{"id": item["id"], "category": item["category"],
                   "typed_prompt": item["typed_input"]["prompt"],
                   "audio_script": item["audio_input"]["script"],
                   "expected_bridge_labels": item["expected_bridge_labels"],
                   "expected_next_text": item["expected_next_text"]}
                  for item in smoke],
    }
    write_or_check(ROOT / "scorable_manifest.jsonl", all_bytes)
    write_or_check(ROOT / "exclusions.json", json_bytes(exclusions))
    write_or_check(ROOT / "scoring_metadata.json", json_bytes(metadata))
    write_or_check(ROOT / "framing_gate.json", json_bytes(framing_gate))
    print(f"scorable={len(scored)} excluded={len(excluded_items)} labels_dropped={len(excluded_labels)}")
    print(f"scorable_sha256={metadata['scorable_manifest_sha256']}")
    print(f"semantic_mask_ids={len(allowed)} sha256={metadata['semantic_ids_uint32le_sha256']}")


if __name__ == "__main__":
    main()
