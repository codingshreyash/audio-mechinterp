"""No-weight, fail-closed preflight for the prospective Voxtral Track B study.

This source can prepare/hash frozen inputs, but cannot authorize inference while
human QA, an independent runner review, and a measured memory bound are absent.
No candidate model is constructed or called by this file.
"""
from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import io
import inspect
import json
import math
import os
import plistlib
import shutil
import struct
import subprocess
import sys
import tempfile
import wave
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import voxtral_fit_pilot_capability_v1 as base

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
V3 = HERE / "VOXTRAL_FP32_TRACK_B_ANSWER_LINK_PROSPECTIVE_V3.md"
V4 = HERE / "VOXTRAL_FP32_TRACK_B_ANSWER_LINK_PROSPECTIVE_V4_ERRATUM.md"
V5 = HERE / "VOXTRAL_FP32_TRACK_B_ANSWER_LINK_PROSPECTIVE_V5_SEQUENTIAL_ALIAS_AMENDMENT.md"
V6 = HERE / "VOXTRAL_FP32_TRACK_B_ANSWER_LINK_PROSPECTIVE_V6_RESOURCE_PILOT_AMENDMENT.md"
V6_DESIGN_GO = HERE / "VOXTRAL_FP32_TRACK_B_ANSWER_LINK_PROSPECTIVE_V6_INDEPENDENT_REVIEW_GO.md"
DESIGN_GO = HERE / "VOXTRAL_FP32_TRACK_B_ANSWER_LINK_PROSPECTIVE_V3_PLUS_V4_INDEPENDENT_GO.md"
TRACK = HERE / "word_disjoint_v2"
MANIFEST = TRACK / "rendered_v2/manifest.json"
BINDING = TRACK / "rendered_v2/TECHNICAL_BINDING.json"
INVENTORY = TRACK / "RENDERED_V2_FINAL_WAV_INVENTORY_V1.json"
PROMPT = 'What word did the speaker say between "said" and "and"? Answer with exactly one word. No explanation or punctuation.'
PAIR_ORDER = (("banana", "tomato"), ("carrot", "cookie"),
              ("train", "plane"), ("ferry", "taxi"))
ENGINES = ("piper", "samantha")
SURFACES = ("lower", "initial", "upper")
EOS = 2
AUDIO_ID = 24
MAX_NEW_TOKENS = 32
MAX_DRIVER_BYTES = 30 * 1024**3
MAX_OUTPUT_BYTES = 2 * 1024**3
MIN_FREE_DISK_BYTES = 12 * 1024**3
SOURCE_HASHES = {
    V3: "f3716897cfff30b26a4bb66d882222fc2137b0911266d76ea0a0c4319854ce0b",
    V4: "87ef41b3c854ed23fc4cfb3f67a23f2b6cd595958a1f941bc9726288ea0d923c",
    V5: "70d24e5d4ba341e1aaa918608d57f1ed237fd56742b0d956816f4ce636ac3115",
    V6: "6a341f316e4056c6bc81c6e191deea3f90dcddcd81050a39c734a257d4f02b9c",
    V6_DESIGN_GO: "28341f87e9eb7ba8a7827460f2872d01ea17adcab3f3a36b649d4542eacae0b1",
    DESIGN_GO: "bd5340733522c4609b4136ba9263dd734f64ff6cfe753c40bb1c9f803314d95f",
    MANIFEST: "f8fd1f43f38877d6d08c47597c8db9f494e1377a69f0a2451c2f99fbcda71ea2",
    BINDING: "027763d4fa758dac95c552c2f2aef60683f1c780b0c034d82327be8bd59e93b9",
    INVENTORY: "6fa29354f488dea5a999818f192a24a99d7c7a6fe07d40ada52de79c9eaf2f8c",
    TRACK / "candidate_common.py": "01e3ece95a8e3c1202369185061c29d6b5307761299df3f23115300164b18312",
    TRACK / "audit.py": "6411c2961ba31700f3c3e099124644c80c38e52c925c4ef02cb9d01173cd831f",
    TRACK / "render.py": "f6524f30f2655c9f62fdc4898be5e480c20c5e0ba84c5e19427ad437656d044c",
    TRACK / "queue.py": "1744be6270352f86fb1990f9ef7223734514dd05a9ec14b717876c8b28d15e3c",
    HERE / "voxtral_fp32_nonbeaver_linearity_continuation_gate_v1.py":
        "1fbc29e7851491d53c1c8245a7acd385563a1491435d21790a4370a7aa3492be",
}
INSTALLED_HASHES = {
    "modeling_voxtral": "2d3ec9d78c469b77c42a8d5ba258185fb1c58f4c2c7bf6007ec724436dfc7056",
    "processing_voxtral": "239e84657d1a0287ab27a540c946a787a69b4b0bf45727d6b24bee21ccec0296",
    "feature_extraction_whisper": "dcce2e7820be059e657a9e12a60e8a55cfd37b1a82ee202ca15fc7d0934374cb",
    "modeling_llama": "affc6a4ab36b562a6b4c90b84c4e7939c5417a69ad7bd37e858be5583f189830",
    "generation_utils": "7944240ff1ee27a7b3acc16571b7a480cc14f16e7222aeb1a79d289ebc96e3ea",
    "mistral_tokenizer": "f132c6dbcce84654bb7f35d13c7f1b6b3de01270d3d6d027fc5d4124431da6e3",
}
PROCESSOR_CONFIG_SHA = "86d67d926e17d3a9bb0bae334c9fc46bc163181992357a0357c0aec2c7e131d1"
TOKENIZER_SHA = "4aaf3836c2a5332f029ce85a7a62255c966f47b6797ef81dedd0ade9c862e4a8"
PROMPT_IDS_SHA = "fe34532345905beef2432631ac5d80328cbeeceae55f587829eb60c084f24319"
PROMPT_MASK_SHA = "19c70da448ff0623e86241ad9911029ad7699979a978bd02024970d6fbd9a65f"
ALIAS_TABLE_SHA = "fae95bea6feba3b5d2506ce2a6a439c037e132893e46bd9b272bf36fd572ea63"
ZERO_GEOMETRY_SHA = "2a6f93ca477438efa9ed3b937cf761ec3a4cf79cb442d34bc5bd697af01fe749"
INPUT_FEATURE_SHAS = {
    ("pair_00", "piper", "banana"): "3409a6f5021ddcf2df0baa60e66cc83b77290399831eb866c01e9ebeb21248f1",
    ("pair_00", "piper", "tomato"): "ff63df9cc71fe09a0965704fa74c97440e1fd2f941bb540fb2fd039c81b4b7c9",
    ("pair_00", "samantha", "banana"): "6fd3229dfa4c29f3c6c32447efc8a681250b4512d07c8a508689efe6319acb8b",
    ("pair_00", "samantha", "tomato"): "2886f1a4f6782c2f429bc3a2f77ad801de12605228279d044016a545cc6a14f7",
    ("pair_01", "piper", "carrot"): "2734773df761cb72e417f18fe15238940ffccdcf83745ed986ce23f9037e7f8f",
    ("pair_01", "piper", "cookie"): "b442084897449f8fe8c73128a0c0730c68832e6368eb660c59be2f46dff75b06",
    ("pair_01", "samantha", "carrot"): "a14ee027d56d43d82f674668f7503b5dd0576ba5ed2d7319dcb6e71f5a877549",
    ("pair_01", "samantha", "cookie"): "db2d0c5dfd08621cb25594c4de9ab379df767ef5f086b66af8996ff8d71d50dd",
    ("pair_02", "piper", "plane"): "9577dfe6bcbd434e47edfc5e0b3039870e21c12b2ab4d80af289f1bf136469c7",
    ("pair_02", "piper", "train"): "a762f27fcdeb0228639320c6add4b6e6e70e052573f18f87e10d24cdf8a7139c",
    ("pair_02", "samantha", "plane"): "0ed8b012b123f0d55f7110607bd10014ddbe99387a4817eb91ff0f9fe653897f",
    ("pair_02", "samantha", "train"): "fa0a77995e1314dd25629098ac8d5cbb59523e917b319d40c4278993a2e960d6",
    ("pair_03", "piper", "ferry"): "58a183dd6f239befffa256f956ab1cdbd9a82ae6c971839f5c0b1d8a0f4413be",
    ("pair_03", "piper", "taxi"): "6313d822524c60d4e53634e0a79281f490f02523a73081c9c976a2285d765d26",
    ("pair_03", "samantha", "ferry"): "044741e4b55112697119ad96c60a729be0660b60d726a80709f148b0f3d97210",
    ("pair_03", "samantha", "taxi"): "676050dae44b3f3d6a58645e287ab19b2d9b5e67c1f2092eb72fc7e5401eceee",
}
EXPECTED_WAV_SHAS = {
    ("pair_00", "piper", "banana"): "242a41dd7f88f9ed3326e6accb1420b1959401916b5aa8ced14509a6bdf9ece0",
    ("pair_00", "piper", "tomato"): "fcf682f202d1c571cde3457e8f935dd38d8aa2b2e17294cad66b7c5a12e4493d",
    ("pair_00", "samantha", "banana"): "51b3328c7baf79bd6b079e270ba11ba950631759e36c21fcc31265b5cba7b88e",
    ("pair_00", "samantha", "tomato"): "cc1e04eab07e7ac8c82e3b3bf2bfa65bddc791165e7512d42776d2af7c07d9c3",
    ("pair_01", "piper", "carrot"): "0a1d0535d8b46fd88d665bcc3473f6806fd842008b4bdd70e450c1d786fffef3",
    ("pair_01", "piper", "cookie"): "79fed2f865bbf91b68cb2c68c6afbfeb65f1d44f33ba7bb4d275d1d1d97e64ba",
    ("pair_01", "samantha", "carrot"): "fb181323f74c9160514b077dfe0746e4a40421b8dda54b84faa0a2a27f1b3c60",
    ("pair_01", "samantha", "cookie"): "087a9db681cca7a834396c6a50e85ecf72436d712512b66ec7fccab6ad9ea6ce",
    ("pair_02", "piper", "train"): "c5d32efb010ef4fdb34aa3a6675c0aeb07b06e84ab8b1b9333c315752d431fb1",
    ("pair_02", "piper", "plane"): "d9b7eb20d0b8b0db8d0a26e79cbdb6617a64abcac42791ff0871e993846945f5",
    ("pair_02", "samantha", "train"): "cb28d07d2bdbad5e302d3bb0006dd0a4d96a53535c647540e4ee3b263c010445",
    ("pair_02", "samantha", "plane"): "dc792492b3fab8222e46d507a78787cbbc966907a7d89649bc65298ecff9386b",
    ("pair_03", "piper", "ferry"): "6e12a7b710819d2aacf7cdd031adc9250189c3ae76f56788af9f37b51c7db0b8",
    ("pair_03", "piper", "taxi"): "8ce912d770e684141740163b1f0d1a52dff4c21e8c0f116c3d9d33022fe44102",
    ("pair_03", "samantha", "ferry"): "5e48b873fe15729d8e52b55ebb862be80ae105a9588695f310c88a378803f3a1",
    ("pair_03", "samantha", "taxi"): "80c55429e8c2aa15dff09948349af7fe1a1fd0ddac695a972c2872c48d8497b7",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def file_sha(path: Path) -> str:
    return base.sha(path)


def tensor_sha(tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def fp32_parameter_payload_bytes() -> int:
    """Count exact FP32 parameter payload from safetensors headers, not weights."""
    total = 0
    for name in ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"):
        with (base.SNAPSHOT / name).open("rb") as stream:
            first = stream.read(8)
            require(len(first) == 8, "Truncated weight header")
            header_size = struct.unpack("<Q", first)[0]
            require(0 < header_size <= 64 * 1024**2, "Weight header size invalid")
            header = json.loads(stream.read(header_size))
        require(isinstance(header, dict) and len(header) > 0, "Weight header invalid")
        for key, item in header.items():
            if key == "__metadata__":
                continue
            shape = item.get("shape")
            require(isinstance(shape, list) and all(type(d) is int and d > 0 for d in shape),
                    "Parameter shape invalid")
            count = math.prod(shape)
            total += count * 4
    require(total > 0, "No parameter payload")
    return total


def verify_static_anchors() -> dict[str, str]:
    actual = {str(path.relative_to(ROOT)): file_sha(path) for path in SOURCE_HASHES}
    for path, expected in SOURCE_HASHES.items():
        require(actual[str(path.relative_to(ROOT))] == expected, f"Frozen anchor changed: {path.name}")
    require("GO for the prospective design only" in DESIGN_GO.read_text(), "Independent V3+V4 GO absent")
    require("Decision: GO for the corrected V6 prospective resource-admission design only" in
            V6_DESIGN_GO.read_text(), "Independent corrected V5+V6 design GO absent")
    return actual


def verify_installed_source() -> dict[str, str]:
    from transformers.models.voxtral import modeling_voxtral, processing_voxtral
    from transformers.models.whisper import feature_extraction_whisper
    from transformers.models.llama import modeling_llama
    from transformers.generation import utils as generation_utils
    from transformers import tokenization_mistral_common
    modules = {"modeling_voxtral": modeling_voxtral,
               "processing_voxtral": processing_voxtral,
               "feature_extraction_whisper": feature_extraction_whisper,
               "modeling_llama": modeling_llama,
               "generation_utils": generation_utils,
               "mistral_tokenizer": tokenization_mistral_common}
    actual = {key: file_sha(Path(inspect.getfile(module))) for key, module in modules.items()}
    require(actual == INSTALLED_HASHES, "Pinned installed Transformers source changed")
    return actual


def selected_cases(all_cases, manifest: dict, inventory: dict, load_pcm) -> tuple[list, list[dict]]:
    expected = set(EXPECTED_WAV_SHAS)
    chosen = [case for case in all_cases if case.pair_id in {f"pair_{i:02d}" for i in range(4)}]
    keys = [(c.pair_id, c.engine, c.word) for c in chosen]
    require(len(chosen) == 16 and len(set(keys)) == 16 and set(keys) == expected,
            "Selected Track B case identity/count changed")
    require(inventory.get("final_wav_count") == 32 and
            inventory.get("manifest_sha256") == SOURCE_HASHES[MANIFEST] and
            inventory.get("technical_binding_sha256") == SOURCE_HASHES[BINDING] and
            inventory.get("all_outside_slot_pcm_equal") is True,
            "Independent WAV inventory binding changed")
    entries = inventory.get("final_wavs")
    require(isinstance(entries, list) and len(entries) == 32, "Final WAV inventory incomplete")
    indexed = {(r.get("pair_id"), r.get("engine"), r.get("word")): r for r in entries}
    require(len(indexed) == 32, "Duplicate final WAV inventory identity")
    manifest_rows = {(r.get("pair_id"), r.get("engine")): r for r in manifest.get("pairs", [])}
    require(len(manifest_rows) == 16, "Manifest pair population changed")
    by_pair = defaultdict(dict)
    receipts = []
    for c in chosen:
        key = (c.pair_id, c.engine, c.word)
        rec = indexed[key]
        m = manifest_rows[(c.pair_id, c.engine)]
        require(c.wav_sha256 == EXPECTED_WAV_SHAS[key] == rec.get("sha256") ==
                m["outputs"][c.word]["file_sha256"] == file_sha(c.path) and
                c.pcm_sha256 == rec.get("pcm_sha256") == m["outputs"][c.word]["pcm_int16_sha256"] and
                c.frames == rec.get("frames") == m["outputs"][c.word]["frames"] and
                tuple(c.slot) == (m["word_slot_start_sample"], m["word_slot_end_sample_exclusive"]) and
                str(c.path.relative_to(ROOT)) == rec.get("path"),
                "Selected WAV bytes, PCM, metadata, or V3 table differ")
        pcm = load_pcm(c)
        require(len(pcm) == c.frames and c.frames < 480000 and 0 <= c.slot[0] < c.slot[1] <= c.frames,
                "Selected WAV PCM or slot invalid")
        by_pair[(c.pair_id, c.engine)][c.word] = pcm
        receipts.append({"pair_id": c.pair_id, "engine": c.engine, "word": c.word,
                         "wav_sha256": c.wav_sha256, "pcm_sha256": c.pcm_sha256,
                         "frames": c.frames, "word_slot": list(c.slot),
                         "path": str(c.path.relative_to(ROOT))})
    for (pair_id, engine), words in by_pair.items():
        first, second = PAIR_ORDER[int(pair_id[-2:])]
        a, b = words[first], words[second]
        start, end = next(c.slot for c in chosen if c.pair_id == pair_id and c.engine == engine)
        require(len(a) == len(b) and bool((a[:start] == b[:start]).all()) and
                bool((a[end:] == b[end:]).all()) and bool((a[start:end] != b[start:end]).any()),
                "Matched pair PCM outside-slot identity or inside difference failed")
    return sorted(chosen, key=lambda c: (c.pair_id, c.engine, c.word)), receipts


def projector_rows(slot: tuple[int, int], frames: int, total_rows: int) -> list[int]:
    start, end = slot
    require(0 <= start < end <= frames and total_rows > 0, "Projector row geometry invalid")
    rows = [r for r in range(total_rows)
            if start <= 160 * (8 * r + 3) < end and 160 * (8 * r + 3) < frames]
    require(bool(rows) and all(160 * (8 * r + 3) < frames for r in rows),
            "Empty, padded, or out-of-range word row mask")
    return rows


def exact_alias_ids(tokenizer, conversation: list[dict], prompt_ids: list[int], surface: str) -> list[int]:
    prepared = tokenizer.apply_chat_template(
        [*conversation, {"role": "assistant", "content": surface}],
        tokenize=True, return_dict=True, continue_final_message=True)
    full = prepared["input_ids"]
    require(isinstance(full, list) and full[:len(prompt_ids)] == prompt_ids,
            "Alias is not an exact continuation of the pinned audio-chat prompt")
    suffix = full[len(prompt_ids):]
    require(bool(suffix) and all(type(x) is int for x in suffix) and
            all(x not in tokenizer.all_special_ids for x in suffix),
            "Alias continuation empty or contains special token")
    return suffix + [EOS]


def prepare_processor(cases: list) -> dict:
    import torch
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(str(base.SNAPSHOT), local_files_only=True)
    require(processor.tokenizer.eos_token_id == EOS and int(processor.audio_token_id) == AUDIO_ID,
            "Pinned EOS/audio token IDs changed")
    batches, prompt_ids_by_pair, aliases = [], defaultdict(list), {}
    first_case_for_word = {}
    for c in cases:
        conversation = [{"role": "user", "content": [
            {"type": "audio", "path": str(c.path)}, {"type": "text", "text": PROMPT}]}]
        batch = processor.apply_chat_template(conversation, tokenize=True,
                                              return_dict=True, return_tensors="pt")
        require(set(batch) == {"input_ids", "attention_mask", "input_features"} and
                batch["input_ids"].dtype == torch.int64 and
                batch["attention_mask"].shape == batch["input_ids"].shape and
                batch["input_features"].dtype == torch.float32 and
                batch["input_features"].shape == (1, 128, 3000) and
                bool(torch.isfinite(batch["input_features"]).all()) and
                bool(torch.all(batch["attention_mask"] == 1)),
                "Pinned processor batch keys, shape, mask, dtype, or finite values changed")
        ids = batch["input_ids"][0].tolist()
        count = ids.count(AUDIO_ID)
        require(count == 375, "Audio placeholder/projector row count changed")
        key = (c.pair_id, c.engine, c.word)
        require(tensor_sha(batch["input_ids"]) == PROMPT_IDS_SHA and
                tensor_sha(batch["attention_mask"]) == PROMPT_MASK_SHA and
                tensor_sha(batch["input_features"]) == INPUT_FEATURE_SHAS[key],
                "Frozen prompt or processor batch bytes changed")
        rows = projector_rows(c.slot, c.frames, count)
        prompt_ids_by_pair[(c.pair_id, c.engine)].append(ids)
        batches.append({"pair_id": c.pair_id, "engine": c.engine, "word": c.word,
                        "input_ids": ids, "input_ids_sha256": tensor_sha(batch["input_ids"]),
                        "attention_mask_sha256": tensor_sha(batch["attention_mask"]),
                        "input_features_sha256": tensor_sha(batch["input_features"]),
                        "input_features_shape": list(batch["input_features"].shape),
                        "audio_placeholder_count": count, "projector_word_rows": rows})
        first_case_for_word.setdefault(c.word, (conversation, ids))
    require(len(prompt_ids_by_pair) == 8 and all(len(rows) == 2 and rows[0] == rows[1]
                                                for rows in prompt_ids_by_pair.values()),
            "Donor/recipient prompt IDs differ within paired comparison")
    for word in sorted(first_case_for_word):
        conversation, ids = first_case_for_word[word]
        surfaces = (word, " " + word, word.capitalize(), " " + word.capitalize(),
                    word.upper(), " " + word.upper())
        seqs, seen = [], set()
        for surface in surfaces:
            sequence = exact_alias_ids(processor.tokenizer, conversation, ids, surface)
            key = tuple(sequence)
            if key not in seen:
                seen.add(key)
                seqs.append({"surface": surface, "token_ids_with_eos": sequence})
        require(1 <= len(seqs) <= 6 and all(s["token_ids_with_eos"][-1] == EOS for s in seqs),
                "Alias count/EOS invalid")
        aliases[word] = seqs
    alias_hash = hashlib.sha256(json.dumps(aliases, sort_keys=True,
                                separators=(",", ":")).encode()).hexdigest()
    require(alias_hash == ALIAS_TABLE_SHA, "Frozen contextual alias plus EOS IDs changed")
    return {"processor_type": type(processor).__name__, "tokenizer_type": type(processor.tokenizer).__name__,
            "prompt": PROMPT, "batches": batches, "aliases": aliases,
            "alias_table_sha256": alias_hash}


def resource_estimate(prepared: dict, snapshot: dict) -> dict:
    config = json.loads((base.SNAPSHOT / "config.json").read_text())
    text = config["text_config"]
    max_aliases = max(len(items) for items in prepared["aliases"].values())
    max_alias_len = max(len(item["token_ids_with_eos"]) for items in prepared["aliases"].values() for item in items)
    max_prompt = max(len(row["input_ids"]) for row in prepared["batches"])
    max_rows = max(row["audio_placeholder_count"] for row in prepared["batches"])
    shard_bytes = sum(item["size_bytes"] for item in snapshot["weight_shards"].values())
    fp32_weight_planning_bytes = 2 * shard_bytes
    fp32_parameter_floor_bytes = fp32_parameter_payload_bytes()
    require(fp32_parameter_floor_bytes <= fp32_weight_planning_bytes,
            "Parameter payload exceeds conservative shard-size planning figure")
    width, intermediate, layers, vocab = (int(text[key]) for key in
                                          ("hidden_size", "intermediate_size", "num_hidden_layers", "vocab_size"))
    # This deliberately counts several live transformer tensors per layer.
    # It is a planning allowance, not a proof of the allocator's peak.
    def estimate(batch_size: int) -> dict:
        single_activation = batch_size * (max_prompt + max_alias_len - 1) * layers * (width + intermediate) * 4
        allowance = 4 * single_activation
        projector = batch_size * max_rows * width * 4
        audio = batch_size * 128 * 3000 * 4
        logits = batch_size * max_alias_len * vocab * 4
        total = fp32_weight_planning_bytes + allowance + projector + audio + logits
        return {"batch_size": batch_size, "single_lm_activation_accounting_bytes": single_activation,
                "four_copy_activation_allowance_bytes": allowance,
                "projector_batch_bytes": projector, "audio_input_batch_bytes": audio,
                "answer_logits_batch_bytes": logits, "planning_total_bytes": total,
                "planning_fits_driver_cap": total < MAX_DRIVER_BYTES}
    sequential = estimate(1)
    batched = estimate(max_aliases)
    return {"max_unique_aliases_per_word": max_aliases, "max_alias_tokens_including_eos": max_alias_len,
            "max_prompt_tokens": max_prompt, "max_projector_rows": max_rows,
            "v3_batched_workload": {"complete_generations": 112,
                                    "no_grad_class_score_forwards": 192,
                                    "grad_class_score_forwards": 32,
                                    "core_projector_calls": 336},
            "v5_sequential_workload": {"complete_generations": 112,
                                       "max_no_grad_singleton_score_forwards": 1152,
                                       "max_grad_singleton_score_forwards": 192,
                                       "max_core_projector_calls": 1456},
            "fp32_weight_planning_bytes": fp32_weight_planning_bytes,
            "fp32_parameter_payload_floor_bytes": fp32_parameter_floor_bytes,
            "v3_alias_batch_sizing": batched, "v5_sequential_b1_sizing": sequential,
            "driver_cap_bytes": MAX_DRIVER_BYTES,
            "can_certify_mps_peak_without_model": False,
            "meaning": "FP32 parameter payload is a lower bound. Four-copy activation accounting is only a planning allowance, not a proven peak. MPS allocator/cache and backward peak require a separately approved measured gate."}


def synthetic_zero_geometry(prepared: dict, wavs: list[dict]) -> dict:
    """Freeze V6's first componentwise max-geometry zero-audio timing input."""
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(str(base.SNAPSHOT), local_files_only=True)
    max_frames = max(row["frames"] for row in wavs)
    target_feature = tuple(max(row["input_features_shape"][axis] for row in prepared["batches"])
                           for axis in range(3))
    target_audio = max(row["audio_placeholder_count"] for row in prepared["batches"])
    target_attention = max(len(row["input_ids"]) for row in prepared["batches"])
    require(max_frames > 0 and target_feature == (1, 128, 3000), "Selected maximum geometry invalid")
    require(shutil.disk_usage(HERE).free >= MIN_FREE_DISK_BYTES,
            "Less than 12 GiB free before synthetic geometry")
    aliases = [(word, item["surface"], item["token_ids_with_eos"])
               for word, items in prepared["aliases"].items() for item in items]
    # Longest sequence, then lexicographically smallest token-ID sequence.
    word, surface, sequence = min(aliases,
                                  key=lambda row: (-len(row[2]), tuple(row[2]), row[0], row[1]))
    with tempfile.TemporaryDirectory(prefix="voxtral_track_b_geometry_") as directory:
        wav = Path(directory) / "zero.wav"
        for increment in range(513):
            frames = max_frames + increment * 128
            pcm = bytes(2 * frames)
            buffer = io.BytesIO()
            with wave.open(buffer, "wb") as writer:
                writer.setnchannels(1)
                writer.setsampwidth(2)
                writer.setframerate(16000)
                writer.writeframes(pcm)
            wav_bytes = buffer.getvalue()
            require(len(wav_bytes) < MAX_OUTPUT_BYTES, "Synthetic geometry exceeds output cap")
            wav.write_bytes(wav_bytes)
            conversation = [{"role": "user", "content": [
                {"type": "audio", "path": str(wav)}, {"type": "text", "text": PROMPT}]}]
            batch = processor.apply_chat_template(conversation, tokenize=True,
                                                  return_dict=True, return_tensors="pt")
            input_ids = batch["input_ids"][0].tolist()
            feature_shape = tuple(batch["input_features"].shape)
            audio_count = input_ids.count(AUDIO_ID)
            attention = int(batch["attention_mask"].shape[-1])
            require(bool(batch["input_features"].isfinite().all()) and
                    bool((batch["attention_mask"] == 1).all()),
                    "Synthetic processor batch nonfinite or masked")
            if (all(got >= need for got, need in zip(feature_shape, target_feature)) and
                    audio_count >= target_audio and
                    attention >= target_attention):
                receipt = {"waveform": "mono_16000hz_pcm16_all_zero",
                        "first_componentwise_satisfying_frames": frames,
                        "step_frames": 128, "max_increments": 512,
                        "selected_max_frames": max_frames,
                        "pcm_sha256": hashlib.sha256(pcm).hexdigest(),
                        "wav_sha256": hashlib.sha256(wav_bytes).hexdigest(),
                        "input_ids_sha256": tensor_sha(batch["input_ids"]),
                        "attention_mask_sha256": tensor_sha(batch["attention_mask"]),
                        "input_features_sha256": tensor_sha(batch["input_features"]),
                        "input_features_shape": list(feature_shape),
                        "audio_placeholder_count": audio_count,
                        "projector_rows_implied": audio_count,
                        "attention_mask_length": attention,
                        "longest_alias": {"word": word, "surface": surface,
                                          "token_ids_with_eos": sequence},
                        "max_selected": {"input_features_shape": list(target_feature),
                                         "audio_placeholder_count": target_audio,
                                         "attention_mask_length": target_attention,
                                         "waveform_frames": max_frames}}
                digest = hashlib.sha256(json.dumps(receipt, sort_keys=True,
                                         separators=(",", ":")).encode()).hexdigest()
                require(digest == ZERO_GEOMETRY_SHA, "Frozen V6 synthetic geometry changed")
                return receipt
    raise ValueError("No bounded componentwise maximum synthetic geometry")


def qa_status(cases: list) -> dict:
    from word_disjoint_v2 import candidate_common as common
    paths = (common.HUMAN_QA_AUDIT, common.QA_RECEIPT, common.QA_REVIEW)
    if not all(path.is_file() for path in paths):
        return {"passed": False, "reason": "Audited 32/32 human QA and independent GO are absent",
                "present": {path.name: path.is_file() for path in paths}}
    result = common.qa_gate(cases)
    return {"passed": True, "qa_receipt_sha256": result["qa_receipt_sha256"],
            "qa_review_sha256": result["qa_review_sha256"],
            "review_export_sha256": result["review_export_sha256"]}


def host_status() -> dict:
    # The previously reviewed continuation host probe records AC, clamshell,
    # ten-minute power-log coverage and absence of Sleep/DarkWake/Thermal.
    import voxtral_fp32_nonbeaver_linearity_continuation_gate_v1 as gate
    try:
        probe = gate._host_probe(require_stable=True)
        raw_log = probe["outputs"]["power_log"].encode()
        replacement_count = 0
    except UnicodeDecodeError:
        # pmset can contain malformed old log bytes. Preserve exact raw bytes;
        # reject malformed *recent* lines and apply the same frozen host rules.
        raw_outputs = {}
        for name, command in gate.HOST_COMMANDS.items():
            completed = subprocess.run(command, capture_output=True, check=True, timeout=30)
            require(bool(completed.stdout), f"Host command returned no evidence: {name}")
            raw_outputs[name] = completed.stdout
        collected = datetime.now(timezone.utc)
        raw_log = raw_outputs["power_log"]
        log = raw_log.decode("utf-8", errors="replace")
        replacement_count = log.count("\ufffd")
        for line in log.splitlines():
            if "\ufffd" not in line:
                continue
            if line.startswith("   pid "):
                # An indented IOKit assertion detail has no event timestamp;
                # it cannot be a Sleep/DarkWake/Thermal event record.
                continue
            try:
                when = datetime.strptime(line[:25], "%Y-%m-%d %H:%M:%S %z")
            except ValueError:
                require(False, "Malformed host power-log line cannot be dated")
            require(when < collected - timedelta(minutes=10),
                    "Malformed recent host power-log evidence")
        require("Now drawing from 'AC Power'" in raw_outputs["power"].decode("utf-8") and
                plistlib.loads(raw_outputs["battery"])[0].get("ExternalConnected") is True and
                plistlib.loads(raw_outputs["clamshell"])[0].get("AppleClamshellState") is False,
                "Host is not on external power with open clamshell")
        gate._power_log_coverage(log, collected - timedelta(minutes=10), collected, collected)
        require(not gate._power_events(log, collected - timedelta(minutes=10), collected),
                "Recent host sleep, DarkWake, or thermal transition")
        probe = {"collected_utc": collected.isoformat(), "commands": gate.HOST_COMMANDS,
                 "outputs": {name: value.decode("utf-8", errors="replace")
                             for name, value in raw_outputs.items()}}
    outputs = probe["outputs"]
    raw = raw_log
    require(len(raw) <= 128 * 1024**2, "Host log exceeds bounded receipt")
    packed = gzip.compress(raw, mtime=0)
    require(len(packed) < 8 * 1024**2, "Host log archive exceeds bounded receipt")
    return {"eligible_at_collection": True, "collected_utc": probe["collected_utc"],
            "commands": probe["commands"],
            "output_sha256": {name: hashlib.sha256(raw if name == "power_log" else value.encode()).hexdigest()
                              for name, value in outputs.items()},
            "power_log_gzip_base64": base64.b64encode(packed).decode(),
            "power_log_gzip_sha256": hashlib.sha256(packed).hexdigest(),
            "power_log_raw_bytes": len(raw),
            "power_log_utf8_replacement_count": replacement_count,
            "short_raw_outputs": {name: value for name, value in outputs.items()
                                  if name != "power_log"},
            "fresh_host_probe_required_at_model_launch": True}


def run(*, check_host: bool = False) -> dict:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    source = verify_static_anchors()
    installed = verify_installed_source()
    snapshot = base.validate_snapshot({"processor_config_sha256": PROCESSOR_CONFIG_SHA,
                                       "tokenizer_asset_sha256": TOKENIZER_SHA})
    from word_disjoint_v2 import candidate_common as common
    cases, fixture = common.fixture_cases()  # Rechecks all 32 WAVs and cross-corpus metadata.
    manifest = json.loads(MANIFEST.read_text())
    inventory = json.loads(INVENTORY.read_text())
    selected, wavs = selected_cases(cases, manifest, inventory, common.load_pcm)
    prepared = prepare_processor(selected)
    resource = resource_estimate(prepared, snapshot)
    geometry = synthetic_zero_geometry(prepared, wavs)
    qa = qa_status(cases)
    host = host_status() if check_host else {"checked": False, "reason": "--check-host not supplied"}
    holds = []
    if not qa["passed"]:
        holds.append("human_qa_32_of_32_independent_go_absent")
    if not resource["can_certify_mps_peak_without_model"]:
        holds.append("fp32_backward_mps_peak_uncertified")
    holds.append("v6_measured_resource_pilot_and_independent_postpilot_go_absent")
    holds.append("independently_reviewed_runner_and_external_admission_freeze_absent")
    if not check_host:
        holds.append("live_host_eligibility_not_checked")
    return {"schema": "voxtral_fp32_track_b_no_weight_preflight_v1",
            "status": "HOLD_no_candidate_model_inference", "utc": datetime.now(timezone.utc).isoformat(),
            "source_sha256": file_sha(Path(__file__)), "anchors": source,
            "installed_source_sha256": installed, "snapshot": snapshot, "fixture": fixture,
            "selected_wavs": wavs, "processor": prepared, "resource": resource,
            "v6_synthetic_zero_geometry": geometry,
            "human_qa": qa, "host": host, "hold_reasons": holds,
            "design_review_status": "corrected_v5_plus_v6_prospective_design_go",
            "candidate_weights_loaded": False, "candidate_forward_count": 0}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-host", action="store_true", help="Collect current AC/lid/power-log evidence")
    parser.add_argument("--output", type=Path, help="Write an exclusive HOLD evidence receipt")
    args = parser.parse_args()
    try:
        receipt = run(check_host=args.check_host)
    except Exception as exc:
        print(json.dumps({"status": "HOLD_preflight_failed", "error": str(exc),
                          "candidate_weights_loaded": False}), file=sys.stderr)
        raise SystemExit(2) from exc
    if args.output:
        with args.output.open("x") as stream:
            json.dump(receipt, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    print(json.dumps({"status": receipt["status"], "hold_reasons": receipt["hold_reasons"],
                      "selected_wavs": len(receipt["selected_wavs"]),
                      "alias_table_sha256": receipt["processor"]["alias_table_sha256"],
                      "v6_zero_pcm_sha256": receipt["v6_synthetic_zero_geometry"]["pcm_sha256"],
                      "v3_planning_total_bytes": receipt["resource"]["v3_alias_batch_sizing"]["planning_total_bytes"],
                      "v5_b1_planning_total_bytes": receipt["resource"]["v5_sequential_b1_sizing"]["planning_total_bytes"]},
                     sort_keys=True))


if __name__ == "__main__":
    main()
