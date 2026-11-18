"""Bounded, FIT-only Whisper encoder spread-versus-use causal assay.

See WHISPER_FIT_SPREAD_CAUSAL_PROSPECTIVE_V1.md. The `preflight` command
does not load model weights. `run` requires a byte-identical preflight receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import resource
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "eval/audio_jlens/corpus_v1/rendered/manifest.json"
MANIFEST_SHA = "adf8cd02d24a43ef576bcb993dc54bd2527440a62798e12fc788def3c063971f"
ASR_SCREEN = ROOT / "eval/audio_jlens/corpus_v1/asr_audit.json"
ASR_SCREEN_SHA = "41da6ee4dd67a2f7b06ece24354fdb0ca6911f7017ec514e170f55f61d38a5e7"
MODEL_ID = "openai/whisper-tiny.en"
MODEL_REVISION = "87c7102498dcde7456f24cfd30239ca606ed9063"
WEIGHT_SHA = "db59695928ded6043adaef491a53ef4e12da9611184d77c53baa691a60b958ad"
PAIR_SPECS = {
    "color_05": {
        "words": ("silver", "teal"),
        "piper": {"slot": (30527, 49847), "W": (90, 161), "P": (10, 81), "Q": (171, 242), "N": 272},
        "kokoro": {"slot": (33920, 57120), "W": (101, 184), "P": (10, 93), "Q": (194, 277), "N": 287},
    },
    "object_02": {
        "words": ("plate", "cup"),
        "piper": {"slot": (30527, 42788), "W": (90, 139), "P": (10, 59), "Q": (149, 198), "N": 250},
        "kokoro": {"slot": (33920, 54320), "W": (101, 175), "P": (10, 84), "Q": (185, 259), "N": 279},
    },
}
ENGINES = ("piper", "kokoro")
SITES = ("post_block0", "post_norm")
CAPTURE_SITES = ("post_conv_pos", "post_block0", "post_block1", "post_block2", "post_block3", "post_norm")
BASE_SEEDS = (20261118, 20261119)
MAX_SECONDS = 1800
MAX_RSS_BYTES = 4 * 1024**3


def derive_arm_seed(pair_id: str, engine: str, recipient_word: str, donor_word: str,
                    site: str, base_seed: int) -> int:
    if (pair_id not in PAIR_SPECS or engine not in ENGINES or site not in SITES
            or base_seed not in BASE_SEEDS):
        raise ValueError("unfrozen random arm identity")
    left, right = PAIR_SPECS[pair_id]["words"]
    if (recipient_word, donor_word) not in ((left, right), (right, left)):
        raise ValueError("unfrozen random word direction")
    source = f"whisper-fit-spread-v1|{pair_id}|{engine}|{recipient_word}|{donor_word}|{site}|{base_seed}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(source).digest()[:8], "big") & ((1 << 63) - 1)


ALL_ARM_SEEDS = {
    derive_arm_seed(pair_id, engine, recipient, donor, site, base)
    for pair_id, pair_spec in PAIR_SPECS.items()
    for engine in ENGINES
    for recipient, donor in ((pair_spec["words"][0], pair_spec["words"][1]),
                             (pair_spec["words"][1], pair_spec["words"][0]))
    for site in SITES
    for base in BASE_SEEDS
}
if len(ALL_ARM_SEEDS) != 32:
    raise RuntimeError("derived random-arm seeds collide")


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True)
class Case:
    pair_id: str
    engine: str
    word: str
    path: Path
    file_sha256: str
    pcm_sha256: str
    frames: int
    slot: tuple[int, int]
    window: tuple[int, int]
    prefix: tuple[int, int]
    postword: tuple[int, int]
    active_rows: int

    @property
    def key(self) -> str:
        return f"{self.pair_id}_{self.engine}_{self.word}"


def region_spans(case: Case, condition: str) -> tuple[tuple[int, int], ...]:
    if condition in ("word", "self"):
        return (case.window,)
    if condition == "prefix":
        return (case.prefix,)
    if condition == "postword":
        return (case.postword,)
    if condition in ("complement", "random"):
        return ((0, case.window[0]), (case.window[1], case.active_rows))
    if condition == "full_donor":
        return ((0, 1500),)
    raise ValueError("unknown frozen region")


def expected_arm_matrix() -> set[tuple[str, str, str, str, str, str, int | None]]:
    matrix = set()
    for pair_id, spec in PAIR_SPECS.items():
        left, right = spec["words"]
        for engine in ENGINES:
            for recipient, donor in ((left, right), (right, left)):
                for site in SITES:
                    for condition in ("word", "prefix", "postword", "complement", "self"):
                        matrix.add((pair_id, engine, recipient, donor, site, condition, None))
                    for base_seed in BASE_SEEDS:
                        matrix.add((pair_id, engine, recipient, donor, site, "random", base_seed))
                matrix.add((pair_id, engine, recipient, donor, "post_norm", "full_donor", None))
    if len(matrix) != 120:
        raise RuntimeError("planned spread arm matrix is not 120 unique interventions")
    return matrix


def check_case_manifest() -> tuple[list[Case], dict[str, Any]]:
    import numpy as np
    import soundfile as sf

    if file_sha(MANIFEST) != MANIFEST_SHA:
        raise ValueError("render manifest hash mismatch")
    if file_sha(ASR_SCREEN) != ASR_SCREEN_SHA:
        raise ValueError("prior FIT ASR screen hash mismatch")
    manifest = json.loads(MANIFEST.read_text())
    asr_screen = json.loads(ASR_SCREEN.read_text())
    prior = [r for r in asr_screen["records"] if r["pair_id"] in PAIR_SPECS and r["phrasing_id"] == "fit_chosen"]
    expected_cases = {(p, e, w) for p, spec in PAIR_SPECS.items() for e in ENGINES for w in spec["words"]}
    if len(prior) != 8 or {(r["pair_id"], r["engine"], r["word"]) for r in prior} != expected_cases:
        raise ValueError("prior ASR eligibility screen has incomplete pair coverage")
    if any(r["normalized_exact"] is not True or r["target_word_exactly_once"] is not True or r["other_word_count"] != 0 for r in prior):
        raise ValueError("prior ASR screen no longer supports frozen pair")
    selected = [p for p in manifest["pairs"] if p["pair_id"] in PAIR_SPECS and p["phrasing_id"] == "fit_chosen"]
    if len(selected) != 4 or {(p["pair_id"], p["engine"]) for p in selected} != {(p, e) for p in PAIR_SPECS for e in ENGINES}:
        raise ValueError("expected exactly two pairs in Piper and Kokoro")
    cases: list[Case] = []
    evidence: dict[str, Any] = {pair_id: {} for pair_id in PAIR_SPECS}
    for pair in selected:
        pair_id = pair["pair_id"]
        engine = pair["engine"]
        spec = PAIR_SPECS[pair_id]
        pair_spec = spec[engine]
        words = spec["words"]
        if pair["split"] != "fit_pool" or pair["map_fit"] is not True or pair["words"] != list(words):
            raise ValueError("pair is not the frozen FIT word contrast")
        if pair["outside_slot_pcm_identical"] is not True or pair["inside_slot_pcm_differs"] is not True:
            raise ValueError("paired-PCM assertions absent")
        slot = (pair["word_slot_start_sample"], pair["word_slot_end_sample_exclusive"])
        if slot != pair_spec["slot"]:
            raise ValueError("frozen sample slot changed")
        start, end = slot
        calculated = (math.floor(start / 16000 * 50) - 5, math.ceil(end / 16000 * 50) + 5)
        if calculated != pair_spec["W"]:
            raise ValueError("frozen encoder window changed")
        width = calculated[1] - calculated[0]
        prefix = (10, 10 + width)
        postword = (calculated[1] + 10, calculated[1] + 10 + width)
        if prefix != pair_spec["P"] or postword != pair_spec["Q"]:
            raise ValueError("frozen equal-width control windows changed")
        if not (prefix[1] <= calculated[0] < calculated[1] <= postword[0]
                and postword[1] <= pair_spec["N"]):
            raise ValueError("control regions overlap or exceed active audio")
        pair_pcm = {}
        for word in words:
            output = pair["outputs"][word]
            prior_record = next(r for r in prior if r["pair_id"] == pair_id and r["engine"] == engine and r["word"] == word)
            if prior_record["audio_sha256"] != output["file_sha256"] or prior_record["pcm_int16_sha256"] != output["pcm_int16_sha256"]:
                raise ValueError("prior ASR record does not match frozen WAV")
            path = ROOT / output["path"]
            if not path.is_file() or file_sha(path) != output["file_sha256"]:
                raise ValueError(f"WAV file hash mismatch: {path}")
            pcm, rate = sf.read(path, dtype="int16", always_2d=False)
            if rate != 16000 or pcm.ndim != 1 or len(pcm) != output["frames"]:
                raise ValueError(f"WAV format mismatch: {path}")
            pcm = np.asarray(pcm, dtype="<i2")
            pcm_hash = hashlib.sha256(pcm.tobytes()).hexdigest()
            if pcm_hash != output["pcm_int16_sha256"]:
                raise ValueError(f"WAV PCM hash mismatch: {path}")
            active_rows = math.ceil(len(pcm) / 320)
            if active_rows != pair_spec["N"]:
                raise ValueError("frozen active-audio row count changed")
            pair_pcm[word] = pcm
            cases.append(Case(pair_id, engine, word, path, output["file_sha256"], pcm_hash,
                              len(pcm), slot, calculated, prefix, postword, active_rows))
        a, b = (pair_pcm[word] for word in words)
        if len(a) != len(b) or not np.array_equal(a[:start], b[:start]) or not np.array_equal(a[end:], b[end:]):
            raise ValueError(f"outside-slot PCM differs for {engine}")
        if np.array_equal(a[start:end], b[start:end]):
            raise ValueError(f"inside-slot PCM does not differ for {engine}")
        evidence[pair_id][engine] = {
            "slot_samples": list(slot), "W": list(calculated),
            "P": list(prefix), "Q": list(postword), "N": active_rows,
            "C": [list(span) for span in region_spans(cases[-1], "complement")],
        }
    return sorted(cases, key=lambda c: (c.pair_id, c.engine, c.word)), evidence


def preflight() -> dict[str, Any]:
    import soundfile
    import torch
    import transformers
    from huggingface_hub import snapshot_download
    from transformers.models.whisper import feature_extraction_whisper, generation_whisper, modeling_whisper, processing_whisper, tokenization_whisper

    cases, regions = check_case_manifest()
    snapshot = Path(snapshot_download(MODEL_ID, revision=MODEL_REVISION, local_files_only=True))
    weights = snapshot / "model.safetensors"
    if not weights.is_file() or file_sha(weights) != WEIGHT_SHA:
        raise ValueError("Whisper checkpoint hash mismatch")
    config = json.loads((snapshot / "config.json").read_text())
    shape = {k: config[k] for k in ("num_mel_bins", "max_source_positions", "d_model", "encoder_layers")}
    if shape != {"num_mel_bins": 80, "max_source_positions": 1500, "d_model": 384, "encoder_layers": 4}:
        raise ValueError("Whisper config shape mismatch")
    source_modules = [modeling_whisper, generation_whisper, feature_extraction_whisper, processing_whisper, tokenization_whisper]
    source_hashes = {module.__name__: file_sha(Path(inspect.getfile(module))) for module in source_modules}
    script_relative = str(Path(__file__).resolve().relative_to(ROOT))
    dirty = subprocess.run(["git", "diff", "--quiet", "--", script_relative], cwd=ROOT, check=False)
    staged_dirty = subprocess.run(["git", "diff", "--cached", "--quiet", "--", script_relative], cwd=ROOT, check=False)
    if dirty.returncode != 0 or staged_dirty.returncode != 0:
        raise ValueError("runner source has uncommitted changes")
    source_commit = subprocess.check_output(
        ["git", "log", "-1", "--format=%H", "--", script_relative], cwd=ROOT, text=True
    ).strip()
    if len(source_commit) != 40:
        raise ValueError("runner has no committed source revision")
    asset_names = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "generation_config.json", "preprocessor_config.json")
    asset_hashes = {}
    for name in asset_names:
        asset = snapshot / name
        if not asset.is_file():
            raise ValueError(f"missing cached Whisper asset: {name}")
        asset_hashes[name] = file_sha(asset)
    snapshot_hashes = {path.name: file_sha(path) for path in sorted(snapshot.iterdir()) if path.is_file()}
    if "model.safetensors" not in snapshot_hashes or snapshot_hashes["model.safetensors"] != WEIGHT_SHA:
        raise ValueError("Whisper snapshot file inventory mismatch")
    return {
        "schema": "whisper_fit_spread_causal_preflight_v1",
        "script_sha256": file_sha(Path(__file__)),
        "script_source_git_commit": source_commit,
        "manifest_sha256": MANIFEST_SHA,
        "prior_asr_screen_sha256": ASR_SCREEN_SHA,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "checkpoint_sha256": WEIGHT_SHA,
        "config_sha256": file_sha(snapshot / "config.json"),
        "asset_sha256": asset_hashes,
        "snapshot_files_sha256": snapshot_hashes,
        "source_hashes": source_hashes,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "soundfile_version": soundfile.__version__,
        "python_version": sys.version.split()[0],
        "device": "cpu",
        "threads": 4,
        "generation": {"do_sample": False, "num_beams": 1, "max_new_tokens": 128, "output_scores": True, "return_dict_in_generate": True},
        "capture_sites": list(CAPTURE_SITES),
        "intervention_sites": list(SITES),
        "random_arm_seeds": [
            {"pair_id": pair_id, "engine": engine, "recipient_word": recipient,
             "donor_word": donor, "site": site, "base_seed": base,
             "arm_seed": derive_arm_seed(pair_id, engine, recipient, donor, site, base)}
            for pair_id, spec in PAIR_SPECS.items()
            for engine in ENGINES
            for recipient, donor in ((spec["words"][0], spec["words"][1]),
                                     (spec["words"][1], spec["words"][0]))
            for site in SITES
            for base in BASE_SEEDS
        ],
        "cases": [{"key": c.key, "file_sha256": c.file_sha256, "pcm_sha256": c.pcm_sha256, "frames": c.frames} for c in cases],
        "regions": regions,
    }


def validate_state(state: Any, expected: Any | None = None) -> None:
    import torch

    if not isinstance(state, torch.Tensor) or state.shape != (1, 1500, 384):
        raise ValueError(f"encoder state must be [1,1500,384], got {getattr(state, 'shape', None)}")
    if state.device.type != "cpu" or state.dtype != torch.float32 or not torch.isfinite(state).all():
        raise ValueError("encoder state is not finite CPU float32")
    if expected is not None and (state.device != expected.device or state.dtype != expected.dtype or state.shape != expected.shape):
        raise ValueError("donor/recipient state mismatch")


def patch_rows(recipient: Any, donor: Any, spans: tuple[tuple[int, int], ...],
               mode: str, seed: int | None = None, audit: dict[str, Any] | None = None) -> Any:
    import torch

    validate_state(recipient)
    validate_state(donor, recipient)
    if not spans or any(not (0 <= start < end <= 1500) for start, end in spans):
        raise ValueError("invalid encoder region spans")
    if any(left[1] > right[0] for left, right in zip(spans, spans[1:])):
        raise ValueError("overlapping or unordered region spans")
    region_width = sum(end - start for start, end in spans)
    result = recipient.clone()
    if mode in {"donor", "self", "full_donor"}:
        if mode == "full_donor":
            if spans != ((0, 1500),):
                raise ValueError("full donor requires all rows")
            result[:] = donor
        else:
            for start, end in spans:
                result[:, start:end, :] = donor[:, start:end, :]
    elif mode == "random":
        if seed not in ALL_ARM_SEEDS:
            raise ValueError("unfrozen random seed")
        difference = torch.cat([donor[:, start:end, :] - recipient[:, start:end, :]
                                for start, end in spans], dim=1)
        target = torch.linalg.vector_norm(difference)
        if not torch.isfinite(target) or target <= 0:
            raise ValueError("random target Frobenius norm is zero or nonfinite")
        generator = torch.Generator(device="cpu").manual_seed(seed)
        noise = torch.randn((1, region_width, 384), generator=generator, dtype=torch.float32)
        norm = torch.linalg.vector_norm(noise)
        if not torch.isfinite(norm) or norm <= 0:
            raise ValueError("zero or nonfinite random norm")
        noise *= target / norm
        offset = 0
        for start, end in spans:
            width = end - start
            result[:, start:end, :] += noise[:, offset:offset + width, :]
            offset += width
        realized = torch.linalg.vector_norm(torch.cat(
            [result[:, start:end, :] - recipient[:, start:end, :] for start, end in spans], dim=1))
        if not torch.allclose(realized, target, rtol=1e-5, atol=1e-6):
            raise ValueError("random edit Frobenius norm mismatch")
        if audit is not None:
            audit.update({
                "random_edit_sha256": hashlib.sha256(noise.numpy().tobytes()).hexdigest(),
                "random_edit_norm": float(realized),
                "random_target_norm": float(target),
            })
    else:
        raise ValueError("unknown intervention mode")
    validate_state(result)
    selected = torch.zeros(1500, dtype=torch.bool)
    for start, end in spans:
        selected[start:end] = True
    if not torch.equal(result[:, ~selected, :], recipient[:, ~selected, :]):
        raise ValueError("patch changed rows outside selected region")
    return result


def site_module(model: Any, site: str) -> tuple[Any, bool]:
    encoder = model.model.encoder
    if site == "post_block0":
        return encoder.layers[0], False
    if site == "post_norm":
        return encoder.layer_norm, False
    raise ValueError(f"unsupported intervention site: {site}")


def install_capture_hooks(model: Any, captures: dict[str, Any], counts: dict[str, int]) -> list[Any]:
    handles = []
    encoder = model.model.encoder

    def pre_hook(_module: Any, args: tuple[Any, ...]) -> None:
        key = "post_conv_pos"
        counts[key] += 1
        if counts[key] != 1:
            raise ValueError("capture hook fired more than once")
        validate_state(args[0])
        captures[key] = args[0].detach().clone()

    handles.append(encoder.layers[0].register_forward_pre_hook(pre_hook))
    for index in range(4):
        key = f"post_block{index}"

        def post_hook(_module: Any, _args: tuple[Any, ...], output: Any, key: str = key) -> None:
            counts[key] += 1
            if counts[key] != 1:
                raise ValueError("capture hook fired more than once")
            validate_state(output)
            captures[key] = output.detach().clone()

        handles.append(encoder.layers[index].register_forward_hook(post_hook))

    def norm_hook(_module: Any, _args: tuple[Any, ...], output: Any) -> None:
        counts["post_norm"] += 1
        if counts["post_norm"] != 1:
            raise ValueError("capture hook fired more than once")
        validate_state(output)
        captures["post_norm"] = output.detach().clone()

    handles.append(encoder.layer_norm.register_forward_hook(norm_hook))
    return handles


def install_patch_hook(model: Any, site: str, donor: Any,
                       spans: tuple[tuple[int, int], ...], mode: str, seed: int | None,
                       count: list[int], audit: dict[str, Any] | None = None) -> Any:
    module, is_pre = site_module(model, site)

    if is_pre:
        def pre_hook(_module: Any, args: tuple[Any, ...]) -> tuple[Any, ...]:
            count[0] += 1
            if count[0] != 1:
                raise ValueError("patch hook fired more than once")
            return (patch_rows(args[0], donor, spans, mode, seed, audit), *args[1:])
        return module.register_forward_pre_hook(pre_hook)

    def post_hook(_module: Any, _args: tuple[Any, ...], output: Any) -> Any:
        count[0] += 1
        if count[0] != 1:
            raise ValueError("patch hook fired more than once")
        return patch_rows(output, donor, spans, mode, seed, audit)
    return module.register_forward_hook(post_hook)


def normalized_words(text: str) -> list[str]:
    import re

    return re.findall(r"[^\W_]+(?:['’][^\W_]+)?", text.casefold())


def transcript_outcome(transcript: str, recipient: str, donor: str, baseline: str) -> dict[str, Any]:
    words = normalized_words(transcript)
    base = normalized_words(baseline)
    expected = [donor if word == recipient else word for word in base]

    def distance(a: list[str], b: list[str]) -> int:
        row = list(range(len(b) + 1))
        for index, token in enumerate(a, 1):
            next_row = [index]
            for column, other in enumerate(b, 1):
                next_row.append(min(next_row[-1] + 1, row[column] + 1, row[column - 1] + (token != other)))
            row = next_row
        return row[-1]

    return {
        "target_replacement": words == expected and base.count(recipient) == 1 and donor not in base,
        "donor_once_recipient_absent": words.count(donor) == 1 and words.count(recipient) == 0,
        "recipient_transcript_match": words == base,
        "wer_to_recipient": distance(words, base) / max(len(base), 1),
        "wer_to_single_word_donor_counterfactual": distance(words, expected) / max(len(expected), 1),
        "token_difference_count": None,  # filled from generated token IDs in the runner
        "normalized_words": words,
    }


def current_rss_bytes() -> int:
    # macOS ru_maxrss is bytes; this study is local to the Mac host.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def run_model(model: Any, processor: Any, batch: Any, *, pair_id: str,
              hook: Any = None, hook_count: list[int] | None = None) -> dict[str, Any]:
    import torch

    start = time.monotonic()
    try:
        with torch.inference_mode():
            generation = model.generate(
                **batch, do_sample=False, num_beams=1, max_new_tokens=128,
                return_dict_in_generate=True, output_scores=True,
            )
    finally:
        if hook is not None:
            hook.remove()
    ids = generation.sequences[0].detach().cpu().tolist()
    transcript = processor.batch_decode(generation.sequences, skip_special_tokens=True)[0].strip()
    eos_id = model.generation_config.eos_token_id
    if isinstance(eos_id, list):
        eos_ids = set(eos_id)
    else:
        eos_ids = {eos_id}
    step_scores = []
    words = PAIR_SPECS[pair_id]["words"]
    alias_tokens = {word: processor.tokenizer.encode(" " + word, add_special_tokens=False) for word in words}

    def finite_or_none(value: Any) -> float | None:
        number = float(value)
        return number if math.isfinite(number) else None

    for step, score in enumerate(generation.scores):
        token = ids[len(ids) - len(generation.scores) + step]
        values = score[0].detach().cpu()
        top = torch.topk(values, k=2)
        margin = None
        margin_status = "multi_token_alias"
        if all(len(alias_tokens[word]) == 1 for word in words):
            margin = finite_or_none(values[alias_tokens[words[0]][0]] - values[alias_tokens[words[1]][0]])
            margin_status = "finite" if margin is not None else "nonfinite_suppressed_alias"
        step_scores.append({
            "token_id": token, "selected_score": finite_or_none(values[token]),
            "top2_ids": top.indices.tolist(), "top2_scores": [finite_or_none(value) for value in top.values],
            f"generation_processed_{words[0]}_minus_{words[1]}_first_token_score": margin,
            "alias_margin_status": margin_status,
        })
    eos_seen = any(token in eos_ids for token in ids)
    eos_terminated = bool(ids) and ids[-1] in eos_ids and sum(token in eos_ids for token in ids) == 1
    return {
        "token_ids": ids,
        "transcript": transcript,
        "eos_seen": eos_seen,
        "eos_terminated": eos_terminated,
        "censored": not eos_terminated,
        "step_scores": step_scores,
        "elapsed_seconds": time.monotonic() - start,
        "rss_bytes": current_rss_bytes(),
        "hook_count": hook_count[0] if hook_count is not None else None,
    }


def append_record(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(canonical(record) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def execute(expected_preflight: Path, output: Path) -> None:
    import numpy as np
    import soundfile as sf
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    current = preflight()  # validates identities and source before weight load
    expected = json.loads(expected_preflight.read_text())
    if canonical(expected) != canonical(current):
        raise ValueError("reviewed preflight receipt differs from current environment or source")
    if output.exists():
        raise FileExistsError("output directory already exists; preserve prior attempt")
    output.mkdir(parents=True)
    append_record(output / "records.jsonl", {
        "type": "preflight", "receipt": current,
        "expected_receipt_sha256": file_sha(expected_preflight),
    })
    start = time.monotonic()

    def check_budget() -> None:
        if time.monotonic() - start > MAX_SECONDS or current_rss_bytes() > MAX_RSS_BYTES:
            raise RuntimeError("time or resident-memory hard stop")

    try:
        torch.set_num_threads(4)
        processor = WhisperProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION, local_files_only=True)
        model = WhisperForConditionalGeneration.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION, local_files_only=True).to("cpu", dtype=torch.float32).eval()
        if any(p.device.type != "cpu" or p.dtype != torch.float32 for p in model.parameters()):
            raise ValueError("model is not entirely CPU float32")
        cases, regions = check_case_manifest()
        by_key = {case.key: case for case in cases}
        baselines: dict[str, dict[str, Any]] = {}
        states: dict[str, dict[str, Any]] = {}
        batches: dict[str, Any] = {}
        capture_hashes: dict[str, dict[str, str]] = {}
        capture_dir = output / "captures"
        capture_dir.mkdir()
        for case in cases:
            check_budget()
            pcm, rate = sf.read(case.path, dtype="int16")
            samples = np.asarray(pcm, dtype=np.float32) / 32768.0
            batch = processor(samples, sampling_rate=rate, return_tensors="pt", return_attention_mask=True)
            if tuple(batch["input_features"].shape) != (1, 80, 3000):
                raise ValueError("processor mel tensor shape changed")
            batches[case.key] = batch
            captures: dict[str, Any] = {}
            counts = {site: 0 for site in CAPTURE_SITES}
            handles = install_capture_hooks(model, captures, counts)
            try:
                result = run_model(model, processor, batch, pair_id=case.pair_id)
            finally:
                for handle in handles:
                    handle.remove()
            capture_complete = all(count == 1 for count in counts.values()) and set(captures) == set(CAPTURE_SITES)
            words = PAIR_SPECS[case.pair_id]["words"]
            other_word = words[1 - words.index(case.word)]
            normalized = normalized_words(result["transcript"])
            eligible = (normalized.count(case.word) == 1 and other_word not in normalized
                        and not result["censored"])
            states[case.key] = captures
            baselines[case.key] = result
            hashes = {}
            mel_path = capture_dir / f"{case.key}_mel.npy"
            np.save(mel_path, batch["input_features"].numpy(), allow_pickle=False)
            hashes["mel"] = file_sha(mel_path)
            for site, tensor in captures.items():
                tensor_path = capture_dir / f"{case.key}_{site}.npy"
                np.save(tensor_path, tensor.numpy(), allow_pickle=False)
                hashes[site] = file_sha(tensor_path)
            capture_hashes[case.key] = hashes
            append_record(output / "records.jsonl", {
                "type": "baseline", "pair_id": case.pair_id, "engine": case.engine,
                "case": case.key, "result": result, "capture_sha256": hashes,
                "eligible": eligible, "capture_complete": capture_complete,
                "capture_hook_counts": counts,
            })
            if not capture_complete:
                raise ValueError(f"capture hooks did not fire exactly once: {case.key}: {counts}")
            if not eligible:
                raise ValueError(f"baseline target-word eligibility failed: {case.key}")
            check_budget()

        for pair_id, pair_spec in PAIR_SPECS.items():
            left_word, right_word = pair_spec["words"]
            for engine in ENGINES:
                left = f"{pair_id}_{engine}_{left_word}"
                right = f"{pair_id}_{engine}_{right_word}"
                case = by_key[left]
                trace_regions = regions[pair_id][engine]
                traces = {}
                for site in ("mel", *CAPTURE_SITES):
                    if site == "mel":
                        difference = batches[left]["input_features"] - batches[right]["input_features"]
                        delta = torch.linalg.vector_norm(difference, dim=1)[0].numpy()
                        maxabs = torch.amax(torch.abs(difference), dim=1)[0].numpy()
                        factor = 2
                    else:
                        difference = states[left][site] - states[right][site]
                        delta = torch.linalg.vector_norm(difference, dim=-1)[0].numpy()
                        maxabs = torch.amax(torch.abs(difference), dim=-1)[0].numpy()
                        factor = 1
                    W = tuple(bound * factor for bound in case.window)
                    P = tuple(bound * factor for bound in case.prefix)
                    Q = tuple(bound * factor for bound in case.postword)
                    C = tuple((a * factor, b * factor) for a, b in region_spans(case, "complement"))
                    active_rows = case.active_rows * factor
                    trace_prefix = f"{pair_id}_{engine}_{site}_{left_word}_{right_word}"
                    trace_path = capture_dir / f"{trace_prefix}_l2.npy"
                    maxabs_path = capture_dir / f"{trace_prefix}_maxabs.npy"
                    np.save(trace_path, delta, allow_pickle=False)
                    np.save(maxabs_path, maxabs, allow_pickle=False)
                    complement_values = np.concatenate([delta[a:b] for a, b in C])
                    traces[site] = {
                        "path": str(trace_path.relative_to(output)), "sha256": file_sha(trace_path),
                        "maxabs_path": str(maxabs_path.relative_to(output)),
                        "maxabs_sha256": file_sha(maxabs_path), "shape": list(delta.shape),
                        "word_window": list(W), "prefix_window": list(P),
                        "postword_window": list(Q), "active_rows": active_rows,
                        "complement_spans": [list(span) for span in C],
                        "word_l2_sum": float(delta[slice(*W)].sum()),
                        "prefix_l2_sum": float(delta[slice(*P)].sum()),
                        "postword_l2_sum": float(delta[slice(*Q)].sum()),
                        "complement_l2_sum": float(complement_values.sum()),
                        "global_max_abs_difference": float(maxabs.max()),
                        "first_nonzero_frame": int(np.flatnonzero(delta > 0)[0]) if np.any(delta > 0) else None,
                    }
                pre_left = states[left]["post_conv_pos"]
                pre_right = states[right]["post_conv_pos"]
                C = region_spans(case, "complement")
                outside_equal = all(torch.equal(pre_left[:, a:b, :], pre_right[:, a:b, :]) for a, b in C)
                word_differs = not torch.equal(pre_left[:, case.window[0]:case.window[1], :],
                                               pre_right[:, case.window[0]:case.window[1], :])
                input_locality_pass = outside_equal and word_differs
                append_record(output / "records.jsonl", {
                    "type": "trace", "pair_id": pair_id, "engine": engine,
                    "words": list(pair_spec["words"]), "regions": trace_regions,
                    "traces": traces, "input_locality_pass": input_locality_pass,
                    "outside_preconv_equal": outside_equal, "word_preconv_differs": word_differs,
                })
                if not input_locality_pass:
                    raise ValueError(f"input-locality gate failed: {pair_id}/{engine}")
                check_budget()

        random_edit_hashes: set[str] = set()
        observed_arms: set[tuple[str, str, str, str, str, str, int | None]] = set()
        for pair_id, pair_spec in PAIR_SPECS.items():
            left, right = pair_spec["words"]
            for engine in ENGINES:
                for recipient_word, donor_word in ((left, right), (right, left)):
                    recipient = f"{pair_id}_{engine}_{recipient_word}"
                    donor = f"{pair_id}_{engine}_{donor_word}"
                    case = by_key[recipient]
                    for site in SITES:
                        arms = [(name, None) for name in ("word", "prefix", "postword", "complement", "self")]
                        arms.extend(("random", seed) for seed in BASE_SEEDS)
                        for condition, base_seed in arms:
                            check_budget()
                            spans = region_spans(case, condition)
                            source_case = recipient if condition == "self" else donor
                            source = states[source_case][site]
                            mode = "self" if condition == "self" else ("random" if condition == "random" else "donor")
                            arm_seed = (derive_arm_seed(pair_id, engine, recipient_word, donor_word, site, base_seed)
                                        if condition == "random" else None)
                            count = [0]
                            patch_audit: dict[str, Any] = {}
                            handle = install_patch_hook(model, site, source, spans, mode, arm_seed, count, patch_audit)
                            result = run_model(model, processor, batches[recipient], pair_id=pair_id,
                                               hook=handle, hook_count=count)
                            control_pass = (count[0] == 1 and result["token_ids"] == baselines[recipient]["token_ids"]
                                            if condition == "self" else None)
                            outcome = transcript_outcome(result["transcript"], recipient_word, donor_word,
                                                         baselines[recipient]["transcript"])
                            outcome["target_replacement"] = outcome["target_replacement"] and not result["censored"]
                            outcome["token_difference_count"] = (
                                sum(a != b for a, b in zip(result["token_ids"], baselines[recipient]["token_ids"]))
                                + abs(len(result["token_ids"]) - len(baselines[recipient]["token_ids"])))
                            append_record(output / "records.jsonl", {
                                "type": "intervention", "pair_id": pair_id, "engine": engine,
                                "recipient": recipient, "donor": donor, "site": site,
                                "condition": condition, "region_spans": [list(span) for span in spans],
                                "base_seed": base_seed, "arm_seed": arm_seed, "patch_audit": patch_audit,
                                "result": result, "hook_count": count[0], "valid_hook": count[0] == 1,
                                "state_shape": [1, 1500, 384], "state_dtype": "float32",
                                "finite_checked": True,
                                "recipient_state_sha256": capture_hashes[recipient][site],
                                "source_state_sha256": capture_hashes[source_case][site],
                                "control_pass": control_pass, "outcome": outcome,
                            })
                            arm_key = (pair_id, engine, recipient_word, donor_word, site, condition, base_seed)
                            if arm_key in observed_arms:
                                raise ValueError("duplicate spread intervention arm")
                            observed_arms.add(arm_key)
                            if count[0] != 1:
                                raise ValueError(f"patch hook fired {count[0]} times: {recipient}/{site}/{condition}")
                            if control_pass is False:
                                raise ValueError(f"self patch changed tokens: {recipient}/{site}")
                            if condition == "random":
                                edit_hash = patch_audit.get("random_edit_sha256")
                                if not isinstance(edit_hash, str) or edit_hash in random_edit_hashes:
                                    raise ValueError("random edit tensor hash missing or reused")
                                random_edit_hashes.add(edit_hash)
                            check_budget()
                    check_budget()
                    spans = region_spans(case, "full_donor")
                    count = [0]
                    handle = install_patch_hook(model, "post_norm", states[donor]["post_norm"],
                                                spans, "full_donor", None, count)
                    result = run_model(model, processor, batches[recipient], pair_id=pair_id,
                                       hook=handle, hook_count=count)
                    control_pass = count[0] == 1 and result["token_ids"] == baselines[donor]["token_ids"]
                    outcome = transcript_outcome(result["transcript"], recipient_word, donor_word,
                                                 baselines[recipient]["transcript"])
                    outcome["target_replacement"] = outcome["target_replacement"] and not result["censored"]
                    outcome["token_difference_count"] = (
                        sum(a != b for a, b in zip(result["token_ids"], baselines[recipient]["token_ids"]))
                        + abs(len(result["token_ids"]) - len(baselines[recipient]["token_ids"])))
                    append_record(output / "records.jsonl", {
                        "type": "intervention", "pair_id": pair_id, "engine": engine,
                        "recipient": recipient, "donor": donor, "site": "post_norm",
                        "condition": "full_donor", "region_spans": [list(span) for span in spans],
                        "base_seed": None, "arm_seed": None, "patch_audit": {},
                        "result": result, "hook_count": count[0], "valid_hook": count[0] == 1,
                        "state_shape": [1, 1500, 384], "state_dtype": "float32", "finite_checked": True,
                        "recipient_state_sha256": capture_hashes[recipient]["post_norm"],
                        "source_state_sha256": capture_hashes[donor]["post_norm"],
                        "control_pass": control_pass, "outcome": outcome,
                    })
                    arm_key = (pair_id, engine, recipient_word, donor_word, "post_norm", "full_donor", None)
                    if arm_key in observed_arms:
                        raise ValueError("duplicate full-donor arm")
                    observed_arms.add(arm_key)
                    if count[0] != 1:
                        raise ValueError(f"patch hook fired {count[0]} times: {recipient}/post_norm/full_donor")
                    if not control_pass:
                        raise ValueError(f"full-donor implementation control failed: {recipient}")
                    check_budget()
        if len(random_edit_hashes) != 32:
            raise ValueError("random edit tensor coverage is incomplete")
        if observed_arms != expected_arm_matrix():
            raise ValueError("128-decode matrix has missing or extra intervention arms")
        check_budget()
        append_record(output / "records.jsonl", {
            "type": "complete", "elapsed_seconds": time.monotonic() - start,
            "rss_bytes": current_rss_bytes(),
        })
    except BaseException as exc:
        append_record(output / "records.jsonl", {
            "type": "failure", "error_type": type(exc).__name__,
            "error": str(exc), "elapsed_seconds": time.monotonic() - start,
        })
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    pre = sub.add_parser("preflight")
    pre.add_argument("--output", type=Path, required=True)
    run = sub.add_parser("run")
    run.add_argument("--expected-preflight", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "preflight":
        receipt = preflight()
        if args.output.exists():
            raise FileExistsError("preflight receipt already exists")
        args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        print(args.output)
    else:
        execute(args.expected_preflight, args.output)


if __name__ == "__main__":
    main()
