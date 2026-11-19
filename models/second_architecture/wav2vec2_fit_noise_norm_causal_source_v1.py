"""No-weight plan and tensor algebra for the FIT normalization causal study.

This module does not load a model or run candidate inference. Its executable
functions bind exact old FIT stimuli and expose the two frozen 2x2 operations.
See WAV2VEC2_FIT_NOISE_NORMALIZATION_CAUSAL_PROSPECTIVE_V1.md.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
OLD_SOURCE = HERE / "acoustic_fit_noise_pilot_v2.py"
OLD_SOURCE_SHA256 = "72a03647cea026b401a9f9db40ffe3dc29ab90b4ceb1e7a23bee31acc85860cb"
OLD_RECORDS = HERE / "runs/acoustic_fit_noise_parent_v2/wav2vec2_attempt_002/worker_output/records.jsonl"
OLD_RECORDS_SHA256 = "c1badaf305dfaebe9beb9567b73f4e4bfdfab84b45ca8194f4c13c943f60038c"
PROTOCOL = HERE / "WAV2VEC2_FIT_NOISE_NORMALIZATION_CAUSAL_PROSPECTIVE_V1.md"
ENGINES = ("piper", "kokoro")
WORDS = ("silver", "teal")
NOISE_ARMS = ("word_noise", "outside_noise")
SOURCE_ARMS = ("clean", "noisy")
INPUT_CELLS = (("clean", "clean"), ("clean", "noisy"),
               ("noisy", "clean"), ("noisy", "noisy"))
GROUPNORM_CELLS = INPUT_CELLS
SLOTS = {"piper": (30527, 49847), "kokoro": (33920, 57120)}
FIRST_CONV_KERNEL = 10
FIRST_CONV_STRIDE = 5
FIRST_GN_EPSILON = 1e-5
INPUT_EPSILON = 1e-7


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def case_order() -> tuple[tuple[str, str, str], ...]:
    return tuple((engine, word, noise_arm)
                 for engine in ENGINES for word in WORDS for noise_arm in NOISE_ARMS)


def input_statistics(pcm: Any) -> tuple[Any, Any]:
    """Exact feature-extractor mean/variance dtype over all samples."""
    import numpy as np

    require(np.asarray(pcm).dtype == np.dtype("<i2") and np.asarray(pcm).ndim == 1,
            "Expected one exact PCM16 waveform")
    x = np.asarray(pcm).astype(np.float32) / np.float32(32768.0)
    mean = np.float32(x.mean())
    variance = np.float32(x.var())
    require(np.isfinite(mean) and np.isfinite(variance) and variance > 0,
            "Input normalization statistics invalid")
    return mean, variance


def normalize_with_statistics(pcm: Any, mean: Any, variance: Any) -> Any:
    """Apply pinned Wav2Vec2 zero-mean/unit-variance formula."""
    import numpy as np

    require(np.asarray(pcm).dtype == np.dtype("<i2") and np.asarray(pcm).ndim == 1,
            "Expected one exact PCM16 waveform")
    mean, variance = np.float32(mean), np.float32(variance)
    require(np.isfinite(mean) and np.isfinite(variance) and variance > 0,
            "Input normalization statistics invalid")
    x = np.asarray(pcm).astype(np.float32) / np.float32(32768.0)
    normalized = np.asarray((x - mean) / np.sqrt(variance + np.float32(INPUT_EPSILON)),
                            dtype=np.float32)
    require(normalized.shape == x.shape and np.isfinite(normalized).all(),
            "Normalized input invalid")
    return normalized


def common_outside_statistics(clean_pcm: Any, noisy_pcm: Any,
                              slot: tuple[int, int]) -> tuple[Any, Any]:
    """Shared statistics from clean samples outside the exact word slot."""
    import numpy as np

    clean = np.asarray(clean_pcm)
    noisy = np.asarray(noisy_pcm)
    require(clean.dtype == noisy.dtype == np.dtype("<i2") and clean.ndim == noisy.ndim == 1
            and clean.shape == noisy.shape, "Shared-stat PCM shape/dtype differs")
    start, end = slot
    require(0 < start < end < clean.size and np.array_equal(clean[:start], noisy[:start])
            and np.array_equal(clean[end:], noisy[end:]),
            "Word-noise PCM changed outside word slot")
    outside = np.concatenate((clean[:start], clean[end:])).astype(np.float32) / np.float32(32768.0)
    mean, variance = np.float32(outside.mean()), np.float32(outside.var())
    require(np.isfinite(mean) and np.isfinite(variance) and variance > 0,
            "Shared outside statistics invalid")
    return mean, variance


def first_conv_outside_mask(frame_count: int, slot: tuple[int, int]) -> Any:
    """True for first-conv frames whose receptive interval misses slot."""
    import numpy as np

    start, end = slot
    require(type(frame_count) is int and frame_count > 0 and 0 < start < end,
            "First-conv geometry invalid")
    frame = np.arange(frame_count, dtype=np.int64)
    overlap = ((FIRST_CONV_STRIDE * frame < end)
               & (FIRST_CONV_STRIDE * frame + FIRST_CONV_KERNEL - 1 >= start))
    require(bool(overlap.any()) and bool((~overlap).any()),
            "Word/outside conv masks empty")
    return ~overlap


def groupnorm_statistics(raw: Any) -> tuple[Any, Any]:
    """First-GroupNorm per-channel full-time FP32 statistics."""
    import torch

    require(torch.is_tensor(raw) and raw.dtype == torch.float32 and raw.ndim == 3
            and raw.shape[0] == 1 and raw.shape[1] == 512 and raw.shape[2] > 1
            and bool(torch.isfinite(raw).all()), "First raw-conv tensor invalid")
    return raw.mean(dim=-1, keepdim=True), raw.var(dim=-1, unbiased=False, keepdim=True)


def groupnorm_with_statistics(raw: Any, stats: tuple[Any, Any],
                              gamma: Any, beta: Any) -> Any:
    """Apply one selected source's GroupNorm statistics to a raw conv tensor."""
    import torch

    mean, variance = stats
    groupnorm_statistics(raw)  # same validation on all arms
    require(all(torch.is_tensor(x) and x.dtype == torch.float32 and x.device == raw.device
                for x in (mean, variance, gamma, beta)),
            "GroupNorm statistic/affine dtype or device differs")
    require(mean.shape == variance.shape == (1, 512, 1)
            and gamma.numel() == beta.numel() == 512
            and bool(torch.isfinite(mean).all()) and bool(torch.isfinite(variance).all())
            and bool(torch.isfinite(gamma).all()) and bool(torch.isfinite(beta).all())
            and bool((variance >= 0).all()), "GroupNorm statistic/affine shape invalid")
    affine_shape = (1, 512, 1)
    output = ((raw - mean) * torch.rsqrt(variance + FIRST_GN_EPSILON)
              * gamma.reshape(affine_shape) + beta.reshape(affine_shape))
    require(output.shape == raw.shape and bool(torch.isfinite(output).all()),
            "GroupNorm counterfactual output invalid")
    return output


def no_weight_plan() -> dict[str, Any]:
    """Bind saved FIT inputs/outcomes and enumerate all cells; no weights."""
    import numpy as np
    from models.second_architecture import acoustic_fit_noise_pilot_v2 as old

    require(np.__version__ == "2.3.5", "Frozen NumPy version differs")
    require(sha_file(OLD_SOURCE) == OLD_SOURCE_SHA256
            and sha_file(OLD_RECORDS) == OLD_RECORDS_SHA256,
            "Prior frozen source/result bytes differ")
    clips = old.stimuli()
    records = [json.loads(line) for line in OLD_RECORDS.read_text().splitlines()]
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for row in records:
        if row.get("type") == "case" and row.get("model") == "wav2vec2":
            key = (row.get("case"), row.get("arm"))
            require(key not in lookup, "Prior case/arm duplicated")
            lookup[key] = row
    pairs = []
    for engine, word, noise_arm in case_order():
        case = f"{engine}_{word}"
        require(set(clips[case]) == set(old.ARMS), "Prior arm matrix incomplete")
        clean, noisy = clips[case]["clean"], clips[case][noise_arm]
        require(clean.shape == noisy.shape and clean.dtype == noisy.dtype == np.dtype("<i2"),
                "Frozen pair PCM differs")
        saved = []
        for arm, pcm in (("clean", clean), (noise_arm, noisy)):
            item = lookup.get((case, arm))
            require(item is not None and item.get("pcm_sha256") == sha_bytes(pcm.tobytes())
                    and isinstance(item.get("output", {}).get("argmax_ids"), list)
                    and isinstance(item.get("output", {}).get("transcript"), str),
                    "Prior stock output or PCM missing")
            saved.append({"arm": arm, "pcm_sha256": item["pcm_sha256"],
                          "argmax_ids_sha256": sha_bytes(json.dumps(item["output"]["argmax_ids"],
                                                            separators=(",", ":")).encode()),
                          "transcript": item["output"]["transcript"]})
        pairs.append({"case": case, "engine": engine, "word": word,
                      "noise_arm": noise_arm, "slot": list(SLOTS[engine]),
                      "prior_stock": saved})
    require(len(pairs) == 8 and len(lookup) == 20, "Prior evidence coverage differs")
    return {"schema": "wav2vec2_fit_noise_norm_no_weight_plan_v1",
            "status": "source_plan_only_no_candidate_inference",
            "protocol_sha256": sha_file(PROTOCOL),
            "prior_source_sha256": OLD_SOURCE_SHA256,
            "prior_records_sha256": OLD_RECORDS_SHA256,
            "pairs": pairs, "stage_i_cells": [list(x) for x in INPUT_CELLS],
            "stage_ii_cases": [p["case"] for p in pairs if p["noise_arm"] == "word_noise"],
            "stage_ii_cells": [list(x) for x in GROUPNORM_CELLS],
            "scored_forward_count": 8 * len(INPUT_CELLS) + 4 * len(GROUPNORM_CELLS),
            "stage_i_stock_no_hook_count": 8,
            "stage_i_stock_read_only_count": 8,
            "stage_ii_shared_no_hook_count": 8,
            "stage_ii_shared_read_only_capture_count": 8,
            "total_forward_ceiling": 80,
            "weight_loads": 0}


if __name__ == "__main__":
    print(json.dumps(no_weight_plan(), sort_keys=True, indent=2))
