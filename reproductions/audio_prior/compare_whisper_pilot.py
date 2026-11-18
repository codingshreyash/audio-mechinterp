"""Compare local Whisper Tiny pilot with the upstream published report.

Reference: audio-jacobian-lens/docs/experiments/WHISPER_TINY_PILOT.md
The output distinguishes exact count matches from directional findings.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
data = json.loads((ROOT / "whisper_tiny_eval_aligned_affine.json").read_text())
metrics = data["metrics"]
asr = data["asr"]

checks = {
    "six_held_out_clips": data["n_examples"] == 6,
    "fifty_generated_positions": metrics["decoder_jlens_L0"]["n"] == 50,
    "published_wer_counts_exact": (asr["word_errors"], asr["reference_words"]) == (5, 47),
    "published_cer_counts_exact": (asr["character_errors"], asr["reference_characters"])
    == (11, 287),
    "published_direct_l0_median_exact": metrics["decoder_direct_L0"]["median_rank"]
    == 7678,
    "published_decoder_l0_top10_exact": metrics["decoder_jlens_L0"]["top_10_rate"]
    == 0.14,
    "decoder_l0_jlens_improves_direct_median_rank": metrics["decoder_jlens_L0"][
        "median_rank"
    ]
    < metrics["decoder_direct_L0"]["median_rank"],
    "negative_aligned_encoder_delta_top10": all(
        metrics[f"encoder_delta_aligned_L{layer}"]["top_10_rate"] == 0
        for layer in range(4)
    ),
    "negative_aligned_encoder_delta_top100_at_most_2pct": max(
        metrics[f"encoder_delta_aligned_L{layer}"]["top_100_rate"]
        for layer in range(4)
    )
    <= 0.02,
    "aligned_and_remote_affine_mrr_near_equal": all(
        abs(
            metrics[f"encoder_affine_aligned_L{layer}"]["mean_reciprocal_rank"]
            - metrics[f"encoder_affine_remote_L{layer}"]["mean_reciprocal_rank"]
        )
        < 0.001
        for layer in range(4)
    ),
}

result = {
    "reference": "audio-jacobian-lens/docs/experiments/WHISPER_TINY_PILOT.md",
    "evidence": "independent local rerun of upstream's published pilot; mixed exact and directional replication",
    "corpus_fingerprint_reported": "46640e2cbca442f599ef",
    "corpus_fingerprint_local": data["estimator"]["encoder"]["corpus_fingerprint"],
    "checks": checks,
    "observed": {
        "asr": asr,
        "decoder_direct_l0_median_rank": metrics["decoder_direct_L0"]["median_rank"],
        "decoder_jlens_l0_median_rank": metrics["decoder_jlens_L0"]["median_rank"],
        "decoder_jlens_l0_top10_rate": metrics["decoder_jlens_L0"]["top_10_rate"],
        "encoder_delta_aligned_top10_rates": [
            metrics[f"encoder_delta_aligned_L{layer}"]["top_10_rate"]
            for layer in range(4)
        ],
        "encoder_delta_aligned_top100_rates": [
            metrics[f"encoder_delta_aligned_L{layer}"]["top_100_rate"]
            for layer in range(4)
        ],
    },
}
output = ROOT / "whisper_replication_check.json"
output.write_text(json.dumps(result, indent=2) + "\n")
print(output)
print(json.dumps(checks, indent=2))
if not all(checks.values()):
    raise SystemExit(1)
