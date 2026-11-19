"""No-weight 88-forward correction to the previously frozen FIT plan."""
from __future__ import annotations

import json
from pathlib import Path

from models.second_architecture import wav2vec2_fit_noise_norm_causal_source_v1 as source

HERE = Path(__file__).resolve().parent
AMENDMENT = HERE / "WAV2VEC2_FIT_NOISE_NORMALIZATION_CAUSAL_88_FORWARD_AMENDMENT_V1.md"


def execution_plan() -> dict:
    plan = dict(source.no_weight_plan())  # validates old code, 20 results, WAV/PCM/noise
    source.require(plan["total_forward_ceiling"] == 80
                   and plan["scored_forward_count"] == 48
                   and len(plan["pairs"]) == 8,
                   "original frozen plan differs")
    unique = {(pair["case"], "clean") for pair in plan["pairs"]}
    unique |= {(pair["case"], pair["noise_arm"]) for pair in plan["pairs"]}
    source.require(len(unique) == 12, "unique stock-waveform count differs")
    plan.update(schema="wav2vec2_fit_noise_norm_execution_plan_v2",
                amendment_sha256=source.sha_file(AMENDMENT),
                stock_waveforms=[list(x) for x in sorted(unique)],
                stage_i_stock_no_hook_count=12,
                stage_i_stock_read_only_count=12,
                total_forward_ceiling=88)
    return plan


if __name__ == "__main__":
    print(json.dumps(execution_plan(), sort_keys=True, indent=2))
