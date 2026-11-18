"""Minimal V2 admission shim for the frozen V1 numerical runner."""

from __future__ import annotations

import argparse
import json

import voxtral_fp32_nonbeaver_linearity_continuation_gate_v1 as gate
import voxtral_fp32_nonbeaver_linearity_runner_v1 as runner


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-context", type=int, required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--continuation-freeze-sha256", required=True)
    args = parser.parse_args()
    admission = gate.verify_worker_start(args.run_context, args.attempt_id,
                                         args.continuation_freeze_sha256)
    result = runner.run_context(args.run_context, args.attempt_id,
                                gate.V1_FREEZE_SHA, admitted=admission)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
