"""Gated VoiceChat fixture-arm worker source; live admission is not installed.

The default entry fails before importing NeMo or creating a model. CPU tests
inject only fake admission/model/helper objects and save exact raw captures.
A separately reviewed production admission gate and fresh-process launcher are
required before this source may run VoiceChat weights.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable

from .live_parity_saved_capture_v1 import save_capture
from .live_parity_source_plan_v1 import CASE_ID, VOICE_ID, NO_CALL_BYTES
from .official_adapter_parity import _copy, capture_stock_helper
from .two_pass_adapter import (
    _default_inference_runner, read_bound_qa_receipt,
    run_live_case_from_audited_wav, sha256_file,
)
from .audio_qa_v2 import check_audio_qa_v2

ARMS = ("adapter", "stock_a", "stock_b", "stock_no_tts")


def _require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def _closed_admission(_request: dict[str, Any]) -> dict[str, Any]:
    raise RuntimeError("External QA/anchor/grants and reviewed freeze admission not installed")


def _qa_before_factory(request: dict[str, Any]) -> dict[str, str]:
    """Check the entire signed cohort and selected exact receipt before NeMo."""
    wav_path = Path(request["wav_path"])
    wav_sha = sha256_file(wav_path)
    cohort = check_audio_qa_v2(
        Path(request["qa_manifest_path"]), Path(request["qa_ledger_path"]),
        controller_key=request.get("controller_key"),
        trusted_controller_id=request.get("controller_id"),
    )
    receipt, _, receipt_sha = read_bound_qa_receipt(
        Path(request["qa_receipt_path"]), case_id=CASE_ID, wav_path=wav_path,
        observed_sha256=wav_sha, confirmation_state=None,
        qa_manifest_path=Path(request["qa_manifest_path"]),
        qa_ledger_path=Path(request["qa_ledger_path"]),
        controller_key=request.get("controller_key"),
        trusted_controller_id=request.get("controller_id"),
    )
    _require(receipt.get("voice_id") == VOICE_ID,
             "QA receipt does not bind selected voice")
    return {"wav_sha256": wav_sha, "qa_receipt_sha256": receipt_sha,
            "qa_manifest_sha256": cohort["qa_manifest_sha256"],
            "qa_ledger_sha256": cohort["qa_ledger_sha256"]}


def _official_after_gate() -> Any:
    from nemo.collections.speechlm2.inference.utils import offline_voicechat
    return offline_voicechat


def _model_after_gate(official: Any, checkpoint_dir: Path, device: str) -> Any:
    return official.build_model(str(checkpoint_dir), device=device)


def _recorder_after_gate(model: Any) -> Any:
    from .passive_dual_head_recorder import PassiveDualHeadRecorder
    return PassiveDualHeadRecorder(model)


def _response_from_exact_file(path: Path, *, byte_cap: int) -> tuple[dict[str, Any], str]:
    _require(path.is_file() and not path.is_symlink()
             and path.stat().st_size <= byte_cap,
             "Stock response absent, linked, or over cap")
    with path.open("rb") as stream:
        raw = stream.read(byte_cap + 1)
    _require(len(raw) <= byte_cap and raw.endswith(b"\n")
             and raw.count(b"\n") == 1,
             "Stock response bytes invalid")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Stock response JSON invalid") from error
    _require(isinstance(value, dict) and list(value) == ["tool_name", "response"]
             and isinstance(value["tool_name"], str)
             and isinstance(value["response"], dict)
             and value["response"].get("executed") is False,
             "Stock response is not inert helper-shaped data")
    expected = (json.dumps(value, ensure_ascii=True).encode("ascii") + b"\n")
    _require(raw == expected or raw == NO_CALL_BYTES,
             "Stock response differs from pinned default JSON bytes")
    return value, hashlib.sha256(raw).hexdigest()


def run_fixture_arm(
    request: dict[str, Any], *, arm_dir: Path,
    admission_gate: Callable[[dict[str, Any]], dict[str, Any]] = _closed_admission,
    qa_precheck: Callable[[dict[str, Any]], dict[str, str]] = _qa_before_factory,
    official_factory: Callable[[], Any] = _official_after_gate,
    model_factory: Callable[[Any, Path, str], Any] = _model_after_gate,
    recorder_factory: Callable[[Any], Any] = _recorder_after_gate,
    adapter_run: Callable[..., dict[str, Any]] = run_live_case_from_audited_wav,
    stock_capture: Callable[..., dict[str, Any]] = capture_stock_helper,
) -> dict[str, Any]:
    """Capture one fixture arm after admission, using no live calendar API.

    The source-only test path requires an injected gate with an explicit test
    status. There is no CLI or implicit permission to execute model weights.
    """
    _require(request.get("arm") in ARMS and request.get("case_id") == CASE_ID
             and request.get("voice_id") == VOICE_ID,
             "Fixed fixture arm/case/voice differs")
    _require(type(request.get("seed")) is int and 0 <= request["seed"] < 2**32,
             "Fixture seed invalid")
    _require(type(request.get("capture_byte_cap")) is int
             and request["capture_byte_cap"] > 0,
             "Fixture capture cap invalid")
    _require(arm_dir.is_dir() and not arm_dir.is_symlink()
             and not (arm_dir / "raw_capture").exists()
             and not (arm_dir / "adapter_high_level").exists(),
             "Fixture arm directory is not an exclusive prepared root")
    admission = admission_gate(request)
    _require(admission.get("status") == "synthetic_test_only_no_model_work",
             "Live fixture admission remains closed")
    _require(admission.get("input_hashes") == request.get("input_hashes")
             and admission.get("seed") == request["seed"],
             "Fixture admission does not bind input hashes and seed")
    qa_binding = qa_precheck(request)
    _require(isinstance(qa_binding, dict)
             and all(isinstance(qa_binding.get(key), str) and
                     len(qa_binding[key]) == 64 for key in
                     ("wav_sha256", "qa_receipt_sha256", "qa_manifest_sha256",
                      "qa_ledger_sha256")),
             "Fixture QA precheck did not bind exact cohort and WAV")
    expected_qa_hashes = {"wav_sha256": "selected_wav",
                          "qa_receipt_sha256": "qa_receipt",
                          "qa_manifest_sha256": "qa_manifest",
                          "qa_ledger_sha256": "qa_ledger"}
    _require(isinstance(request.get("input_hashes"), dict) and all(
        request["input_hashes"].get(source_key) == qa_binding[qa_key]
        for qa_key, source_key in expected_qa_hashes.items()),
        "Fixture QA differs from source-plan bound hashes")
    arm = request["arm"]
    api_response: dict[str, Any] | None = None
    response_sha: str | None = None
    if arm != "adapter":
        api_response, response_sha = _response_from_exact_file(
            Path(request["response_path"]),
            byte_cap=request["capture_byte_cap"])
        _require(response_sha == request.get("response_sha256"),
                 "Stock response differs from expected response hash")

    # These calls occur only after the admission callback has returned.
    official = official_factory()
    model = model_factory(official, Path(request["checkpoint_dir"]),
                          request.get("device", "cpu"))
    if arm == "adapter":
        recorder = recorder_factory(model)
        observed_calls: list[dict[str, Any]] = []
        wrapped = recorder.wrap_runner(_default_inference_runner)

        def observe(actual_model: Any, **kwargs: Any) -> dict[str, Any]:
            _require(actual_model is model and len(observed_calls) < 2,
                     "Adapter observer changed model or pass count")
            observed_calls.append(_copy(kwargs))
            return wrapped(actual_model, **kwargs)

        child = arm_dir / "adapter_high_level"
        trace = adapter_run(
            model, case_id=CASE_ID, wav_path=Path(request["wav_path"]),
            qa_receipt_path=Path(request["qa_receipt_path"]),
            qa_manifest_path=Path(request["qa_manifest_path"]),
            qa_ledger_path=Path(request["qa_ledger_path"]),
            template_path=Path(request["template_path"]),
            output_dir=child, device=request.get("device", "cpu"),
            controller_key=request.get("controller_key"),
            trusted_controller_id=request.get("controller_id"),
            inference_runner=observe,
        )
        _require(child.is_dir() and not child.is_symlink()
                 and len(observed_calls) == 2
                 and observed_calls[0].get("decode_audio") is False
                 and observed_calls[1].get("decode_audio") is True,
                 "Adapter high-level output or pass capture incomplete")
        recorder.bind_adapter_trace(trace)
        capture = {"arm": arm, "trace": trace, "calls": observed_calls,
                   "passive_frames": recorder.passes}
        response_sha = None
    else:
        wav_1d, input_signal, input_lens = official.load_wav_16k_mono(
            str(request["wav_path"]), device=request.get("device", "cpu"))
        system_bytes = Path(request["system_message_path"]).read_bytes()
        tools_bytes = Path(request["tools_path"]).read_bytes()
        prompt = official.render_fc_system_prompt(
            str(request["template_path"]), system_bytes.decode("ascii"),
            json.loads(tools_bytes.decode("ascii")))
        prompt_tokens, prompt_lens = official.encode_system_prompt(
            model, prompt, device=request.get("device", "cpu"))
        _require(prompt_tokens is not None and prompt_lens is not None,
                 "Pinned calendar prompt did not encode")
        stock = stock_capture(
            official, model, api_response=api_response,
            input_signal=input_signal, input_signal_lens=input_lens,
            prompt_tokens=prompt_tokens, prompt_token_lens=prompt_lens,
            device=request.get("device", "cpu"),
            capture_tts=arm != "stock_no_tts",
        )
        capture = {"arm": arm, "stock": stock,
                   "loaded_wav_1d": wav_1d,
                   "rendered_prompt": prompt,
                   "system_message_exact": system_bytes,
                   "tools_exact": tools_bytes}
    receipt = save_capture(capture, arm_dir / "raw_capture",
                           byte_cap=request["capture_byte_cap"],
                           trusted_root=arm_dir)
    return {"status": "synthetic_test_only_no_model_work",
            "arm": arm, "process_id": os.getpid(), "seed": request["seed"],
            "input_hashes": admission["input_hashes"],
            "qa_binding": qa_binding,
            "response_sha256": response_sha,
            "capture": receipt,
            "raw_capture_path": str((arm_dir / "raw_capture").resolve())}
