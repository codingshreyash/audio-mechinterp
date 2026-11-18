"""Low-level, source-only VoiceChat stock/adapter identity controls.

This module never loads a checkpoint or authorizes a fixture run. The caller
must first pass the reviewed 12-WAV human-QA gate, pinned-source preflight and
official-sample check, then reset model and RNG state identically for every
arm. A passing comparison is an instrumentation identity result, not a
security finding.
"""

from __future__ import annotations

import copy
import math
from typing import Any

from .two_pass_adapter import _capture_tts_codes


DISCRETE_RESULT_KEYS = (
    "tokens_text", "tokens_function_pred", "tokens_function", "tokens_len",
    "tokens_audio", "src_text", "text", "audio_len",
)
COMMON_INPUT_KEYS = (
    "input_signal", "input_signal_lens", "prompt_tokens", "prompt_token_lens",
)
INJECTION_KEYS = (
    "function_calls", "function_call_lengths", "function_call_steps",
    "function_responses", "function_response_lengths", "function_response_steps",
)
REQUIRED_RESULT_KEYS = (
    "tokens_text", "tokens_function_pred", "tokens_function",
    "tokens_len", "text", "tokens_audio",
)


def _copy(value: Any) -> Any:
    if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "clone"):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _copy(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        copied = [_copy(item) for item in value]
        return tuple(copied) if isinstance(value, tuple) else copied
    return copy.deepcopy(value)


def _plain(value: Any) -> Any:
    if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "tolist"):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _same(left: Any, right: Any) -> bool:
    return _plain(left) == _plain(right)


def _require_same(left: Any, right: Any, label: str) -> None:
    if not _same(left, right):
        raise ValueError(f"VoiceChat parity failed at {label}")


def _has_content(value: Any) -> bool:
    value = _plain(value)
    if value is None:
        return False
    if isinstance(value, (dict, tuple, list, str, bytes)):
        return len(value) > 0
    return True


def capture_stock_helper(
    official: Any, model: Any, *, api_response: dict[str, Any],
    input_signal: Any, input_signal_lens: Any, prompt_tokens: Any,
    prompt_token_lens: Any, device: Any = "cuda", capture_tts: bool = True,
) -> dict[str, Any]:
    """Capture both unmodified official passes and their exact call arguments.

    Only the module's ``run_offline_inference`` reference is wrapped. Its
    original function is called once per pass with unchanged arguments, and
    the reference is restored even on error. Call this serially, not from
    concurrent threads, because the official helper uses a module global.
    """
    original = official.run_offline_inference
    calls: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    codes: list[list[Any]] = []

    def record(*args: Any, **kwargs: Any) -> Any:
        if len(calls) >= 2:
            raise ValueError("Official helper made more than two inference calls")
        if len(args) != 1 or args[0] is not model:
            raise ValueError("Official helper switched model instances")
        calls.append(_copy(kwargs))
        one_pass_codes: list[Any] = []
        if capture_tts:
            with _capture_tts_codes(model, one_pass_codes):
                result = original(*args, **kwargs)
        else:
            result = original(*args, **kwargs)
        if not isinstance(result, dict):
            raise ValueError("Official offline result is not a mapping")
        results.append(_copy(result))
        codes.append(_copy(one_pass_codes))
        return result

    official.run_offline_inference = record
    try:
        returned = official.run_fc_offline_inference(
            model, input_signal=input_signal, input_signal_lens=input_signal_lens,
            prompt_tokens=prompt_tokens, prompt_token_lens=prompt_token_lens,
            api_response=api_response, device=device,
        )
    finally:
        official.run_offline_inference = original
    if len(calls) != 2 or len(results) != 2 or len(codes) != 2:
        raise ValueError("Official helper did not complete exactly two passes")
    if calls[0].get("decode_audio") is not False or calls[1].get("decode_audio") is not True:
        raise ValueError("Official helper pass order changed")
    if not isinstance(returned, tuple) or len(returned) != 5:
        raise ValueError("Official helper return shape changed")
    return {
        "pass1": {"kwargs": calls[0], "result": results[0], "codes": codes[0]},
        "pass2": {"kwargs": calls[1], "result": results[1], "codes": codes[1]},
        "fc_output": _copy(returned[1]),
        "call_step": int(returned[2]),
        "response_step": int(returned[3]),
        "response_token_ids": _copy(returned[4]),
        "api_response": _copy(api_response),
        "tts_wrapped": capture_tts,
    }


def _flat_floats(value: Any) -> list[float]:
    value = _plain(value)
    if isinstance(value, (tuple, list)):
        output: list[float] = []
        for item in value:
            output.extend(_flat_floats(item))
        return output
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError("Raw audio is not numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("Raw audio contains non-finite samples")
    return [number]


def _shape(value: Any) -> tuple[int, ...]:
    if hasattr(value, "shape"):
        return tuple(int(axis) for axis in value.shape)
    if isinstance(value, (tuple, list)):
        if not value:
            return (0,)
        child = _shape(value[0])
        if any(_shape(item) != child for item in value[1:]):
            raise ValueError("Raw audio has ragged shape")
        return (len(value),) + child
    return ()


def _pcm_delta(left: Any, right: Any, label: str) -> float:
    if _shape(left) != _shape(right):
        raise ValueError(f"VoiceChat parity failed at {label} shape")
    a, b = _flat_floats(left), _flat_floats(right)
    if not a or len(a) != len(b):
        raise ValueError(f"VoiceChat parity failed at {label} shape")
    return max(abs(x - y) for x, y in zip(a, b))


def _compare_discrete(left: dict[str, Any], right: dict[str, Any], label: str) -> None:
    for key in DISCRETE_RESULT_KEYS:
        if (key in left) != (key in right):
            raise ValueError(f"VoiceChat parity failed at {label}.{key} presence")
        _require_same(left.get(key), right.get(key), f"{label}.{key}")


def _require_result_evidence(result: dict[str, Any], phase: str, label: str) -> None:
    for key in REQUIRED_RESULT_KEYS:
        if key not in result:
            raise ValueError(f"VoiceChat parity missing {label}.{phase}.{key}")
        if key != "tokens_audio" and not _has_content(result[key]):
            raise ValueError(f"VoiceChat parity missing {label}.{phase}.{key} value")
    if phase == "pass2" and ("audio_len" not in result or result["audio_len"] is None or
                             "audio" not in result or result["audio"] is None):
        raise ValueError(f"VoiceChat parity missing {label}.pass2 raw audio or length")


def _require_injection(kwargs: dict[str, Any], *, replay: bool, label: str) -> None:
    present = {key for key in INJECTION_KEYS if key in kwargs}
    expected = set(INJECTION_KEYS) if replay else set()
    if present != expected:
        raise ValueError(f"VoiceChat parity missing or unexpected {label} injection fields")
    if any(not _has_content(kwargs[key]) for key in present):
        raise ValueError(f"VoiceChat parity missing {label} injection tensor")


def _compare_inputs(left: dict[str, Any], right: dict[str, Any], label: str) -> None:
    for key in COMMON_INPUT_KEYS:
        if key not in left or key not in right:
            raise ValueError(f"VoiceChat parity missing {label}.{key}")
        _require_same(left[key], right[key], f"{label}.{key}")


def _compare_injection(left: dict[str, Any], right: dict[str, Any], label: str) -> None:
    for key in INJECTION_KEYS:
        if (key in left) != (key in right):
            raise ValueError(f"VoiceChat parity failed at {label}.{key} presence")
        if key in left:
            _require_same(left[key], right[key], f"{label}.{key}")


def compare_stock_adapter(
    stock_a: dict[str, Any], stock_b: dict[str, Any],
    stock_no_tts_wrapper: dict[str, Any], adapter_trace: dict[str, Any],
    adapter_calls: list[dict[str, Any]], *, waveform_tolerance: float = 1e-4,
) -> dict[str, Any]:
    """Fail closed on any identity difference before interpreting fixture output.

    ``adapter_calls`` are the exact two kwargs dictionaries observed by a
    passive inference-runner wrapper around ``run_two_pass``. The three stock
    arms and adapter arm must start from identically reset model and RNG state;
    this function cannot establish that external condition.
    """
    if not (0 <= waveform_tolerance <= 1e-4) or not math.isfinite(waveform_tolerance):
        raise ValueError("Waveform tolerance must be finite and no more than 1e-4")
    if len(adapter_calls) != 2:
        raise ValueError("Adapter did not record exactly two pass calls")
    if adapter_calls[0].get("decode_audio") is not False or \
            adapter_calls[1].get("decode_audio") is not True:
        raise ValueError("Adapter pass order changed")
    if not stock_a.get("tts_wrapped") or not stock_b.get("tts_wrapped") or \
            stock_no_tts_wrapper.get("tts_wrapped") is not False:
        raise ValueError("Stock TTS-wrapper controls are missing")
    if adapter_trace.get("external_tool_execution") is not False:
        raise ValueError("Adapter trace must have no external tool execution")

    replay = adapter_trace.get("replay")
    for label, stock in (("stock_a", stock_a), ("stock_b", stock_b),
                         ("stock_no_tts_wrapper", stock_no_tts_wrapper)):
        for phase in ("pass1", "pass2"):
            _require_result_evidence(stock[phase]["result"], phase, label)
            _require_injection(stock[phase]["kwargs"],
                               replay=replay is not None and phase == "pass2",
                               label=f"{label}.{phase}")
    for phase, call in zip(("pass1", "pass2"), adapter_calls):
        _require_result_evidence(adapter_trace[phase]["raw_result"], phase, "adapter")
        _require_injection(call, replay=replay is not None and phase == "pass2",
                           label=f"adapter.{phase}")
    if not all(_has_content(codes) and all(_has_content(code) for code in codes)
               for codes in (stock_a["pass2"]["codes"], stock_b["pass2"]["codes"],
                             adapter_trace["pass2"]["codec_codes_per_step"])):
        raise ValueError("VoiceChat parity missing pass-2 TTS code capture")

    for label, stock in (("stock_b", stock_b), ("stock_no_tts_wrapper", stock_no_tts_wrapper)):
        _require_same(stock_a["fc_output"], stock["fc_output"], f"{label}.fc_output")
        for key in ("call_step", "response_step", "response_token_ids", "api_response"):
            _require_same(stock_a[key], stock[key], f"{label}.{key}")
        for phase in ("pass1", "pass2"):
            _compare_inputs(stock_a[phase]["kwargs"], stock[phase]["kwargs"], f"{label}.{phase}.input")
            _compare_injection(stock_a[phase]["kwargs"], stock[phase]["kwargs"], f"{label}.{phase}.injection")
            _compare_discrete(stock_a[phase]["result"], stock[phase]["result"], f"{label}.{phase}")
            if label == "stock_b":
                _require_same(stock_a[phase]["codes"], stock[phase]["codes"], f"{label}.{phase}.codes")

    for phase, call in zip(("pass1", "pass2"), adapter_calls):
        _compare_inputs(stock_a[phase]["kwargs"], call, f"adapter.{phase}.input")
        _compare_injection(stock_a[phase]["kwargs"], call, f"adapter.{phase}.injection")
        _compare_discrete(stock_a[phase]["result"], adapter_trace[phase]["raw_result"],
                          f"adapter.{phase}")
        _require_same(stock_a[phase]["codes"], adapter_trace[phase]["codec_codes_per_step"],
                      f"adapter.{phase}.codes")

    parsed = adapter_trace["pass1"]["stock_first_parse"]["calls"]
    _require_same(stock_a["fc_output"]["tool_calls"], parsed, "adapter.pass1.parser")
    if replay is None:
        if stock_a["fc_output"]["tool_calls"] or any(
            key in adapter_calls[1] for key in INJECTION_KEYS
        ):
            raise ValueError("No-call replay differs from official helper")
    else:
        if not parsed or not _same(parsed[0], replay["selected_call"]):
            raise ValueError("Adapter selected call differs from official helper")
        _require_same(stock_a["call_step"], replay["call_step"], "adapter.call_step")
        _require_same(stock_a["response_step"], replay["response_step"], "adapter.response_step")
        _require_same(stock_a["response_token_ids"], replay["response_token_ids"],
                      "adapter.response_token_ids")
        _require_same(stock_a["api_response"], {
            "tool_name": replay["response_selector"]["tool_name"],
            "response": replay["response_selector"]["response"],
        }, "adapter.api_response")

    baseline = stock_a["pass2"]["result"].get("audio")
    if baseline is None:
        raise ValueError("Official pass 2 has no unspliced audio")
    repeat_delta = _pcm_delta(baseline, stock_b["pass2"]["result"].get("audio"),
                              "stock repeat audio")
    wrapper_delta = _pcm_delta(baseline, stock_no_tts_wrapper["pass2"]["result"].get("audio"),
                               "no-wrapper audio")
    adapter_delta = _pcm_delta(baseline, adapter_trace["pass2"]["raw_result"].get("audio"),
                               "adapter audio")
    if any(delta > waveform_tolerance for delta in
           (repeat_delta, wrapper_delta, adapter_delta)):
        raise ValueError("Raw unspliced audio exceeds predeclared waveform tolerance")
    return {"status": "source_comparison_passed_pending_live_provenance_review",
            "waveform_tolerance": waveform_tolerance,
            "stock_repeat_max_abs_pcm": repeat_delta,
            "stock_wrapper_max_abs_pcm": wrapper_delta,
            "stock_adapter_max_abs_pcm": adapter_delta,
            "discrete_identity": True, "external_tool_execution": False}
