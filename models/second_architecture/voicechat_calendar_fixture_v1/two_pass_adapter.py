"""Pass-labelled, inert adaptation of pinned NeMo VoiceChat's offline FC helper.

Source: NVIDIA-NeMo/Speech@097dfe9e2f55baf653b83035868bdc89849f1b47,
``nemo/collections/speechlm2/inference/utils/offline_voicechat.py``.
No checkpoint or audio file is loaded when this module is imported. Callers must
complete independent audio QA before invoking ``run_two_pass`` with a model.
"""

from __future__ import annotations

import copy
import hashlib
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable

from models.second_architecture.voicechat_calendar_fixture_v1.calendar_stub import (
    CASES,
    dispatch,
    normalize_official_call,
)

NEMO_SPEECH_REVISION = "097dfe9e2f55baf653b83035868bdc89849f1b47"
OFFICIAL_HELPER_SHA256 = "ebde701baec12a014443eac36262a6af7a7a62b3e052f1c4a387db6aaad8da44"
OFFICIAL_TEMPLATE_SHA256 = "8b4f88e9f0052c0dfc19896b9d6a84821f68f84a1049406908e1f664155edcc9"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_qa_receipt(receipt: dict[str, Any], *, case_id: str,
                        wav_path: str | Path, observed_sha256: str) -> None:
    """Require the separate audio-QA record before any model call.

    This checks consistency and affirmative sign-off, not whether a reviewer
    actually listened or whether ASR is correct. Those are external duties.
    """
    if case_id not in CASES:
        raise ValueError(f"Unknown fixture case: {case_id}")
    if not isinstance(receipt, dict):
        raise ValueError("Independent audio QA receipt is required")
    required_equal = {
        "fixture_version": "voicechat_calendar_fixture_v1",
        "case_id": case_id,
        "script_exact": CASES[case_id]["script"],
        "final_wav_sha256": observed_sha256,
        "sample_rate_hz": 16000,
        "channels": 1,
        "sample_width_bytes": 2,
        "appended_zero_frames": 80000,
        "include_before_inference": True,
        "listener_clear": True,
        "case_semantics_preserved": True,
    }
    for key, expected in required_equal.items():
        if receipt.get(key) != expected:
            raise ValueError(f"Audio QA field {key!r} is missing or mismatched")
    if not isinstance(receipt.get("final_wav_path"), str) or \
            Path(receipt["final_wav_path"]).resolve() != Path(wav_path).resolve():
        raise ValueError("Audio QA path does not match submitted WAV")
    for key in ("voice_id", "listener_id", "listener_heard_transcript",
                "independent_asr_model", "independent_asr_transcript",
                "qa_completed_at_utc", "external_audit_id", "external_reviewer_id"):
        if not isinstance(receipt.get(key), str) or not receipt[key].strip():
            raise ValueError(f"Audio QA field {key!r} is required")
    if case_id == "C" and receipt.get("confirmation_phrase_heard_exactly") is not True:
        raise ValueError("C audio lacks the independently heard confirmation phrase")


def _clean_call_text(call_text: str) -> str:
    # Exactly the stock helper's order of substitutions and splitting.
    clean = call_text.replace("<SPECIAL_20>", "").replace("<SPECIAL_21>", "").strip()
    if "<TOOLCALL>" in clean:
        clean = clean.split("<TOOLCALL>")[1].split("</TOOLCALL>")[0].strip()
    return clean


def _parse_position(call: dict[str, Any], group_index: int, call_index: int) -> dict[str, Any]:
    """Audit every extracted position; replay still follows the first only."""
    raw = call["call_text"]
    clean = _clean_call_text(raw)
    row: dict[str, Any] = {
        "group_index": group_index,
        "call_index": call_index,
        "start_pos": call["start_pos"],
        "end_pos": call["end_pos"],
        "call_text_raw": raw,
        "call_text_clean": clean,
        "parsed_items": [],
        "parse_error": None,
    }
    try:
        calls = json.loads(clean) if clean.startswith("[") else [json.loads(clean)]
        if not isinstance(calls, list):
            raise TypeError("decoded call container is not a list")
        for index, item in enumerate(calls):
            normalized, diagnostic = normalize_official_call(item)
            row["parsed_items"].append({
                "item_index": index,
                "raw_call": item,
                "normalized_call": normalized,
                "normalization": diagnostic,
            })
    except Exception as error:  # The stock helper catches all parser errors.
        row["parse_error"] = f"{type(error).__name__}: {error}"
    return row


def _all_positions(model: Any, result: dict[str, Any]) -> list[dict[str, Any]]:
    func_tokens = result.get("tokens_function_pred", result.get("tokens_function", None))
    if func_tokens is None:
        return []
    extracted = model.stt_model._extract_function_call_positions(
        func_tokens, result.get("tokens_len"), result.get("tokens_text")
    )
    if len(extracted) != 1:
        raise ValueError("This adapter requires exactly one VoiceChat input per run")
    return [
        _parse_position(call, batch_index, call_index)
        for batch_index, info in enumerate(extracted)
        for call_index, call in enumerate(info.get("function_calls", []))
    ]


def _stock_first_parse(first: dict[str, Any] | None, tokenizer: Any) -> dict[str, Any]:
    """Mirror the stock helper's first-position loop, including partial parses."""
    record: dict[str, Any] = {"calls": [], "call_token_ids": [], "parse_error": None}
    if first is None:
        return record
    clean = first["call_text_clean"]
    try:
        calls = json.loads(clean) if clean.startswith("[") else [json.loads(clean)]
        for item in calls:
            name = item.get("name", "")
            args = item.get("arguments", {})
            if isinstance(args, str):
                args = json.loads(args) if args.strip().startswith("{") else {}
            record["calls"].append({
                "name": name,
                "arguments": args,
                "step": first["start_pos"],
                "raw": first["call_text_raw"],
            })
            record["call_token_ids"] = tokenizer.text_to_ids(clean)
    except Exception as error:
        record["parse_error"] = f"{type(error).__name__}: {error}"
    return record


def _clone_code(code: Any) -> Any:
    if hasattr(code, "detach") and hasattr(code, "cpu"):
        return code.detach().cpu().clone()
    return copy.deepcopy(code)


@contextmanager
def _capture_tts_codes(model: Any, collector: list[Any]):
    """Record each generated code without changing the method's return value."""
    tts = getattr(model, "tts_model", None)
    original = getattr(tts, "infer_codes_one_step", None)
    if original is None:
        yield
        return
    had_instance_attribute = "infer_codes_one_step" in vars(tts)
    original_instance_attribute = vars(tts).get("infer_codes_one_step")

    def recording_call(*args: Any, **kwargs: Any) -> Any:
        value = original(*args, **kwargs)
        code = value[0] if isinstance(value, tuple) else value
        collector.append(_clone_code(code))
        return value

    tts.infer_codes_one_step = recording_call
    try:
        yield
    finally:
        if had_instance_attribute:
            tts.infer_codes_one_step = original_instance_attribute
        else:
            delattr(tts, "infer_codes_one_step")


def _default_tensor_builder(values: list, device: Any) -> Any:
    import torch  # Lazy: source-level tests need no PyTorch or NeMo installation.

    return torch.tensor(values, dtype=torch.long, device=device)


def _default_inference_runner(model: Any, **kwargs: Any) -> dict[str, Any]:
    # Import only at execution time, from the pinned NeMo checkout on PYTHONPATH.
    from nemo.collections.speechlm2.inference.utils import offline_voicechat

    if sha256_file(offline_voicechat.__file__) != OFFICIAL_HELPER_SHA256:
        raise RuntimeError("Loaded NeMo offline helper differs from pinned source")

    return offline_voicechat.run_offline_inference(model, **kwargs)


def run_two_pass(
    model: Any,
    *,
    case_id: str,
    wav_path: str | Path,
    qa_receipt: dict[str, Any],
    input_signal: Any,
    input_signal_lens: Any,
    prompt_tokens: Any,
    prompt_token_lens: Any,
    device: Any = "cuda",
    confirmation_state: dict[str, Any] | None = None,
    controller_key: bytes | None = None,
    trusted_controller_id: str | None = None,
    inference_runner: Callable[..., dict[str, Any]] = _default_inference_runner,
    tensor_builder: Callable[[list, Any], Any] = _default_tensor_builder,
    on_pass_complete: Callable[[str, dict[str, Any], list[Any]], None] | None = None,
) -> dict[str, Any]:
    """Run stock-shaped pass 1 and pass 2; execute no external tool.

    ``wav_path`` is hashed here, rather than accepting a caller-supplied hash.
    A future official-sample runner must load ``input_signal`` from this same
    path after external QA; this low-level function cannot prove tensor origin.
    The optional callback can persist each unmodified result before pass 2.
    """
    source_wav_sha256 = sha256_file(wav_path)
    validate_qa_receipt(qa_receipt, case_id=case_id, wav_path=wav_path,
                        observed_sha256=source_wav_sha256)
    if hasattr(input_signal, "shape") and input_signal.shape[0] != 1:
        raise ValueError("This adapter requires batch size one")
    tokenizer = model.stt_model.tokenizer
    common = {
        "input_signal": input_signal,
        "input_signal_lens": input_signal_lens,
        "prompt_tokens": prompt_tokens,
        "prompt_token_lens": prompt_token_lens,
    }

    pass1_codes: list[Any] = []
    with _capture_tts_codes(model, pass1_codes):
        pass1_result = inference_runner(model, **common, decode_audio=False)
    if on_pass_complete is not None:
        on_pass_complete("pass1", pass1_result, pass1_codes)
    positions1 = _all_positions(model, pass1_result)
    first = positions1[0] if positions1 else None
    stock = _stock_first_parse(first, tokenizer)
    selected = stock["calls"][0] if stock["calls"] else None
    replay = None
    fc_kwargs: dict[str, Any] = {}

    if first is not None and selected is not None and stock["call_token_ids"]:
        verdict = dispatch(
            case_id, selected, confirmation_state=confirmation_state,
            source_wav_sha256=source_wav_sha256, controller_key=controller_key,
            trusted_controller_id=trusted_controller_id,
        )
        # The stock helper uses default json.dumps and this exact wrapper.
        response_text = verdict["formatted_response"]
        response_bytes = response_text.encode("utf-8")
        response_ids = tokenizer.text_to_ids(response_text)
        call_step = first["start_pos"]
        response_step = first["end_pos"] + 1
        fc_kwargs = {
            "function_calls": tensor_builder([[stock["call_token_ids"]]], device),
            "function_call_lengths": tensor_builder([[len(stock["call_token_ids"])]], device),
            "function_call_steps": tensor_builder([[call_step]], device),
            "function_responses": tensor_builder([[response_ids]], device),
            "function_response_lengths": tensor_builder([[len(response_ids)]], device),
            "function_response_steps": tensor_builder([[response_step]], device),
        }
        replay = {
            "selected_group_index": first["group_index"],
            "selected_call_index": first["call_index"],
            "selected_parsed_item_index": 0,
            "selected_call": selected,
            "call_step": call_step,
            "call_end_pos": first["end_pos"],
            "call_text_clean": first["call_text_clean"],
            "call_token_ids": stock["call_token_ids"],
            "response_step": response_step,
            "response_selector": verdict,
            "response_text": response_text,
            "response_utf8_hex": response_bytes.hex(),
            "response_token_ids": response_ids,
            "multi_item_call_text": len(stock["calls"]) > 1,
        }

    pass2_codes: list[Any] = []
    with _capture_tts_codes(model, pass2_codes):
        pass2_result = inference_runner(model, **common, decode_audio=True, **fc_kwargs)
    if on_pass_complete is not None:
        on_pass_complete("pass2", pass2_result, pass2_codes)
    positions2 = _all_positions(model, pass2_result)

    counterfactuals = []
    for position in positions2:
        for item in position["parsed_items"]:
            normalized = item["normalized_call"]
            if normalized is None:
                continue
            verdict = dispatch(
                case_id, normalized, confirmation_state=confirmation_state,
                source_wav_sha256=source_wav_sha256, controller_key=controller_key,
                trusted_controller_id=trusted_controller_id,
            )
            counterfactuals.append({
                "position": {"group_index": position["group_index"],
                             "call_index": position["call_index"],
                             "start_pos": position["start_pos"],
                             "end_pos": position["end_pos"]},
                "item_index": item["item_index"],
                "proposal": normalized,
                "counterfactual_response": verdict,
                "replayed": False,
                "executed": False,
            })

    if sha256_file(wav_path) != source_wav_sha256:
        raise RuntimeError("Input WAV changed during two-pass inference")
    return {
        "source": {"nemo_speech_revision": NEMO_SPEECH_REVISION,
                   "case_id": case_id, "wav_path": str(wav_path),
                   "source_wav_sha256": source_wav_sha256},
        "pass1": {"raw_result": pass1_result, "codec_codes_per_step": pass1_codes,
                  "all_extracted_positions": positions1, "stock_first_parse": stock},
        "replay": replay,
        "pass2": {"raw_result": pass2_result, "codec_codes_per_step": pass2_codes,
                  "all_extracted_positions": positions2,
                  "counterfactual_proposals": counterfactuals},
        "external_tool_execution": False,
    }


def save_trace(trace: dict[str, Any], output_dir: str | Path) -> dict[str, str]:
    """Persist raw tensor dictionaries and JSON audit without splicing audio."""
    import torch

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=False)
    paths = {}
    for label in ("pass1", "pass2"):
        path = out / f"{label}_raw.pt"
        torch.save({"result": trace[label]["raw_result"],
                    "codec_codes_per_step": trace[label]["codec_codes_per_step"]}, path)
        paths[label] = str(path)
    audit = {"source": trace["source"], "replay": trace["replay"],
             "pass1": {"all_extracted_positions": trace["pass1"]["all_extracted_positions"],
                       "stock_first_parse": trace["pass1"]["stock_first_parse"]},
             "pass2": {"all_extracted_positions": trace["pass2"]["all_extracted_positions"],
                       "counterfactual_proposals": trace["pass2"]["counterfactual_proposals"]},
             "external_tool_execution": False}
    audit_path = out / "audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=True, default=str) + "\n")
    paths["audit"] = str(audit_path)
    return paths
