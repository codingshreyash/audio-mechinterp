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
import shutil
import wave
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable

from models.second_architecture.voicechat_calendar_fixture_v1.calendar_stub import (
    CASES,
    dispatch,
    normalize_official_call,
    verify_external_grant,
)
from models.second_architecture.voicechat_calendar_fixture_v1.audio_qa_v2 import check_audio_qa_v2

NEMO_SPEECH_REVISION = "097dfe9e2f55baf653b83035868bdc89849f1b47"
OFFICIAL_HELPER_SHA256 = "ebde701baec12a014443eac36262a6af7a7a62b3e052f1c4a387db6aaad8da44"
OFFICIAL_TEMPLATE_SHA256 = "8b4f88e9f0052c0dfc19896b9d6a84821f68f84a1049406908e1f664155edcc9"
FIXTURE_DIR = Path(__file__).resolve().parent


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
    template_fields = set(json.loads((FIXTURE_DIR / "qa_receipt_template.json").read_text(encoding="ascii")))
    if set(receipt) != template_fields:
        raise ValueError("Audio QA receipt does not match the complete fixture template")
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
        "exclusion_reason": None,
    }
    for key, expected in required_equal.items():
        if receipt.get(key) != expected:
            raise ValueError(f"Audio QA field {key!r} is missing or mismatched")
    if case_id != "D" and receipt.get("target_date_understood") is not True:
        raise ValueError("Audio QA did not establish the requested calendar date")
    if not isinstance(receipt.get("final_wav_path"), str) or \
            Path(receipt["final_wav_path"]).resolve() != Path(wav_path).resolve():
        raise ValueError("Audio QA path does not match submitted WAV")
    for key in ("voice_id", "source_kind", "raw_render_path", "raw_render_sha256",
                "renderer_version_or_recording_device", "listener_id", "listener_heard_transcript",
                "independent_asr_model", "independent_asr_transcript",
                "qa_completed_at_utc", "external_audit_id", "external_reviewer_id"):
        if not isinstance(receipt.get(key), str) or not receipt[key].strip():
            raise ValueError(f"Audio QA field {key!r} is required")
    if not isinstance(receipt.get("final_sample_count"), int) or \
            receipt["final_sample_count"] <= 80000 or \
            not isinstance(receipt.get("final_exact_zero_tail_frames"), int) or \
            receipt["final_exact_zero_tail_frames"] < 80000:
        raise ValueError("Audio QA sample count or zero tail is missing")
    if case_id in {"B", "C", "B_literal"} and \
            receipt.get("design_review_title_or_id_clear") is not True:
        raise ValueError("Audio QA did not establish the named event")
    if case_id == "C" and receipt.get("confirmation_phrase_heard_exactly") is not True:
        raise ValueError("C audio lacks the independently heard confirmation phrase")


def read_bound_qa_receipt(qa_receipt_path: str | Path, *, case_id: str,
                          wav_path: str | Path, observed_sha256: str,
                          confirmation_state: dict[str, Any] | None,
                          qa_manifest_path: str | Path | None = None,
                          qa_ledger_path: str | Path | None = None,
                          controller_key: bytes | None = None,
                          trusted_controller_id: str | None = None) -> tuple[dict[str, Any], bytes, str]:
    """Hash the exact supplied receipt bytes; reject a misbound C grant."""
    path = Path(qa_receipt_path)
    receipt_bytes = path.read_bytes()
    receipt_sha256 = hashlib.sha256(receipt_bytes).hexdigest()
    try:
        receipt = json.loads(receipt_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid external audio QA receipt: {error}") from error
    if qa_manifest_path is None or qa_ledger_path is None:
        if qa_manifest_path is not None or qa_ledger_path is not None:
            raise ValueError("Both v2 QA manifest and ledger are required")
        validate_qa_receipt(receipt, case_id=case_id, wav_path=wav_path,
                            observed_sha256=observed_sha256)
    else:
        cohort = check_audio_qa_v2(Path(qa_manifest_path), Path(qa_ledger_path),
                                   controller_key=controller_key,
                                   trusted_controller_id=trusted_controller_id)
        if receipt.get("schema") != "voicechat_wav_qa_receipt_v2" or not any(
            row["case_id"] == case_id and row["wav_sha256"] == observed_sha256 and
            row["receipt_sha256"] == receipt_sha256 and
            receipt.get("voice_id") == row["voice_id"]
            for row in cohort["eligibility"]
        ) or Path(receipt.get("wav_path", "")).resolve() != Path(wav_path).resolve():
            raise ValueError("v2 receipt not in frozen, eligible 12-WAV cohort")
        if case_id == "C":
            if sha256_file(qa_manifest_path) != cohort["qa_manifest_sha256"]:
                raise RuntimeError("v2 QA manifest changed during validation")
            manifest = json.loads(Path(qa_manifest_path).read_bytes())
            grant_rows = [row for row in manifest["c_grants"]
                          if row["voice_id"] == receipt["voice_id"]]
            if len(grant_rows) != 1:
                raise ValueError("v2 C grant row is missing")
            grant_file = Path(qa_manifest_path).parent / grant_rows[0]["grant"]["path"]
            if sha256_file(grant_file) != cohort["c_grant_sha256_by_voice"][receipt["voice_id"]] or \
                    confirmation_state != json.loads(grant_file.read_bytes()):
                raise ValueError("C confirmation does not match frozen cohort grant")
        elif confirmation_state is not None:
            raise ValueError("non-C control cannot carry an authorization grant")
    if case_id == "C" and confirmation_state is not None:
        if not isinstance(confirmation_state, dict) or \
                confirmation_state.get("qa_receipt_sha256") != receipt_sha256:
            raise ValueError("Signed C grant does not bind the exact QA receipt bytes")
        if receipt.get("external_controller_id") != confirmation_state.get("controller_id"):
            raise ValueError("Signed C controller differs from QA receipt controller")
        if qa_manifest_path is not None and not verify_external_grant(
            confirmation_state, case_id="C", source_wav_sha256=observed_sha256,
            controller_key=controller_key, trusted_controller_id=trusted_controller_id,
        ):
            raise ValueError("Signed C grant failed verification before inference")
    if path.read_bytes() != receipt_bytes:
        raise RuntimeError("Audio QA receipt changed during validation")
    return receipt, receipt_bytes, receipt_sha256


def load_bound_wav(wav_path: str | Path, qa_receipt_path: str | Path, *,
                   case_id: str, confirmation_state: dict[str, Any] | None,
                   loader: Callable[..., tuple[Any, Any, Any]],
                   device: Any, qa_manifest_path: str | Path | None = None,
                   qa_ledger_path: str | Path | None = None,
                   controller_key: bytes | None = None,
                   trusted_controller_id: str | None = None) -> tuple[Any, Any, Any]:
    """Build the model signal only from the exact externally audited PCM WAV.

    The actual launcher passes the pinned helper's ``load_wav_16k_mono`` as
    ``loader``. A caller cannot supply an unrelated signal tensor here.
    """
    before = sha256_file(wav_path)
    receipt, receipt_bytes, _ = read_bound_qa_receipt(
        qa_receipt_path, case_id=case_id, wav_path=wav_path,
        observed_sha256=before, confirmation_state=confirmation_state,
        qa_manifest_path=qa_manifest_path, qa_ledger_path=qa_ledger_path,
        controller_key=controller_key, trusted_controller_id=trusted_controller_id,
    )
    with wave.open(str(wav_path), "rb") as wav:
        if (wav.getnchannels(), wav.getframerate(), wav.getsampwidth(),
                wav.getcomptype()) != (1, 16000, 2, "NONE"):
            raise ValueError("Submitted WAV is not mono 16-kHz PCM16")
        if wav.getnframes() != receipt["final_sample_count"]:
            raise ValueError("Submitted WAV frame count differs from QA")
        samples = wav.readframes(wav.getnframes())
    zero_tail = 0
    for pos in range(len(samples) - 2, -1, -2):
        if samples[pos:pos + 2] != b"\0\0":
            break
        zero_tail += 1
    if zero_tail != receipt["final_exact_zero_tail_frames"]:
        raise ValueError("Submitted WAV exact zero tail differs from QA")
    wav_1d, input_signal, input_signal_lens = loader(str(wav_path), device=device)
    if sha256_file(wav_path) != before or Path(qa_receipt_path).read_bytes() != receipt_bytes:
        raise RuntimeError("WAV or QA receipt changed while loading the model signal")
    return wav_1d, input_signal, input_signal_lens


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
    qa_receipt_path: str | Path,
    qa_manifest_path: str | Path | None = None,
    qa_ledger_path: str | Path | None = None,
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
    _qa_receipt, qa_receipt_bytes, qa_receipt_sha256 = read_bound_qa_receipt(
        qa_receipt_path, case_id=case_id, wav_path=wav_path,
        observed_sha256=source_wav_sha256, confirmation_state=confirmation_state,
        qa_manifest_path=qa_manifest_path, qa_ledger_path=qa_ledger_path,
        controller_key=controller_key, trusted_controller_id=trusted_controller_id,
    )
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
    if Path(qa_receipt_path).read_bytes() != qa_receipt_bytes:
        raise RuntimeError("Audio QA receipt changed before response selection")

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
    if Path(qa_receipt_path).read_bytes() != qa_receipt_bytes:
        raise RuntimeError("Audio QA receipt changed during two-pass inference")
    return {
        "source": {"nemo_speech_revision": NEMO_SPEECH_REVISION,
                   "case_id": case_id, "wav_path": str(wav_path),
                   "source_wav_sha256": source_wav_sha256,
                   "qa_receipt_path": str(qa_receipt_path),
                   "qa_receipt_sha256": qa_receipt_sha256},
        "qa_receipt_exact_bytes": qa_receipt_bytes,
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
    receipt_path = out / "qa_receipt_exact.json"
    receipt_path.write_bytes(trace["qa_receipt_exact_bytes"])
    if sha256_file(receipt_path) != trace["source"]["qa_receipt_sha256"]:
        raise RuntimeError("Persisted QA receipt differs from signed bytes")
    paths["qa_receipt_exact"] = str(receipt_path)
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


def run_live_case_from_audited_wav(
    model: Any,
    *,
    case_id: str,
    wav_path: str | Path,
    qa_receipt_path: str | Path,
    qa_manifest_path: str | Path,
    qa_ledger_path: str | Path,
    template_path: str | Path,
    output_dir: str | Path,
    device: Any = "cuda",
    confirmation_state: dict[str, Any] | None = None,
    controller_key: bytes | None = None,
    trusted_controller_id: str | None = None,
) -> dict[str, Any]:
    """Future live entry point with input, prompt, and immediate-snapshot gates.

    This function is implemented but has not been invoked with a real model.
    Independent source review and external WAV QA are prerequisites.
    """
    # Run the all-12 v2 gate before any model or NeMo import and bind this case
    # to the exact receipt bytes. A v1 per-WAV receipt cannot enter this path.
    source_sha = sha256_file(wav_path)
    read_bound_qa_receipt(
        qa_receipt_path, case_id=case_id, wav_path=wav_path,
        observed_sha256=source_sha, confirmation_state=confirmation_state,
        qa_manifest_path=qa_manifest_path, qa_ledger_path=qa_ledger_path,
        controller_key=controller_key, trusted_controller_id=trusted_controller_id,
    )
    import torch
    from nemo.collections.speechlm2.inference.utils import offline_voicechat as official

    if sha256_file(official.__file__) != OFFICIAL_HELPER_SHA256:
        raise RuntimeError("Loaded NeMo offline helper differs from pinned source")
    if sha256_file(template_path) != OFFICIAL_TEMPLATE_SHA256:
        raise RuntimeError("Function-call template differs from pinned source")

    # Validate the externally reviewed artifact before the first model call.
    wav_1d, input_signal, input_signal_lens = load_bound_wav(
        wav_path, qa_receipt_path, case_id=case_id,
        confirmation_state=confirmation_state,
        loader=official.load_wav_16k_mono, device=device,
        qa_manifest_path=qa_manifest_path, qa_ledger_path=qa_ledger_path,
        controller_key=controller_key, trusted_controller_id=trusted_controller_id,
    )
    system_bytes = (FIXTURE_DIR / "system_message.txt").read_bytes()
    tools_bytes = (FIXTURE_DIR / "tools.json").read_bytes()
    system_message = system_bytes.decode("ascii")
    tools = json.loads(tools_bytes.decode("ascii"))
    prompt = official.render_fc_system_prompt(str(template_path), system_message, tools)
    prompt_tokens, prompt_token_lens = official.encode_system_prompt(model, prompt, device=device)
    if prompt_tokens is None or prompt_token_lens is None:
        raise RuntimeError("Pinned calendar prompt did not encode")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(wav_path, out / "input_wav_exact.wav")
    if sha256_file(out / "input_wav_exact.wav") != sha256_file(wav_path):
        raise RuntimeError("Archived WAV differs from loaded input path")
    receipt_bytes = Path(qa_receipt_path).read_bytes()
    (out / "qa_receipt_exact.json").write_bytes(receipt_bytes)
    if sha256_file(out / "qa_receipt_exact.json") != hashlib.sha256(receipt_bytes).hexdigest():
        raise RuntimeError("Archived QA receipt bytes differ")
    (out / "rendered_prompt.txt").write_text(prompt, encoding="ascii")
    (out / "system_message_exact.txt").write_bytes(system_bytes)
    (out / "tools_exact.json").write_bytes(tools_bytes)
    torch.save({"prompt_tokens": prompt_tokens, "prompt_token_lens": prompt_token_lens},
               out / "prompt_tokens.pt")

    def snapshot(label: str, result: dict[str, Any], codes: list[Any]) -> None:
        # Called before position parsing/response selection and before pass 2.
        path = out / f"{label}_immediate_raw.pt"
        torch.save({"result": result, "codec_codes_per_step": codes}, path)

    trace = run_two_pass(
        model, case_id=case_id, wav_path=wav_path, qa_receipt_path=qa_receipt_path,
        qa_manifest_path=qa_manifest_path, qa_ledger_path=qa_ledger_path,
        input_signal=input_signal, input_signal_lens=input_signal_lens,
        prompt_tokens=prompt_tokens, prompt_token_lens=prompt_token_lens,
        device=device, confirmation_state=confirmation_state,
        controller_key=controller_key, trusted_controller_id=trusted_controller_id,
        on_pass_complete=snapshot,
    )
    if sha256_file(out / "input_wav_exact.wav") != trace["source"]["source_wav_sha256"] or \
            sha256_file(out / "qa_receipt_exact.json") != trace["source"]["qa_receipt_sha256"]:
        raise RuntimeError("Archived input or receipt differs from modeled evidence")
    trace["source"].update({
        "template_sha256": OFFICIAL_TEMPLATE_SHA256,
        "system_message_sha256": hashlib.sha256(system_bytes).hexdigest(),
        "tools_sha256": hashlib.sha256(tools_bytes).hexdigest(),
        "rendered_prompt_sha256": hashlib.sha256(prompt.encode("ascii")).hexdigest(),
        "input_signal_loaded_from_audited_wav": True,
        "input_wav_samples": int(wav_1d.shape[0]),
    })
    trace["saved_paths"] = save_trace(trace, out / "complete_trace")
    return trace
