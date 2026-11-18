"""Source-only passive VoiceChat frame recorder; live use requires the QA gate.

Wrap the existing ``run_two_pass`` inference runner, never the policy dispatcher.
The pinned NeMo methods are called exactly once with their original arguments;
hooks only copy tensors to CPU. Real-model identity controls remain mandatory.
"""

from __future__ import annotations

import copy
import hashlib
import inspect
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable


STT_SOURCE_SHA256 = "415e1e01584bab96d9c059d5b0fe9162adaf19677ef43e565c3afad6e073e74b"
VOICECHAT_SOURCE_SHA256 = "fe11dcdde3c31d7144469dab0c6ec9629ff8f7e4dbf881d8d1db258243a67f80"
MAX_FRAMES_PER_PASS = 4096
MAX_CAPTURE_BYTES_PER_PASS = 2 * 1024**3


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_pinned_model_sources(model: Any) -> None:
    """Fail before a model call if the loaded classes differ from pinned NeMo."""
    for label, obj, expected in (
        ("VoiceChat", model, VOICECHAT_SOURCE_SHA256),
        ("DuplexSTT", model.stt_model, STT_SOURCE_SHA256),
    ):
        source = inspect.getsourcefile(type(obj))
        if source is None or _sha(Path(source)) != expected:
            raise RuntimeError(f"{label} source differs from pinned NeMo commit")


def _last_step_copy(value: Any, label: str):
    import torch

    if (not isinstance(value, torch.Tensor) or value.ndim != 3 or
            value.shape[0] != 1 or value.shape[1] < 1):
        raise ValueError(f"{label} must be a batch-one rank-three tensor")
    last = value[:, -1:, :]
    if not torch.isfinite(last).all():
        raise ValueError(f"{label} is not finite")
    return last.detach().cpu().contiguous().clone(), int(value.shape[1])


def _copy(value: Any) -> Any:
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().contiguous().clone()
    if isinstance(value, dict):
        return {key: _copy(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        converted = [_copy(item) for item in value]
        return tuple(converted) if isinstance(value, tuple) else converted
    return copy.deepcopy(value)


def _tensor_bytes(value: Any) -> int:
    import torch

    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(_tensor_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_tensor_bytes(item) for item in value)
    return 0


def _selected_id(state: dict[str, Any], key: str, frame: int) -> int | None:
    value = state.get(key)
    if value is None:
        return None
    if value.ndim != 2 or value.shape[0] != 1 or frame >= value.shape[1]:
        raise ValueError(f"{key} selected-token shape changed")
    return int(value[0, frame].item())


class PassiveDualHeadRecorder:
    """Capture frames from the official two-pass model without changing outputs.

    The default verifier checks exact pinned class-source bytes. Tests inject a
    fake verifier; live code should retain the default and use the existing
    QA-gated adapter for its model calls.
    """

    def __init__(self, model: Any, *,
                 source_verifier: Callable[[Any], None] = verify_pinned_model_sources,
                 max_frames_per_pass: int = MAX_FRAMES_PER_PASS,
                 max_capture_bytes_per_pass: int = MAX_CAPTURE_BYTES_PER_PASS):
        if max_frames_per_pass <= 0 or max_capture_bytes_per_pass <= 0:
            raise ValueError("Recorder bounds must be positive")
        self.model = model
        self.source_verifier = source_verifier
        self.max_frames_per_pass = max_frames_per_pass
        self.max_capture_bytes_per_pass = max_capture_bytes_per_pass
        self.passes: dict[str, dict[str, Any]] = {}
        self._results: dict[str, Any] = {}
        self._bound = False

    def _validate_modules(self) -> None:
        stt = self.model.stt_model
        if not getattr(stt, "use_function_head", False):
            raise ValueError("Pinned VoiceChat function head is required")
        for label in ("llm", "lm_head", "function_head"):
            module = getattr(stt, label, None)
            if module is None or not callable(getattr(module, "register_forward_hook", None)):
                raise ValueError(f"Missing hookable {label}")
        if stt.lm_head is stt.function_head:
            raise ValueError("Text and function heads unexpectedly share one module")
        for label in ("_step_zero", "_step_inference", "_post_inference"):
            if not callable(getattr(stt, label, None)):
                raise ValueError(f"Missing pinned inference method {label}")

    def _charge(self, record: dict[str, Any], value: Any) -> None:
        record["captured_tensor_bytes"] += _tensor_bytes(value)
        if record["captured_tensor_bytes"] > self.max_capture_bytes_per_pass:
            raise MemoryError("Passive recorder tensor budget exceeded")

    @contextmanager
    def _observe_pass(self, record: dict[str, Any]):
        stt = self.model.stt_model
        original_zero = stt._step_zero
        original_step = stt._step_inference
        original_post = stt._post_inference
        tts = self.model.tts_model
        original_tts = tts.infer_codes_one_step
        owned_attributes = {
            (obj, name): (name in vars(obj), vars(obj).get(name))
            for obj, name in ((stt, "_step_zero"), (stt, "_step_inference"),
                              (stt, "_post_inference"), (tts, "infer_codes_one_step"))
        }
        handles = []
        current: dict[str, Any] | None = None

        def put(name: str, value: Any, *, rank_three: bool = True) -> None:
            if current is None or name in current:
                raise ValueError(f"Unexpected or duplicate {name} head call")
            if rank_three:
                copied, prefix_length = _last_step_copy(value, name)
                if "observed_prefix_length" in current and \
                        current["observed_prefix_length"] != prefix_length:
                    raise ValueError("Dual-head prefix lengths differ within frame")
                current["observed_prefix_length"] = prefix_length
            else:
                copied = _copy(value)
            self._charge(record, copied)
            current[name] = copied

        def llm_hook(_module, _args, output):
            hidden = output["last_hidden_state"] if isinstance(output, dict) else output.last_hidden_state
            put("shared_hidden_state", hidden)

        def text_head_hook(_module, _args, output):
            # This copy precedes DuplexSTTModel.forward's in-place text boosts.
            put("raw_lm_head_logits", output)

        def function_head_hook(_module, _args, output):
            put("raw_function_head_logits", output)

        def stt_hook(_module, _args, output):
            if not isinstance(output, dict) or output.get("function_logits") is None:
                raise ValueError("DuplexSTT dual-head return changed")
            put("returned_text_logits", output["text_logits"])
            put("returned_function_logits", output["function_logits"])

        def finish(frame: int, state: dict[str, Any], pre_function_id: int | None,
                   prompt_position: bool) -> None:
            nonlocal current
            required = {"shared_hidden_state", "raw_lm_head_logits",
                        "raw_function_head_logits", "returned_text_logits",
                        "returned_function_logits", "observed_prefix_length"}
            if current is None or set(current) != required:
                raise ValueError("One dual-head frame was not captured completely")
            widths = {int(current[key].shape[-1]) for key in
                      ("raw_lm_head_logits", "raw_function_head_logits",
                       "returned_text_logits", "returned_function_logits")}
            tensor_keys = required - {"observed_prefix_length"}
            if len(widths) != 1 or any(current[key].shape[1] != 1 for key in tensor_keys):
                raise ValueError("Dual-head frame dimensions changed")
            if "use_cache" not in state or not isinstance(state["use_cache"], bool):
                raise ValueError("Pinned cache-mode flag missing")
            expected_prefix = 1 if state["use_cache"] else frame + 1
            if current["observed_prefix_length"] != expected_prefix:
                raise ValueError("Observed head prefix differs from cache mode and frame")
            current["frame_index_untrimmed"] = frame
            current["use_cache"] = state["use_cache"]
            current["selected_text_token_id"] = _selected_id(state, "gen_text", frame)
            current["selected_function_token_id"] = _selected_id(state, "gen_function", frame)
            current["function_token_source"] = (
                "prompt" if prompt_position else
                "injected" if pre_function_id is not None and
                pre_function_id != int(stt.text_pad_id) else "model"
            )
            current["text_token_source"] = "prompt" if prompt_position else "model"
            record["frames"].append(current)
            current = None

        def start(frame: int) -> None:
            nonlocal current
            if (not isinstance(frame, int) or frame != len(record["frames"]) or
                    current is not None or len(record["frames"]) >= self.max_frames_per_pass):
                raise ValueError("Passive recorder frame sequence or cap changed")
            current = {}

        def zero(state: dict[str, Any]):
            start(0)
            before_function = _selected_id(state, "gen_function", 0)
            result = original_zero(state)
            if not isinstance(result, tuple) or len(result) != 2 or result[1] is not state:
                raise ValueError("Pinned step-zero return changed")
            finish(0, state, before_function, bool(state.get("start_gen_pos", 0) > 0))
            return result

        def step(frame: int, state: dict[str, Any], *args: Any, **kwargs: Any):
            start(frame)
            before_function = _selected_id(state, "gen_function", frame)
            prompt_mask = state["is_prompt_position_mask"]
            if prompt_mask.ndim != 2 or prompt_mask.shape[0] != 1:
                raise ValueError("Prompt-position mask shape changed")
            prompt = bool(prompt_mask[0, frame].item())
            result = original_step(frame, state, *args, **kwargs)
            finish(frame, state, before_function, prompt)
            return result

        def post(state: dict[str, Any], *args: Any, **kwargs: Any):
            if "untrimmed_selected" in record:
                raise ValueError("Post-inference called more than once")
            total = state.get("T")
            if (not isinstance(total, int) or total < 1 or
                    len(record["frames"]) != total or
                    len(record["tts_steps"]) != total - 1 or
                    [row["frame_index_untrimmed"] for row in record["tts_steps"]] !=
                    list(range(1, total))):
                raise ValueError("STT/TTS frame count differs from pinned inference length")
            selected = {key: _copy(state.get(key)) for key in
                        ("gen_text", "gen_function", "prompt_token_lens",
                         "is_prompt_position_mask", "function_response_spans")}
            self._charge(record, selected)
            record["untrimmed_selected"] = selected
            return original_post(state, *args, **kwargs)

        def tts_step(*args: Any, **kwargs: Any):
            expected_frame = len(record["tts_steps"]) + 1
            if len(record["frames"]) != expected_frame + 1:
                raise ValueError("TTS code step does not follow its STT frame")
            result = original_tts(*args, **kwargs)
            entry = {"frame_index_untrimmed": expected_frame,
                     "current_subword_id": _copy(kwargs.get("current_subword_id")),
                     "prev_subword_id": _copy(kwargs.get("prev_subword_id")),
                     "code": _copy(result[0] if isinstance(result, tuple) else result)}
            self._charge(record, entry)
            record["tts_steps"].append(entry)
            return result

        try:
            handles.append(stt.llm.register_forward_hook(llm_hook))
            handles.append(stt.lm_head.register_forward_hook(text_head_hook))
            handles.append(stt.function_head.register_forward_hook(function_head_hook))
            handles.append(stt.register_forward_hook(stt_hook))
            stt._step_zero = zero
            stt._step_inference = step
            stt._post_inference = post
            tts.infer_codes_one_step = tts_step
            yield
            if current is not None or not record["frames"] or "untrimmed_selected" not in record:
                raise ValueError("Passive recorder did not observe a complete offline pass")
        finally:
            for handle in handles:
                handle.remove()
            for (obj, name), (existed, original) in owned_attributes.items():
                if existed:
                    setattr(obj, name, original)
                else:
                    delattr(obj, name)

    def wrap_runner(self, inference_runner: Callable[..., dict[str, Any]]):
        """Return a pass-labelled runner for the existing QA-gated adapter."""
        def instrumented(model: Any, **kwargs: Any) -> dict[str, Any]:
            if model is not self.model or self._bound:
                raise ValueError("Recorder model or lifecycle differs")
            expected = "pass1" if not self.passes else "pass2"
            if expected in self.passes or len(self.passes) >= 2 or \
                    kwargs.get("decode_audio") is not (expected == "pass2"):
                raise ValueError("Recorder requires official pass order and decode flags")
            self.source_verifier(model)
            self._validate_modules()
            record: dict[str, Any] = {
                "pass_label": expected, "frames": [], "tts_steps": [],
                "captured_tensor_bytes": 0,
                "source_sha256": {"duplex_stt": STT_SOURCE_SHA256,
                                  "nemotron_voicechat": VOICECHAT_SOURCE_SHA256},
            }
            with self._observe_pass(record):
                result = inference_runner(model, **kwargs)
            if not isinstance(result, dict):
                raise ValueError("Offline inference result is not a mapping")
            channels = {key: _copy(result.get(key)) for key in
                        ("tokens_text", "tokens_function", "tokens_function_pred",
                         "tokens_len", "tokens_audio", "src_text", "text", "audio", "audio_len")}
            self._charge(record, channels)
            record["result_channels_unspliced"] = channels
            self._results[expected] = result
            self.passes[expected] = record
            return result
        return instrumented

    def bind_adapter_trace(self, trace: dict[str, Any]) -> None:
        """Attach parser/replay decisions without reinterpreting raw model tokens."""
        if self._bound or set(self.passes) != {"pass1", "pass2"}:
            raise ValueError("Two complete recorded passes are required")
        for label in ("pass1", "pass2"):
            if trace[label]["raw_result"] is not self._results[label]:
                raise ValueError("Adapter trace does not contain recorded raw result")
        self.passes["pass1"]["parser"] = {
            "all_extracted_positions": _copy(trace["pass1"]["all_extracted_positions"]),
            "stock_first_parse": _copy(trace["pass1"]["stock_first_parse"]),
            "replay": _copy(trace["replay"]),
        }
        self.passes["pass2"]["parser"] = {
            "all_extracted_positions": _copy(trace["pass2"]["all_extracted_positions"]),
            "counterfactual_proposals": _copy(trace["pass2"]["counterfactual_proposals"]),
            "external_tool_execution": trace["external_tool_execution"],
        }
        self._bound = True

    def save(self, output_dir: str | Path) -> dict[str, str]:
        """Save exact CPU copies; never splice speech or silently overwrite."""
        if not self._bound:
            raise ValueError("Adapter parser trace must be bound before saving")
        import torch

        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=False)
        paths = {}
        for label in ("pass1", "pass2"):
            path = root / f"{label}_passive_frames.pt"
            with path.open("xb") as stream:
                torch.save(self.passes[label], stream)
            paths[label] = str(path)
        return paths
