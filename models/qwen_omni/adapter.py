"""Source-pinned hook scaffold for Qwen2.5-Omni-3B's text-only Thinker.

This module deliberately imports neither torch nor transformers. Importing it does
not download weights or allocate a GPU. The caller supplies the pinned Thinker
model and processor after separately installing the pinned dependencies.

Pinned source and checkpoint are recorded in README.md and pins.json.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable


MODEL_ID = "Qwen/Qwen2.5-Omni-3B"
MODEL_REVISION = "f75b40e3da2003cdd6e1829b1f420ca70797c34e"
TRANSFORMERS_REVISION = "e7e568dc7673ac99a87660c7c50e84830268285c"
EXPECTED_AUDIO_LAYERS = 32
EXPECTED_TEXT_LAYERS = 36


@dataclass(frozen=True)
class HookPoint:
    name: str
    module: Any
    meaning: str


def _expect_attr(obj: Any, name: str, path: str) -> Any:
    if not hasattr(obj, name):
        raise ValueError(f"Pinned Qwen2.5-Omni layout missing {path}.{name}")
    return getattr(obj, name)


def _selected_indices(indices: Iterable[int], length: int, label: str) -> tuple[int, ...]:
    selected = tuple(indices)
    if len(set(selected)) != len(selected):
        raise ValueError(f"Duplicate {label} layer index")
    for index in selected:
        if not isinstance(index, int) or not 0 <= index < length:
            raise ValueError(f"{label} layer index {index!r} outside 0..{length - 1}")
    return selected


def resolve_hook_points(
    model: Any,
    *,
    audio_layers: Iterable[int] = (0, 15, 31),
    text_layers: Iterable[int] = (0, 17, 35),
) -> tuple[HookPoint, ...]:
    """Return exact module boundaries for a pinned Thinker or full Omni wrapper.

    The encoder layer hook sees the layer output; the projection hook sees the
    pooled, normalized audio features in language-model width; text layer hooks
    see post-block residuals. No hook changes the forward pass.
    """
    thinker = getattr(model, "thinker", model)
    core = _expect_attr(thinker, "model", "thinker")
    audio = _expect_attr(core, "audio_tower", "thinker.model")
    language = _expect_attr(core, "language_model", "thinker.model")
    audio_blocks = _expect_attr(audio, "layers", "thinker.model.audio_tower")
    text_blocks = _expect_attr(language, "layers", "thinker.model.language_model")
    if len(audio_blocks) != EXPECTED_AUDIO_LAYERS or len(text_blocks) != EXPECTED_TEXT_LAYERS:
        raise ValueError(
            "Model depth disagrees with pinned 3B config: "
            f"audio={len(audio_blocks)}, text={len(text_blocks)}"
        )
    config = getattr(thinker, "config", None)
    if config is not None:
        audio_config = _expect_attr(config, "audio_config", "thinker.config")
        text_config = _expect_attr(config, "text_config", "thinker.config")
        if audio_config.output_dim != 2048 or text_config.hidden_size != 2048:
            raise ValueError("Hidden widths disagree with pinned 3B config")
    projection = _expect_attr(audio, "proj", "thinker.model.audio_tower")
    _expect_attr(language, "norm", "thinker.model.language_model")
    _expect_attr(language, "embed_tokens", "thinker.model.language_model")
    _expect_attr(thinker, "lm_head", "thinker")

    points = [
        HookPoint(
            f"audio.encoder.{i}", audio_blocks[i], "post audio-encoder layer output"
        )
        for i in _selected_indices(audio_layers, len(audio_blocks), "audio")
    ]
    points.append(
        HookPoint(
            "audio.to_language",
            projection,
            "post pooling/normalization projection into text hidden width",
        )
    )
    points.extend(
        HookPoint(
            f"text.residual.{i}", text_blocks[i], "post text-decoder block residual"
        )
        for i in _selected_indices(text_layers, len(text_blocks), "text")
    )
    return tuple(points)


def make_audio_conversation(
    audio_path: str | Path, *, system_prompt: str, user_text: str = ""
) -> list[dict[str, Any]]:
    """Create the processor's audio chat input from an existing local file.

    We leave resampling and special-token expansion to the pinned processor.
    Keep the original bytes and their SHA-256 alongside each experiment result.
    """
    path = Path(audio_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if not system_prompt.strip():
        raise ValueError("system_prompt must be nonempty")
    content: list[dict[str, str]] = [{"type": "audio", "audio": str(path)}]
    if user_text:
        content.append({"type": "text", "text": user_text})
    return [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": content},
    ]


def prepare_audio_batch(
    processor: Any, audio_path: str | Path, *, system_prompt: str, user_text: str = ""
) -> tuple[Any, str]:
    """Ask the pinned processor to load audio and expand its placeholder tokens.

    Returns (processor_batch, input_sha256). The batch can be passed to the
    Thinker forward/generate call after moving tensors to its device. This is
    intentionally not a complete inference function: generation and J-lens
    fitting need different cache, gradient, and decoding settings.
    """
    conversation = make_audio_conversation(
        audio_path, system_prompt=system_prompt, user_text=user_text
    )
    path = Path(audio_path).expanduser().resolve()
    digest = sha256(path.read_bytes()).hexdigest()
    batch = processor.apply_chat_template(
        conversation,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    return batch, digest


def load_pinned_thinker(*, allow_download: bool = False, **model_kwargs: Any) -> tuple[Any, Any]:
    """Load the exact HF revision, only from cache unless explicitly enabled.

    This optional function is for the later full-weight benchmark. It is not
    called by the scaffold tests. Install Transformers at
    ``TRANSFORMERS_REVISION`` before using it.
    """
    try:
        from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration
    except ImportError as exc:
        raise RuntimeError("Install the pinned Transformers source before loading") from exc
    common = {
        "revision": MODEL_REVISION,
        "local_files_only": not allow_download,
    }
    processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_ID, **common)
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        MODEL_ID, **common, **model_kwargs
    )
    model.eval()
    resolve_hook_points(model)
    return model, processor


def _activation(output: Any) -> Any:
    if isinstance(output, (tuple, list)):
        if not output:
            raise ValueError("Hooked module returned an empty sequence")
        output = output[0]
    if hasattr(output, "last_hidden_state"):
        output = output.last_hidden_state
    if output is None:
        raise ValueError("Hooked module returned no activation")
    return output


class ActivationCapture:
    """Context manager for post-module activations from ``resolve_hook_points``.

    With ``detach=False``, returned tensors retain their autograd graph for a
    future lens fitter. Default detached CPU copies are for observation only.
    A hooked module may run several times; values are kept in call order.
    """

    def __init__(self, points: Iterable[HookPoint], *, detach: bool = True):
        self.points = tuple(points)
        self.detach = detach
        self.activations: dict[str, list[Any]] = {p.name: [] for p in self.points}
        self._handles: list[Any] = []

    def __enter__(self) -> ActivationCapture:
        if self._handles:
            raise RuntimeError("ActivationCapture is already active")
        for point in self.points:
            self._handles.append(point.module.register_forward_hook(self._hook(point.name)))
        return self

    def _hook(self, name: str):
        def record(_module: Any, _inputs: Any, output: Any) -> None:
            value = _activation(output)
            if self.detach:
                value = value.detach().cpu().clone()
            self.activations[name].append(value)
            return None

        return record

    def __exit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> None:
        for handle in reversed(self._handles):
            handle.remove()
        self._handles.clear()
