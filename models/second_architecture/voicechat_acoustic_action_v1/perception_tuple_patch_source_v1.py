"""Source-only VoiceChat perception-output intervention primitive.

The pinned ``AudioPerceptionModule.forward(return_encoder_emb=True)`` returns
``(projected_frames, encoded_lengths, encoder_frames)``.  This helper operates
on that entire tuple before DuplexSTTModel._init_inference prepends the system
prompt, constructs the audio/ASR timelines, and fuses row zero.  It is not a
live hook, an executable model admission, or evidence of a causal result.
"""

from __future__ import annotations

from typing import Literal

import torch


PerceptionOutput = tuple[torch.Tensor, torch.Tensor, torch.Tensor]
Mode = Literal["read_only", "self", "full_donor", "rows_donor"]


class OneCallPerceptionHook:
    """Fail closed if a hook leaks from pass 1 into the replay pass.

    Attach to the final perception module with ``register_forward_hook`` only
    around one named ``offline_inference`` pass.  The surrounding runner must
    remove it in a ``finally`` block and verify ``calls == 1``.  This class is
    a fake-model wiring primitive; it does not authenticate the model or pass.
    """

    def __init__(self, *, mode: Mode, donor: object | None = None,
                 rows: tuple[int, int] | None = None) -> None:
        self.mode = mode
        self.donor = donor
        self.rows = rows
        self.calls = 0

    def __call__(self, _module: torch.nn.Module, _args: tuple,
                 output: object) -> PerceptionOutput:
        if self.calls != 0:
            raise ValueError("Perception hook leaked into another forward/pass")
        self.calls += 1
        return patch(output, self.donor, mode=self.mode, rows=self.rows)


def _checked(output: object, label: str) -> PerceptionOutput:
    if not isinstance(output, tuple) or len(output) != 3:
        raise ValueError(f"{label}: perception output must be one three-tensor tuple")
    projected, lengths, encoder = output
    if not all(isinstance(item, torch.Tensor) for item in output):
        raise ValueError(f"{label}: every perception output must be a tensor")
    if projected.ndim != 3 or encoder.ndim != 3 or lengths.ndim != 1:
        raise ValueError(f"{label}: projected/encoder/length ranks differ")
    if projected.shape[0] != 1 or encoder.shape[0] != 1 or lengths.shape[0] != 1:
        raise ValueError(f"{label}: this study requires one-sample batches")
    if projected.shape[:2] != encoder.shape[:2]:
        raise ValueError(f"{label}: identity-adapter frame alignment absent")
    if projected.device != encoder.device or projected.device != lengths.device:
        raise ValueError(f"{label}: perception outputs are on different devices")
    if not projected.is_floating_point() or not encoder.is_floating_point():
        raise ValueError(f"{label}: perception embeddings must be floating tensors")
    if lengths.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"{label}: encoded length must be integer")
    if not torch.isfinite(projected).all() or not torch.isfinite(encoder).all():
        raise ValueError(f"{label}: nonfinite perception embedding")
    valid = int(lengths[0].item())
    if valid < 1 or valid > projected.shape[1]:
        raise ValueError(f"{label}: encoded length outside captured frames")
    return projected, lengths, encoder


def capture(output: object) -> PerceptionOutput:
    """Make an exact-dtype/device detached copy without touching model weights."""
    return tuple(item.detach().clone() for item in _checked(output, "capture"))


def patch(
    recipient: object,
    donor: object | None = None,
    *,
    mode: Mode,
    rows: tuple[int, int] | None = None,
) -> PerceptionOutput:
    """Return a complete perception tuple for one pre-generation intervention.

    ``rows`` index preprompt perception frames.  A local intervention changes
    *both* projected frames and the separate encoder-frame route.  The caller
    must freeze an acoustic-to-frame mapping independently before outputs.
    Same-length clean/noise WAVs must have identical encoded length and shape.
    A live wrapper must apply this only to the intended pass and verify full
    token/parser parity; this function cannot establish either property.
    """
    rec = _checked(recipient, "recipient")
    if mode not in ("read_only", "self", "full_donor", "rows_donor"):
        raise ValueError("Unknown perception intervention mode")
    if mode in ("read_only", "self"):
        if donor is not None or rows is not None:
            raise ValueError("Read-only/self arm must not have a donor or rows")
        if mode == "read_only":
            return rec
        return tuple(item.clone() for item in rec)

    if rows is not None and mode != "rows_donor":
        raise ValueError("Full donor arm must not have a row window")
    if donor is None:
        raise ValueError("Donor perception tuple missing")
    src = _checked(donor, "donor")
    for name, left, right in zip(("projected", "lengths", "encoder"), rec, src):
        if left.shape != right.shape or left.dtype != right.dtype or left.device != right.device:
            raise ValueError(f"{name}: donor/recipient geometry or dtype differs")
    if not torch.equal(rec[1], src[1]):
        raise ValueError("Donor/recipient encoded lengths differ")
    if mode == "full_donor":
        return tuple(item.clone() for item in src)

    if (not isinstance(rows, tuple) or len(rows) != 2 or
            any(type(x) is not int for x in rows)):
        raise ValueError("Local donor arm requires exact integer row interval")
    start, end = rows
    if not 0 <= start < end <= int(rec[1][0].item()):
        raise ValueError("Local donor interval outside valid perception frames")
    projected, lengths, encoder = (item.clone() for item in rec)
    projected[:, start:end, :] = src[0][:, start:end, :]
    encoder[:, start:end, :] = src[2][:, start:end, :]
    return projected, lengths, encoder
