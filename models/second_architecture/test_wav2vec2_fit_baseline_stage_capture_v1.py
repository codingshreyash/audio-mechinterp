"""No-checkpoint tests for the prospective FIT baseline runner."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from transformers import Wav2Vec2Processor


SCRIPT = Path(__file__).with_name("wav2vec2_fit_baseline_stage_capture_v1.py")
spec = importlib.util.spec_from_file_location("wav2vec2_fit_baseline_stage_capture_v1", SCRIPT)
assert spec and spec.loader
runner = importlib.util.module_from_spec(spec)
import sys
sys.modules[spec.name] = runner
spec.loader.exec_module(runner)


def test_trusted_fit_inputs_and_common_normalization() -> None:
    cases, stats = runner.trusted_cases()
    assert len(cases) == 8
    assert {case.pair_id for case in cases} == {"color_05", "object_02"}
    processor = Wav2Vec2Processor.from_pretrained(
        runner.MODEL_ID, revision=runner.MODEL_REVISION, local_files_only=True)
    for pair_id in runner.PAIR_WORDS:
        for engine in runner.ENGINES:
            left, right = [case for case in cases if case.pair_id == pair_id and case.engine == engine]
            pcm_a, pcm_b = runner.load_pcm(left), runner.load_pcm(right)
            pair_stats = stats[(pair_id, engine)]
            a = runner.normalized_input(pcm_a, mode="C", stats=pair_stats)
            b = runner.normalized_input(pcm_b, mode="C", stats=pair_stats)
            start, end = left.slot
            assert np.array_equal(a[:start], b[:start])
            assert np.array_equal(a[end:], b[end:])
            assert not np.array_equal(a[start:end], b[start:end])
            stock = runner.normalized_input(pcm_a, mode="S", stats=pair_stats,
                                            processor=processor)
            assert stock.dtype == np.float32 and stock.shape == a.shape
            assert not np.array_equal(stock[:start], a[:start])


def test_trace_geometry_and_exact_offslot_detection() -> None:
    case = runner.Case("probe", "fake", "one", Path("unused"), "", "", 800, (200, 300))
    x = np.zeros((1, 800), dtype=np.float32)
    y = x.copy()
    y[0, 200:300] = 1
    l2, maxabs, summary = runner.trace_arrays(y, x, "input_values", case)
    assert np.array_equal(l2, maxabs)
    assert summary == {"nominal_window": [200, 300],
                       "first_nonzero_index": 200, "offslot_maxabs": 0.0}
    conv = np.zeros((1, 512, (800 - 10) // 5 + 1), dtype=np.float32)
    leaked = conv.copy()
    leaked[0, 0, 0] = 0.25
    _, _, summary = runner.trace_arrays(leaked, conv, "first_conv_norm", case)
    assert summary["first_nonzero_index"] == 0
    assert summary["offslot_maxabs"] == 0.25
    assert summary["nominal_window"] == [runner.stage_window(case, "first_conv_norm", conv.shape[2])[0],
                                         runner.stage_window(case, "first_conv_norm", conv.shape[2])[1]]


def test_capture_write_cap_is_prechecked_and_preserves_previous_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner, "MAX_OUTPUT_BYTES", 1 << 20)
    monkeypatch.setattr(runner, "MAX_RSS_BYTES", 1 << 50)
    start = __import__("time").monotonic()
    value = np.ones((64,), dtype=np.float32)
    with pytest.raises(RuntimeError, match="would exceed"):
        runner.save_array(tmp_path / "too_big.npy", value, tmp_path, start)
    assert not (tmp_path / "too_big.npy").exists()
    monkeypatch.setattr(runner, "MAX_OUTPUT_BYTES", 1 << 22)
    first = tmp_path / "first.npy"
    digest = runner.save_array(first, value, tmp_path, start)
    assert digest == runner.sha(first)
    with pytest.raises(FileExistsError):
        runner.save_array(first, value, tmp_path, start)
    assert runner.sha(first) == digest


class _ResultModule(nn.Module):
    def forward(self, value: torch.Tensor) -> tuple[torch.Tensor]:
        return (value,)


class _FakeModel(nn.Module):
    def __init__(self, frames: int) -> None:
        super().__init__()
        t1 = (frames - 10) // 5 + 1
        t7 = (frames - 400) // 320 + 1
        self.t1, self.t7 = t1, t7
        self.wav2vec2 = nn.Module()
        self.wav2vec2.feature_extractor = nn.Identity()
        self.wav2vec2.feature_extractor.conv_layers = nn.ModuleList([nn.Module()])
        self.wav2vec2.feature_extractor.conv_layers[0].conv = nn.Identity()
        self.wav2vec2.feature_extractor.conv_layers[0].layer_norm = nn.Identity()
        self.wav2vec2.feature_projection = _ResultModule()
        self.wav2vec2.encoder = nn.Module()
        self.wav2vec2.encoder.layers = nn.ModuleList([_ResultModule() for _ in range(12)])
        self.lm_head = nn.Identity()

    def forward(self, *, input_values: torch.Tensor, **_: object) -> SimpleNamespace:
        conv = torch.ones((1, 512, self.t1), dtype=torch.float32)
        conv = self.wav2vec2.feature_extractor.conv_layers[0].conv(conv)
        conv = self.wav2vec2.feature_extractor.conv_layers[0].layer_norm(conv)
        self.wav2vec2.feature_extractor(torch.ones((1, 512, self.t7), dtype=torch.float32))
        projected = self.wav2vec2.feature_projection(
            torch.ones((1, self.t7, 768), dtype=torch.float32))[0]
        for layer in self.wav2vec2.encoder.layers:
            projected = layer(projected)[0]
        logits = self.lm_head(torch.ones((1, self.t7, 32), dtype=torch.float32))
        return SimpleNamespace(logits=logits)


def test_fake_forward_captures_each_stage_once_without_weights() -> None:
    frames = 800
    model = _FakeModel(frames)
    arrays, counts = runner.model_forward(model, torch.zeros((1, frames)), frames)
    assert set(arrays) == set(runner.STAGES)
    assert counts == dict.fromkeys(runner.STAGES, 1)
    assert arrays["first_conv_raw"].shape == (1, 512, model.t1)
    assert arrays["post_block11"].shape == (1, model.t7, 768)
    assert arrays["logits"].shape == (1, model.t7, 32)
