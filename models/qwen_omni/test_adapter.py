"""Dependency-free interface checks; no checkpoint or GPU is loaded."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from adapter import (
    ActivationCapture,
    MODEL_ID,
    MODEL_REVISION,
    TRANSFORMERS_REVISION,
    make_audio_conversation,
    prepare_audio_batch,
    resolve_hook_points,
)


ROOT = Path(__file__).resolve().parent


class FakeHandle:
    def __init__(self, module, callback):
        self.module = module
        self.callback = callback

    def remove(self):
        self.module.hooks.remove(self.callback)


class FakeModule:
    def __init__(self):
        self.hooks = []

    def register_forward_hook(self, callback):
        self.hooks.append(callback)
        return FakeHandle(self, callback)

    def fire(self, output):
        for callback in list(self.hooks):
            assert callback(self, (), output) is None
        return output


class FakeTensor:
    def __init__(self, name):
        self.name = name
        self.operations = []

    def detach(self):
        self.operations.append("detach")
        return self

    def cpu(self):
        self.operations.append("cpu")
        return self

    def clone(self):
        self.operations.append("clone")
        return self


def fake_thinker():
    audio = SimpleNamespace(layers=[FakeModule() for _ in range(32)], proj=FakeModule())
    language = SimpleNamespace(
        layers=[FakeModule() for _ in range(36)], norm=FakeModule(), embed_tokens=FakeModule()
    )
    return SimpleNamespace(
        model=SimpleNamespace(audio_tower=audio, language_model=language),
        lm_head=FakeModule(),
    )


class FakeProcessor:
    def __init__(self):
        self.conversation = None
        self.kwargs = None

    def apply_chat_template(self, conversation, **kwargs):
        self.conversation = conversation
        self.kwargs = kwargs
        return {"input_ids": [1, 2, 3], "input_features": "fake-audio-features"}


class AdapterTests(unittest.TestCase):
    def test_pins_match_downloaded_config_snapshot(self):
        pins = json.loads((ROOT / "pins.json").read_text())
        raw = (ROOT / "config.snapshot.json").read_bytes()
        config = json.loads(raw)
        self.assertEqual(hashlib.sha256(raw).hexdigest(), pins["config_snapshot_sha256"])
        self.assertEqual((MODEL_ID, MODEL_REVISION, TRANSFORMERS_REVISION), (
            pins["model_id"], pins["model_revision"], pins["transformers_revision"]
        ))
        thinker = config["thinker_config"]
        self.assertEqual(thinker["audio_config"]["encoder_layers"], 32)
        self.assertEqual(thinker["audio_config"]["output_dim"], 2048)
        self.assertEqual(thinker["text_config"]["num_hidden_layers"], 36)
        self.assertEqual(thinker["text_config"]["hidden_size"], 2048)

    def test_hook_paths_support_thinker_and_full_wrapper(self):
        thinker = fake_thinker()
        direct = resolve_hook_points(thinker, audio_layers=(0, 31), text_layers=(0, 35))
        wrapped = resolve_hook_points(SimpleNamespace(thinker=thinker), audio_layers=(0, 31), text_layers=(0, 35))
        self.assertEqual([p.name for p in direct], [
            "audio.encoder.0", "audio.encoder.31", "audio.to_language",
            "text.residual.0", "text.residual.35",
        ])
        self.assertEqual([p.module for p in direct], [p.module for p in wrapped])

    def test_capture_preserves_forward_output_and_removes_hooks(self):
        thinker = fake_thinker()
        points = resolve_hook_points(thinker, audio_layers=(0,), text_layers=(35,))
        sample = FakeTensor("encoder")
        with ActivationCapture(points) as capture:
            output = (sample,)
            self.assertIs(thinker.model.audio_tower.layers[0].fire(output), output)
            self.assertIs(thinker.model.audio_tower.proj.fire(sample), sample)
            self.assertIs(thinker.model.language_model.layers[35].fire(sample), sample)
        self.assertEqual(sample.operations.count("detach"), 3)
        self.assertEqual([len(p.module.hooks) for p in points], [0, 0, 0])
        self.assertEqual(len(capture.activations["audio.encoder.0"]), 1)

    def test_capture_can_retain_gradient_graph(self):
        points = resolve_hook_points(fake_thinker(), audio_layers=(0,), text_layers=())
        sample = FakeTensor("kept")
        with ActivationCapture(points, detach=False) as capture:
            points[0].module.fire(sample)
        self.assertIs(capture.activations["audio.encoder.0"][0], sample)
        self.assertEqual(sample.operations, [])

    def test_audio_processor_input_and_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.wav"
            path.write_bytes(b"mock-wav-bytes")
            processor = FakeProcessor()
            batch, digest = prepare_audio_batch(
                processor, path, system_prompt="Describe the recording.", user_text="What happened?"
            )
            self.assertEqual(digest, hashlib.sha256(b"mock-wav-bytes").hexdigest())
            self.assertEqual(batch["input_ids"], [1, 2, 3])
            self.assertEqual(processor.conversation[1]["content"][0],
                             {"type": "audio", "audio": str(path.resolve())})
            self.assertEqual(processor.conversation[1]["content"][1],
                             {"type": "text", "text": "What happened?"})
            self.assertEqual(processor.kwargs, {
                "tokenize": True, "add_generation_prompt": True,
                "return_dict": True, "return_tensors": "pt",
            })

    def test_invalid_layout_or_input_fails(self):
        thinker = fake_thinker()
        thinker.model.audio_tower.layers.pop()
        with self.assertRaisesRegex(ValueError, "Model depth disagrees"):
            resolve_hook_points(thinker)
        with self.assertRaises(FileNotFoundError):
            make_audio_conversation("/does/not/exist.wav", system_prompt="test")


if __name__ == "__main__":
    unittest.main()
