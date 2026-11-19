"""Candidate VoiceChat public-sample step-400 diagnostic child.

This is a constituent worker, not an executable GO or a replacement for an
independently reviewed parent, host freeze, and one-attempt admission. It
loads the pinned official public sample and stops during the first pass before
the official helper can parse a tool call or begin pass 2. It saves no model
outputs, token values, audio, prompt, or tensor values.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import stat
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

from models.second_architecture import voicechat_h100_local_preflight as pinned
from models.second_architecture.voicechat_calendar_fixture_v1 import public_sample_smoke_v1 as sample
from models.second_architecture.voicechat_calendar_fixture_v1 import voicechat_step400_probe_core_v1 as probe


SCHEMA = "voicechat_step400_worker_completion_v1"
MAX_COMPLETION_BYTES = 32_768
HEARTBEAT_ENV = "VOICECHAT_DIAG_HEARTBEAT_FD"
EXPECTED_FILES = {"stdout.log", "stderr.log", "gpu_poll.jsonl"}


def need(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def canonical(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       allow_nan=False) + "\n").encode("utf-8")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_once(path: Path, data: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


class Heartbeat:
    """Cooperative liveness signal; the independent parent enforces the cap."""

    def __init__(self, fd: int) -> None:
        need(fd >= 3 and stat.S_ISFIFO(os.fstat(fd).st_mode),
             "inherited heartbeat pipe missing")
        self.fd = fd
        self.stop = threading.Event()
        self.error: BaseException | None = None
        self.thread: threading.Thread | None = None

    def _send(self) -> None:
        try:
            while not self.stop.is_set():
                os.write(self.fd, b"\x01")
                self.stop.wait(0.5)
        except BaseException as error:
            self.error = error

    def __enter__(self) -> "Heartbeat":
        # Establish parent visibility before any hash or model work starts.
        os.write(self.fd, b"\x01")
        self.thread = threading.Thread(target=self._send, name="diag-heartbeat", daemon=True)
        self.thread.start()
        return self

    def check(self) -> None:
        need(self.error is None and self.thread is not None and self.thread.is_alive(),
             "heartbeat failed")

    def __exit__(self, _type: Any, _value: Any, _trace: Any) -> None:
        self.stop.set()
        assert self.thread is not None
        self.thread.join(timeout=2)
        need(not self.thread.is_alive() and self.error is None, "heartbeat failed at exit")


def _heartbeat_fd() -> int:
    value = os.environ.get(HEARTBEAT_ENV, "")
    need(value.isdecimal(), "missing parent heartbeat descriptor")
    return int(value)


def _exact_paths(nemo_dir: Path, checkpoint_dir: Path, output_dir: Path) -> None:
    need(all(p.is_absolute() and p == p.resolve() for p in
             (nemo_dir, checkpoint_dir, output_dir)), "unresolved diagnostic path")
    need(nemo_dir.is_dir() and not nemo_dir.is_symlink(), "NeMo checkout missing")
    need(checkpoint_dir.is_dir() and not checkpoint_dir.is_symlink(),
         "checkpoint directory missing")
    need(output_dir.is_dir() and not output_dir.is_symlink(),
         "parent diagnostic directory missing")
    need(Path.cwd().resolve() == output_dir, "worker must run in parent output directory")
    need(all((output_dir / name).is_file() and not (output_dir / name).is_symlink()
             for name in EXPECTED_FILES), "parent evidence files missing")
    need(not (output_dir / "checkpoints.jsonl").exists()
         and not (output_dir / "checkpoints.jsonl").is_symlink()
         and not (output_dir / "worker_completion.json").exists()
         and not (output_dir / "worker_completion.json").is_symlink(),
         "diagnostic worker output consumed")


def verify_pinned_public_sample(nemo_dir: Path, checkpoint_dir: Path) -> dict[str, Any]:
    """Verify pinned bytes before importing the official model code."""
    head = subprocess.run(("git", "-C", str(nemo_dir), "rev-parse", "HEAD"),
                          check=True, capture_output=True, text=True, timeout=10).stdout.strip()
    need(head == pinned.NEMO_COMMIT, "NeMo checkout revision drift")
    sources = {}
    for relative, expected in pinned.SOURCE_SHA256.items():
        actual, _size = sample.hash_file(nemo_dir / relative, cap=2 << 20)
        need(actual == expected, "pinned NeMo source changed")
        sources[relative] = actual
    samples = sample.check_sample(nemo_dir)
    config, _size = sample.hash_file(checkpoint_dir / "config.json", cap=2 << 20)
    need(config == pinned.CONFIG_SHA256, "VoiceChat config changed")
    # The prior H200 freeze verified the full weight. Recheck its bytes in this
    # new child, because a saved completion must bind the checkpoint it loaded.
    weight, size = sample.hash_file(checkpoint_dir / "model.safetensors",
                                    cap=pinned.WEIGHT_BYTES)
    need(weight == pinned.WEIGHT_SHA256 and size == pinned.WEIGHT_BYTES,
         "VoiceChat weight changed")
    return {"nemo_revision": head, "voicechat_revision": pinned.VOICECHAT_COMMIT,
            "nano_revision": pinned.NANO_COMMIT, "source_sha256": sources,
            "sample_sha256": samples, "sample_bytes": sample.SAMPLE_BYTES,
            "config_sha256": config, "weight_sha256": weight, "weight_bytes": size}


def import_pinned_official(nemo_dir: Path) -> tuple[Any, Any]:
    """Import the exact official example to reuse its immutable defaults."""
    script = nemo_dir / sample.SAMPLE_SCRIPT
    spec = importlib.util.spec_from_file_location("voicechat_pinned_official_fc", script)
    need(spec is not None and spec.loader is not None, "official script unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    helper = sys.modules.get("nemo.collections.speechlm2.inference.utils.offline_voicechat")
    need(helper is not None and Path(helper.__file__).resolve() ==
         (nemo_dir / "nemo/collections/speechlm2/inference/utils/offline_voicechat.py").resolve(),
         "official helper import escaped pinned NeMo checkout")
    model_module = sys.modules.get("nemo.collections.speechlm2.models.nemotron_voicechat")
    need(model_module is not None and Path(model_module.__file__).resolve() ==
         (nemo_dir / "nemo/collections/speechlm2/models/nemotron_voicechat.py").resolve(),
         "VoiceChat model import escaped pinned NeMo checkout")
    need(module.build_model is helper.build_model
         and module.load_wav_16k_mono is helper.load_wav_16k_mono
         and module.encode_system_prompt is helper.encode_system_prompt
         and module.render_fc_system_prompt is helper.render_fc_system_prompt,
         "official example helper binding changed")
    need(module.run_fc_offline_inference is helper.run_fc_offline_inference
         and callable(helper.run_offline_inference),
         "official first-pass helper binding changed")
    return module, helper


def prepare_official_inputs(official: Any, checkpoint_dir: Path, nemo_dir: Path,
                            *, device: str = "cuda") -> tuple[Any, Any, Any, Any, Any, dict]:
    """Mirror the pinned example's defaults through its own imported helpers."""
    need(device == "cuda", "diagnostic device changed")
    template = official.DEFAULT_TEMPLATE
    need(Path(template).resolve() ==
         (nemo_dir / "examples/speechlm2/function_calling/template.jinja").resolve(),
         "official template path changed")
    message = official.DEFAULT_SYSTEM_MESSAGE
    tools = official.DEFAULT_TOOLS
    need(type(message) is str and type(tools) is list and len(tools) == 1
         and tools[0]["function"]["name"] == "generate_random_number",
         "official default prompt or tool changed")
    prompt = official.render_fc_system_prompt(template, message, tools)
    model = official.build_model(str(checkpoint_dir), device=device)
    _wav, signal, lengths = official.load_wav_16k_mono(
        str(nemo_dir / sample.SAMPLE_WAV), device=device)
    prompt_tokens, prompt_lengths = official.encode_system_prompt(
        model, prompt, device=device)
    need(prompt_tokens is not None and prompt_lengths is not None,
         "official prompt tokenization empty")
    hashes = {"template_sha256": pinned.SOURCE_SHA256[
                 "examples/speechlm2/function_calling/template.jinja"],
              "system_message_sha256": digest(message.encode("utf-8")),
              "tools_sha256": digest(canonical({"tools": tools})),
              "prompt_sha256": digest(prompt.encode("utf-8")),
              "sample_id": "sample_fc"}
    return model, signal, lengths, prompt_tokens, prompt_lengths, hashes


def run_worker(*, nemo_dir: Path, checkpoint_dir: Path, output_dir: Path,
               heartbeat_fd: int) -> dict[str, Any]:
    """Run pinned pass 1 only; the exact parent/host must be reviewed elsewhere."""
    _exact_paths(nemo_dir, checkpoint_dir, output_dir)
    # The caller must set PYTHONPATH to the pinned checkout before this module
    # is imported; verify that imports resolve to that checkout after load.
    with Heartbeat(heartbeat_fd) as heartbeat:
        identity = verify_pinned_public_sample(nemo_dir, checkpoint_dir)
        heartbeat.check()
        official, helper = import_pinned_official(nemo_dir)
        heartbeat.check()
        model, signal, lengths, tokens, token_lengths, prompt_hashes = \
            prepare_official_inputs(official, checkpoint_dir, nemo_dir)
        identity.update(prompt_hashes)
        heartbeat.check()
        ledger = probe.CheckpointLedger(output_dir / "checkpoints.jsonl")
        try:
            core = probe.run_pinned_pass1_only(
                model=model, input_signal=signal, input_signal_lens=lengths,
                prompt_tokens=tokens, prompt_token_lens=token_lengths,
                offline_inference=helper.run_offline_inference,
                ledger=ledger, cuda=__import__("torch").cuda,
                nvml=probe.nvml_process_sample, device=0)
        finally:
            if not ledger.handle.closed:
                ledger.handle.close()
        heartbeat.check()
        receipt = {"schema": SCHEMA, "status": core["status"],
                   "identity": identity, "core": core, "no_second_pass": True}
        data = canonical(receipt)
        need(len(data) <= MAX_COMPLETION_BYTES, "worker receipt exceeds cap")
        _write_once(output_dir / "worker_completion.json", data)
        return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nemo-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    # No direct-launch option is provided. A future signed launcher must pass
    # the inherited heartbeat pipe and must supervise every byte and GPU poll.
    run_worker(nemo_dir=args.nemo_dir, checkpoint_dir=args.checkpoint_dir,
               output_dir=args.output_dir, heartbeat_fd=_heartbeat_fd())


if __name__ == "__main__":
    main()
