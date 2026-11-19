"""Source-only pinned FP32 Voxtral loader; public model load is hard HOLD.

The private factory is for fake source tests and a later independently reviewed
launcher. It rechecks actual QA/preflight/host via the admission launcher before
constructing the processor or loading model weights. No live entry is enabled.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable

from models.second_architecture import voxtral_track_b_v6_pilot_worker_source_v1 as worker
from models.second_architecture import voxtral_track_b_v6_pre_pilot_admission_contract_v1 as contract
from models.second_architecture import voxtral_track_b_v6_pre_pilot_launcher_source_v1 as admission_launcher
from models.second_architecture import voxtral_track_b_v6_mps_runtime_adapter_source_v1 as runtime


def _pinned_snapshot_path() -> Path:
    source_dir = str(Path(__file__).resolve().parent)
    sys.path.insert(0, source_dir)
    try:
        import voxtral_fit_pilot_capability_v1 as base
        return Path(base.SNAPSHOT)
    finally:
        sys.path.remove(source_dir)


def _snapshot_manifest() -> dict[str, Any]:
    """Rehash pinned tree, config, index, shards, processor and tokenizer bytes."""
    source_dir = str(Path(__file__).resolve().parent)
    sys.path.insert(0, source_dir)
    try:
        import voxtral_fit_pilot_capability_v1 as base
        from models.second_architecture import voxtral_fp32_track_b_preflight_v1 as preflight
        return base.validate_snapshot({
            'processor_config_sha256': preflight.PROCESSOR_CONFIG_SHA,
            'tokenizer_asset_sha256': preflight.TOKENIZER_SHA})
    finally:
        sys.path.remove(source_dir)


def _installed_source_map() -> dict[str, str]:
    source_dir = str(Path(__file__).resolve().parent)
    sys.path.insert(0, source_dir)
    try:
        from models.second_architecture import voxtral_fp32_track_b_preflight_v1 as preflight
        return preflight.verify_installed_source()
    finally:
        sys.path.remove(source_dir)


def _load_after_verified(*, files: worker.AdmissionFiles, review_path: Path,
                         pins: contract.ExternalPins, torch_api: Any,
                         snapshot: Path, processor_factory: Callable[..., Any],
                         model_factory: Callable[..., Any]) -> tuple[Any, Any, dict[str, Any]]:
    """Private dependency-injected path; fake tests prove gate/order/state checks."""
    proof = admission_launcher.verify_pre_pilot(files=files, review_path=review_path,
                                                  pins=pins)
    if (proof.get('worker_sha256') != pins.worker_sha256 or
            proof.get('admission_sha256') != pins.admission_sha256 or
            proof.get('review_sha256') != pins.review_sha256):
        raise ValueError('Preload proof is not bound to pinned W/A/R')
    if snapshot.resolve(strict=True) != _pinned_snapshot_path().resolve(strict=True):
        raise ValueError('Supplied snapshot path differs from pinned Voxtral snapshot')
    frozen = json.loads(files.preflight.read_text())
    gate = json.loads(files.admission.read_text())
    expected_snapshot = frozen['snapshot']
    bound_tree = gate['bindings']['checkpoint_tree_sha256']
    before = _snapshot_manifest()
    if (before != expected_snapshot or
            before.get('snapshot_tree', {}).get('sha256') != bound_tree):
        raise ValueError('Pinned snapshot bytes changed before factory')
    expected_source = frozen['installed_source_sha256']
    if _installed_source_map() != expected_source:
        raise ValueError('Pinned installed source bytes changed before factory')
    if not torch_api.backends.mps.is_available():
        raise RuntimeError('Pinned FP32 Voxtral loader requires MPS')
    cap = runtime.configure_mps_allocator(torch_api.mps)
    expected = frozen['processor']
    processor = processor_factory.from_pretrained(str(snapshot), local_files_only=True)
    if (type(processor).__name__ != expected['processor_type'] or
            type(processor.tokenizer).__name__ != expected['tokenizer_type'] or
            int(processor.tokenizer.eos_token_id) != 2 or
            int(processor.audio_token_id) != 24):
        raise ValueError('Pinned Voxtral processor type or token IDs differ')
    model = model_factory.from_pretrained(
        str(snapshot), local_files_only=True, dtype=torch_api.float32,
        low_cpu_mem_usage=True, device_map='mps',
        attn_implementation='sdpa').eval()
    model.requires_grad_(False)
    parameters = list(model.parameters())
    pad = model.generation_config.pad_token_id
    if (not parameters or model.training or
            any(p.dtype != torch_api.float32 or p.device.type != 'mps'
                for p in parameters) or
            int(model.generation_config.eos_token_id) != 2 or
            not isinstance(pad, int) or pad < 0):
        raise ValueError('Pinned Voxtral FP32/eval/MPS/generation state differs')
    torch_api.mps.synchronize()
    after = _snapshot_manifest()
    if (after != expected_snapshot or
            after.get('snapshot_tree', {}).get('sha256') != bound_tree):
        raise ValueError('Pinned snapshot bytes changed during factory')
    if _installed_source_map() != expected_source:
        raise ValueError('Pinned installed source bytes changed during factory')
    if int(torch_api.mps.driver_allocated_memory()) > runtime.CAP_BYTES:
        raise MemoryError('Pinned Voxtral model load exceeded allocator cap')
    return model, processor, {'preload_proof': proof, 'allocator_cap': cap,
                              'fp32_parameter_tensors': len(parameters), 'pad_token_id': pad}


def _load_pinned_fp32_voxtral(*, files: worker.AdmissionFiles,
                              review_path: Path,
                              pins: contract.ExternalPins) -> tuple[Any, Any, dict[str, Any]]:
    """Future child-only production composition; no current caller or public path."""
    import torch
    from transformers import AutoProcessor, VoxtralForConditionalGeneration
    return _load_after_verified(
        files=files, review_path=review_path, pins=pins,
        torch_api=torch, snapshot=_pinned_snapshot_path(),
        processor_factory=AutoProcessor,
        model_factory=VoxtralForConditionalGeneration)


def load_v6_pilot_model(*_args: Any, **_kwargs: Any) -> None:
    """Public entry remains hard HOLD pending real QA/admission/launcher review."""
    raise ValueError('V6 pinned Voxtral loader HOLD: no reviewed live launcher')
