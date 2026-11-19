"""No-model V6 pilot batch adapter; public live execution remains HOLD.

Prospectively selects the older FIT color_05 Piper/silver clip. V6 says
"silver" but does not itself choose a voice; this source makes that choice
explicit for independent review before any model weight load.
"""

from __future__ import annotations

import hashlib
import io
import json
import sys
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import torch

from models.second_architecture import voxtral_track_b_v6_pilot_worker_source_v1 as worker
from models.second_architecture import voxtral_track_b_v6_pre_pilot_admission_contract_v1 as contract
from models.second_architecture import voxtral_track_b_v6_pre_pilot_launcher_source_v1 as admission_launcher
from models.second_architecture import voxtral_track_b_v6_pinned_loader_source_v1 as pinned_loader


OLD_WAV_SHA256 = 'fe554972f68b470fb0f4a1c844b874e70758279c281bba47e963792bef2634a9'
OLD_FIT_RAW_HASHES = {
    'input_ids': '915a4530875920b9c988724ca9e7ed8e63550bdfec3302ddb66e84e67b4b314a',
    'attention_mask': '9ecc0f254cc29a01ab79e006021d96ef320b7438202692dcee10c770fbd65282',
    'input_features': '36ec6b38ba8202821e6cabf8e4047ff493a58dce826279385eecb86f6b325d48',
}
FIT_PROMPT = 'Answer the question in the audio with exactly one word. No explanation or punctuation.'
TRACK_PROMPT = ('What word did the speaker say between "said" and "and"? '
                'Answer with exactly one word. No explanation or punctuation.')
AUDIO_ID = 24
EOS_ID = 2


def _sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _raw_tensor_sha(value: torch.Tensor) -> str:
    return _sha_bytes(value.detach().cpu().contiguous().numpy().tobytes())


def _sibling_import(name: str) -> Any:
    source_dir = str(Path(__file__).resolve().parent)
    sys.path.insert(0, source_dir)
    try:
        return __import__(name)
    finally:
        sys.path.remove(source_dir)


def _resolve_old_fit_silver() -> Path:
    """Check frozen FIT selection and return exact Piper/color_05/silver WAV."""
    first = _sibling_import('voxtral_fit_v2_causal_first_record')
    first.verify_frozen()
    if first.base.PROMPT != FIT_PROMPT:
        raise ValueError('Frozen FIT prompt differs')
    masks = json.loads(first.MASKS.read_text())
    records = [row for row in masks['records']
               if (row.get('pair_id'), row.get('engine'), row.get('phrasing_id')) ==
               ('color_05', 'piper', 'fit_chosen')]
    if len(records) != 1 or records[0].get('words') != ['silver', 'teal']:
        raise ValueError('Frozen FIT Piper silver record differs')
    clips = [clip for clip in records[0]['clips'] if clip.get('word') == 'silver']
    if len(clips) != 1 or clips[0].get('wav_sha256') != OLD_WAV_SHA256:
        raise ValueError('Frozen FIT Piper silver WAV binding differs')
    raw_path = first.ROOT / clips[0]['wav_path']
    path = raw_path.resolve(strict=True)
    expected_root = (first.ROOT / 'eval/audio_jlens/corpus_v1/rendered/clips').resolve()
    if (raw_path.is_symlink() or not path.is_relative_to(expected_root) or
            worker.sha_file(path) != OLD_WAV_SHA256):
        raise ValueError('Frozen FIT Piper silver WAV path or bytes differ')
    with wave.open(str(path), 'rb') as wav:
        if ((wav.getnchannels(), wav.getframerate(), wav.getsampwidth(), wav.getcomptype()) !=
                (1, 16000, 2, 'NONE') or wav.getnframes() != clips[0]['frames']):
            raise ValueError('Frozen FIT Piper silver WAV format/frames differ')
    return path


def _wav_from_zero_pcm(frames: int) -> tuple[bytes, bytes]:
    if not isinstance(frames, int) or not 1 <= frames <= 480000 + 512 * 128:
        raise ValueError('Synthetic zero PCM frame count invalid')
    pcm = bytes(2 * frames)
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(pcm)
    return pcm, buffer.getvalue()


def _checked_batch(batch: Mapping[str, torch.Tensor], raw_hashes: dict[str, str],
                   *, feature_shape: tuple[int, ...], placeholders: int,
                   attention_length: int) -> dict[str, torch.Tensor]:
    if not isinstance(batch, Mapping) or set(batch) != {
            'input_ids', 'attention_mask', 'input_features'}:
        raise ValueError('Processor batch field set differs')
    value = {key: batch[key].detach().cpu().contiguous() for key in batch}
    ids, mask, features = (value[key] for key in ('input_ids', 'attention_mask', 'input_features'))
    if (ids.dtype != torch.int64 or mask.dtype != torch.int64 or
            features.dtype != torch.float32 or ids.ndim != 2 or ids.shape[0] != 1 or
            mask.shape != ids.shape or ids.shape[1] != attention_length or
            tuple(features.shape) != feature_shape or
            int((ids == AUDIO_ID).sum()) != placeholders or
            not bool(torch.all(mask == 1)) or not bool(torch.isfinite(features).all())):
        raise ValueError('Processor batch dtype, shape, mask or audio placeholders differ')
    for key, expected in raw_hashes.items():
        if _raw_tensor_sha(value[key]) != expected:
            raise ValueError(f'Processor {key} raw bytes differ from frozen receipt')
    return value


@dataclass(frozen=True)
class BoundPilotBatches:
    old_fit: dict[str, torch.Tensor]
    synthetic: dict[str, torch.Tensor]
    old_fit_sha256: str
    synthetic_sha256: str
    old_wav_sha256: str
    synthetic_wav_sha256: str
    synthetic_pcm_sha256: str


def _build_with_verified_processor(*, processor: Any, old_wav: Path,
                                   frozen: dict[str, Any],
                                   bindings: dict[str, Any]) -> BoundPilotBatches:
    """No model. Exact PCM/WAV/prompt/tensor hashes precede worker loading."""
    expected_processor = frozen['processor']
    if (type(processor).__name__ != expected_processor['processor_type'] or
            type(processor.tokenizer).__name__ != expected_processor['tokenizer_type'] or
            int(processor.tokenizer.eos_token_id) != EOS_ID or
            int(processor.audio_token_id) != AUDIO_ID or
            expected_processor['prompt'] != TRACK_PROMPT):
        raise ValueError('Pinned processor or Track B prompt differs')
    if worker.sha_file(old_wav) != OLD_WAV_SHA256:
        raise ValueError('Old FIT silver WAV bytes differ')
    old_conv = [{'role': 'user', 'content': [
        {'type': 'audio', 'path': str(old_wav)}, {'type': 'text', 'text': FIT_PROMPT}]}]
    old_raw = processor.apply_chat_template(
        old_conv, tokenize=True, return_dict=True, return_tensors='pt')
    old = _checked_batch(old_raw, OLD_FIT_RAW_HASHES,
                         feature_shape=(1, 128, 3000), placeholders=375,
                         attention_length=396)
    geometry = frozen['v6_synthetic_zero_geometry']
    if (worker.sha_bytes(json.dumps(geometry, sort_keys=True,
            separators=(',', ':')).encode()) != bindings['synthetic_geometry_sha256'] or
            geometry['longest_alias']['token_ids_with_eos'] != list(worker.LONGEST_ALIAS) or
            geometry['projector_rows_implied'] != geometry['audio_placeholder_count']):
        raise ValueError('Synthetic geometry or alias differs from admission')
    frames = geometry['first_componentwise_satisfying_frames']
    max_frames = geometry['selected_max_frames']
    if (frames < max_frames or (frames - max_frames) % 128 or
            (frames - max_frames) // 128 > 512):
        raise ValueError('Synthetic frame selection differs from V6 bounded search')
    pcm, wav_bytes = _wav_from_zero_pcm(frames)
    if (_sha_bytes(pcm) != geometry['pcm_sha256'] or
            _sha_bytes(wav_bytes) != geometry['wav_sha256']):
        raise ValueError('Synthetic zero PCM or WAV bytes differ')
    with tempfile.TemporaryDirectory(prefix='voxtral_v6_bound_zero_') as temp:
        wav_path = Path(temp) / 'zero.wav'
        wav_path.write_bytes(wav_bytes)
        zero_conv = [{'role': 'user', 'content': [
            {'type': 'audio', 'path': str(wav_path)},
            {'type': 'text', 'text': TRACK_PROMPT}]}]
        zero_raw = processor.apply_chat_template(
            zero_conv, tokenize=True, return_dict=True, return_tensors='pt')
    zero = _checked_batch(
        zero_raw,
        {'input_ids': geometry['input_ids_sha256'],
         'attention_mask': geometry['attention_mask_sha256'],
         'input_features': geometry['input_features_sha256']},
        feature_shape=tuple(geometry['input_features_shape']),
        placeholders=int(geometry['audio_placeholder_count']),
        attention_length=int(geometry['attention_mask_length']))
    old_digest, zero_digest = worker._batch_sha(old), worker._batch_sha(zero)
    if (old_digest != bindings['old_fit_batch_sha256'] or
            zero_digest != bindings['synthetic_batch_sha256']):
        raise ValueError('Pilot batch SHA differs from admission')
    return BoundPilotBatches(old, zero, old_digest, zero_digest, OLD_WAV_SHA256,
                             _sha_bytes(wav_bytes), _sha_bytes(pcm))


def build_bound_pilot_batches(*, files: worker.AdmissionFiles,
                              review_path: Path, pins: contract.ExternalPins,
                              processor_factory: Callable[..., Any] | None = None) -> BoundPilotBatches:
    """Fail closed on actual QA and no-weight pins; no candidate model factory."""
    admission_launcher.verify_pre_pilot(files=files, review_path=review_path, pins=pins)
    gate = contract.verify_contract(files, review_path, pins)
    bindings = gate['bindings']
    frozen = json.loads(files.preflight.read_text())
    old_wav = _resolve_old_fit_silver()
    snapshot = pinned_loader._pinned_snapshot_path().resolve(strict=True)
    before = pinned_loader._snapshot_manifest()
    source_before = pinned_loader._installed_source_map()
    if before != frozen['snapshot'] or source_before != frozen['installed_source_sha256']:
        raise ValueError('Pinned snapshot/source bytes differ before processor')
    if processor_factory is None:
        from transformers import AutoProcessor
        processor_factory = AutoProcessor
    processor = processor_factory.from_pretrained(str(snapshot), local_files_only=True)
    result = _build_with_verified_processor(processor=processor, old_wav=old_wav,
                                            frozen=frozen, bindings=bindings)
    if (pinned_loader._snapshot_manifest() != before or
            pinned_loader._installed_source_map() != source_before):
        raise ValueError('Pinned snapshot/source bytes changed during batch construction')
    return result
