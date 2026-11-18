"""Pinned, single-clip AudioSAE inference smoke test; not a paper replication."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import soundfile as sf
import torch
import librosa
from huggingface_hub import hf_hub_download, snapshot_download


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "audiosae_demo"))

from audio_sae import BatchTopKSAE, reconstruction_metrics  # noqa: E402
from audio_sae.models import MyHubert  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    backbone_revision = "dba3bb02fda4248b6e082697eee756de8fe8aa8a"
    sae_revision = "52e90df0a81bc23005e418a01cab475ca40141e7"
    backbone_dir = snapshot_download(
        repo_id="facebook/hubert-base-ls960",
        revision=backbone_revision,
        allow_patterns=["config.json", "pytorch_model.bin", "model.safetensors"],
    )
    checkpoint = Path(
        hf_hub_download(
            repo_id="Egorgij21/Audio-SAE-hubert-base",
            filename="layer_3/ae.pt",
            revision=sae_revision,
        )
    )
    audio = ROOT / "audio-jacobian-lens/artifacts/tts_corpus/audio/001.wav"
    # The generator uses numbered WAV names; select its first manifest row if needed.
    if not audio.exists():
        first = json.loads(
            (ROOT / "audio-jacobian-lens/artifacts/tts_corpus/manifest.jsonl")
            .read_text()
            .splitlines()[0]
        )
        audio = ROOT / "audio-jacobian-lens/artifacts/tts_corpus" / first["audio"]
    samples, sample_rate = sf.read(audio, dtype="float32")
    if samples.ndim != 1:
        raise ValueError((sample_rate, samples.shape))
    original_sample_rate = sample_rate
    if sample_rate != 16000:
        samples = librosa.resample(samples, orig_sr=sample_rate, target_sr=16000)
        sample_rate = 16000

    backbone = MyHubert(str(backbone_dir), sae_after_layer=3).eval()
    sae = BatchTopKSAE.from_pretrained(str(checkpoint), device="cpu").eval()
    with torch.no_grad():
        activations = backbone(torch.from_numpy(samples).unsqueeze(0))
        features = sae.encode(activations, use_threshold=True)
        reconstructed = sae.decode(features)
        metrics = reconstruction_metrics(activations, features, reconstructed)

    result = {
        "evidence": "pretrained-model inference smoke; not paper metric reproduction",
        "audio_path": str(audio.relative_to(ROOT)),
        "audio_sha256": sha256(audio),
        "source_sample_rate": original_sample_rate,
        "backbone_sample_rate": sample_rate,
        "backbone_repo": "facebook/hubert-base-ls960",
        "backbone_revision": backbone_revision,
        "sae_repo": "Egorgij21/Audio-SAE-hubert-base",
        "sae_revision": sae_revision,
        "sae_checkpoint_sha256": sha256(checkpoint),
        "layer": 3,
        "activation_shape": list(activations.shape),
        "feature_shape": list(features.shape),
        "metrics": vars(metrics),
        "torch_version": torch.__version__,
    }
    output = ROOT / "audiosae_smoke.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
