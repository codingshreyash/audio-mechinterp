"""Download exact model and matching Neuronpedia Jacobian lens for local reproduction."""

from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download


ROOT = Path(__file__).resolve().parent
MODEL_REPO = "Qwen/Qwen3-1.7B"
MODEL_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
LENS_REPO = "neuronpedia/jacobian-lens"
LENS_REVISION = "d2527c3cdbaa60dd4b65cbca1604921f044e989a"
LENS_FILENAME = (
    "qwen3-1.7b/jlens/Salesforce-wikitext/"
    "Qwen3-1.7B_jacobian_lens.pt"
)


if __name__ == "__main__":
    cache = ROOT / "hf_cache"
    model = snapshot_download(
        MODEL_REPO,
        revision=MODEL_REVISION,
        cache_dir=cache,
        allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt"],
    )
    lens = hf_hub_download(
        LENS_REPO,
        LENS_FILENAME,
        revision=LENS_REVISION,
        cache_dir=cache,
    )
    print(f"model={model}")
    print(f"lens={lens}")
