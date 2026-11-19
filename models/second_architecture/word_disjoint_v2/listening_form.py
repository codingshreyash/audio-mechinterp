"""Build an offline, reference-free listening form from the public 32-WAV queue.

The builder reads only ``listening/QUEUE.json`` and its opaque WAV filenames.
It does not open the private case map, manifest, or model outputs. The browser
stores first submissions locally and exports one record file per reviewer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


HERE = Path(__file__).resolve().parent
DEFAULT_BUNDLE = HERE / "rendered_v2/listening"
TEMPLATE = HERE / "listening_form_template.html"
APP = HERE / "listening_form.js"
EXPECTED_FIELDS = ["opaque_id", "reviewer_id", "verbatim_transcript",
                   "middle_word", "clear_or_unclear", "confidence"]
ID = re.compile(r"[0-9a-f]{20}\Z")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def public_data(bundle: Path) -> dict:
    queue_file = bundle / "QUEUE.json"
    if not queue_file.is_file() or queue_file.is_symlink():
        raise ValueError("public queue absent")
    queue = json.loads(queue_file.read_text())
    if (set(queue) != {"status", "scope", "provenance", "planned_clips",
                       "opaque_ids_in_order", "review_form_fields"}
            or queue["status"] != "blinded_listening_queue_no_results"
            or queue["planned_clips"] != 32
            or queue["review_form_fields"] != EXPECTED_FIELDS):
        raise ValueError("public queue schema differs")
    provenance = queue["provenance"]
    if (not isinstance(provenance, dict)
            or set(provenance) != {"manifest_sha256", "technical_binding_sha256",
                                   "auditor_sha256", "queue_builder_sha256"}
            or any(not isinstance(v, str) or re.fullmatch(r"[0-9a-f]{64}", v) is None
                   for v in provenance.values())):
        raise ValueError("public queue provenance invalid")
    order = queue["opaque_ids_in_order"]
    if (not isinstance(order, list) or len(order) != 32
            or len(set(order)) != 32
            or any(not isinstance(x, str) or ID.fullmatch(x) is None for x in order)):
        raise ValueError("opaque queue incomplete or duplicated")
    actual_wavs = {p.name for p in bundle.glob("*.wav")}
    expected_wavs = {ident + ".wav" for ident in order}
    if actual_wavs != expected_wavs:
        raise ValueError("anonymous WAV population differs")
    if {p.name for p in bundle.iterdir()} != expected_wavs | {"QUEUE.json"}:
        raise ValueError("listener-facing directory contains unexpected files")
    clips = []
    for ident in order:
        path = bundle / (ident + ".wav")
        if not path.is_file() or path.is_symlink():
            raise ValueError("anonymous WAV missing or linked")
        wav_sha = sha(path)
        derived_id = hashlib.sha256(("word-disjoint-listening-v2|"
                                     + provenance["manifest_sha256"] + "|"
                                     + wav_sha).encode()).hexdigest()[:20]
        if ident != derived_id:
            raise ValueError("anonymous WAV does not match its derived opaque ID")
        clips.append({"opaque_id": ident, "file": path.name,
                      "wav_sha256": wav_sha})
    return {"schema": "word_disjoint_blind_form_v1",
            "queue_sha256": sha(queue_file), "planned_clips": 32,
            "clips": clips}


def build(bundle: Path = DEFAULT_BUNDLE, output: Path | None = None) -> dict:
    bundle = bundle.resolve()
    if output is None:
        output = bundle / "LISTENING_FORM.html"
    if output.exists():
        raise FileExistsError("refusing to overwrite listening form")
    data = public_data(bundle)
    template = TEMPLATE.read_text()
    if template.count("__EMBEDDED_DATA__") != 1 or template.count("__APP_JS__") != 1:
        raise ValueError("form template slots differ")
    encoded = json.dumps(data, separators=(",", ":"), ensure_ascii=False)
    encoded = encoded.replace("<", "\\u003c").replace("&", "\\u0026")
    html = template.replace("__EMBEDDED_DATA__", encoded).replace("__APP_JS__", APP.read_text())
    output.write_text(html)
    return {"status": "blind_form_ready_no_human_reviews",
            "output": str(output), "output_sha256": sha(output),
            "queue_sha256": data["queue_sha256"], "clips": len(data["clips"])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.bundle, args.output), indent=2))


if __name__ == "__main__":
    main()
