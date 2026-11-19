"""Preserve and verify every archive for the isolated VoiceChat FFmpeg prefix.

This source-only staging helper never imports a model or edits an installed package.
It writes archives exclusively under PREFIX/package-archives and can resume after a
network interruption. A later native manifest binds these bytes to inference.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
from urllib.parse import unquote, urlparse
from urllib.request import urlopen


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def records(prefix: Path) -> list[dict]:
    rows = []
    for meta_path in sorted((prefix / "conda-meta").glob("*.json")):
        row = json.loads(meta_path.read_text())
        url, filename, expected = row.get("url"), row.get("fn"), row.get("sha256")
        if not all(isinstance(value, str) for value in (url, filename, expected)):
            raise ValueError(f"missing archive identity in {meta_path}")
        parsed = urlparse(url)
        if (parsed.scheme != "https" or parsed.netloc != "conda.anaconda.org"
                or not parsed.path.startswith("/conda-forge/linux-64/")
                and not parsed.path.startswith("/conda-forge/noarch/")):
            raise ValueError(f"unexpected channel URL: {url}")
        if unquote(parsed.path.rsplit("/", 1)[-1]) != filename or Path(filename).name != filename:
            raise ValueError(f"filename mismatch in {meta_path}")
        if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
            raise ValueError(f"missing valid SHA-256 in {meta_path}")
        rows.append({"metadata_path": f"conda-meta/{meta_path.name}",
                     "url": url, "filename": filename, "sha256": expected})
    if not rows or len({r["filename"] for r in rows}) != len(rows):
        raise ValueError("empty or duplicate package inventory")
    ffmpeg = [r for r in rows if Path(r["metadata_path"]).name.startswith("ffmpeg-7.1.1-")]
    if len(ffmpeg) != 1:
        raise ValueError("expected exactly one ffmpeg 7.1.1 package")
    return rows


def preserve(row: dict, archive_dir: Path) -> dict:
    target = archive_dir / row["filename"]
    expected = row["sha256"]
    if target.exists() and sha256(target) == expected:
        return {"path": f"package-archives/{row['filename']}", "sha256": expected,
                "bytes": target.stat().st_size, "metadata_path": row["metadata_path"]}
    temporary = archive_dir / f".{row['filename']}.part-{os.getpid()}"
    try:
        with urlopen(row["url"], timeout=120) as source, temporary.open("wb") as sink:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                sink.write(chunk)
        actual = sha256(temporary)
        if actual != expected:
            raise ValueError(f"archive SHA-256 mismatch: {row['filename']}: {actual}")
        os.replace(temporary, target)
        return {"path": f"package-archives/{row['filename']}", "sha256": expected,
                "bytes": target.stat().st_size, "metadata_path": row["metadata_path"]}
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    prefix = args.prefix.resolve(strict=True)
    rows = records(prefix)
    archive_dir = prefix / "package-archives"
    archive_dir.mkdir(exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(lambda row: preserve(row, archive_dir), rows))
    results.sort(key=lambda row: row["path"])
    output = prefix / "package-archives-verified.json"
    output.write_text(json.dumps(results, sort_keys=True, separators=(",", ":")) + "\n")
    print(json.dumps({"count": len(results), "total_bytes": sum(r["bytes"] for r in results),
                      "inventory_sha256": sha256(output), "output": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
