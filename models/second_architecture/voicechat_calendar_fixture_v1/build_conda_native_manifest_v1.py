"""Freeze the isolated Conda FFmpeg package bytes for VoiceChat V4.

This source-only helper uses installed conda-meta records and preserved archives.
It performs no model import or inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import unquote, urlparse


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    prefix = args.prefix.resolve(strict=True)
    evidence = args.evidence.resolve(strict=True)
    packages = []
    archives = []
    for path in sorted((prefix / "conda-meta").glob("*.json")):
        raw = path.read_bytes()
        row = json.loads(raw)
        url, name, expected = row["url"], row["fn"], row["sha256"]
        parsed = urlparse(url)
        if parsed.scheme != "https" or parsed.netloc != "conda.anaconda.org":
            raise ValueError(f"non-official URL: {url}")
        if not (parsed.path.startswith("/conda-forge/linux-64/") or
                parsed.path.startswith("/conda-forge/noarch/")):
            raise ValueError(f"non-conda-forge package: {url}")
        if unquote(parsed.path.rsplit("/", 1)[-1]) != name:
            raise ValueError(f"URL/filename mismatch: {url}")
        if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
            raise ValueError(f"invalid metadata SHA-256: {path}")
        archive = prefix / "package-archives" / name
        if not archive.is_file() or archive.is_symlink():
            raise ValueError(f"missing or linked archive: {archive}")
        if digest(archive) != expected:
            raise ValueError(f"archive SHA-256 mismatch: {archive}")
        packages.append({"name": row["name"], "version": row["version"],
                         "build": row["build"], "channel": row["channel"],
                         "url": url, "filename": name, "sha256": expected,
                         "md5": row.get("md5"), "metadata_path": f"conda-meta/{path.name}",
                         "metadata_sha256": hashlib.sha256(raw).hexdigest()})
        archives.append({"path": f"package-archives/{name}",
                         "sha256": expected, "bytes": archive.stat().st_size})
    if len(packages) != 129 or len({p["filename"] for p in packages}) != 129:
        raise ValueError(f"unexpected package inventory count: {len(packages)}")
    ffmpeg = [p for p in packages if p["name"] == "ffmpeg"]
    if len(ffmpeg) != 1 or ffmpeg[0]["version"] != "7.1.1":
        raise ValueError("FFmpeg 7.1.1 package missing")
    packages.sort(key=lambda row: row["url"])
    archives.sort(key=lambda row: row["path"])
    spec = prefix / "conda-explicit.txt"
    spec.write_text("@EXPLICIT\n" + "\n".join(
        p["url"] + "#sha256=" + p["sha256"] for p in packages) + "\n")
    manifest = {
        "schema": "voicechat_native_source_manifest_v1",
        "kind": "conda_explicit",
        "ffmpeg_major": 7,
        "explicit_spec": {"path": "conda-explicit.txt", "sha256": digest(spec),
                          "bytes": spec.stat().st_size},
        "package_archives": archives,
    }
    manifest_path = prefix / "native-source-manifest.json"
    canonical(manifest_path, manifest)
    canonical(evidence / "package-inventory.json", packages)

    library_rows = []
    for path in sorted((prefix / "lib").rglob("*")):
        relative = str(path.relative_to(prefix))
        if path.is_symlink():
            resolved = path.resolve(strict=True)
            if not resolved.is_relative_to(prefix):
                raise ValueError(f"external library symlink: {path}")
            library_rows.append({"path": relative, "type": "symlink",
                                 "target": str(path.readlink()),
                                 "resolved": str(resolved.relative_to(prefix)),
                                 "resolved_sha256": digest(resolved) if resolved.is_file() else None})
        elif path.is_file():
            library_rows.append({"path": relative, "type": "file", "bytes": path.stat().st_size,
                                 "sha256": digest(path)})
        elif not path.is_dir():
            raise ValueError(f"unexpected library file type: {path}")
    canonical(evidence / "library-inventory.json", library_rows)
    print(json.dumps({"package_count": len(packages), "archive_bytes": sum(r["bytes"] for r in archives),
                      "library_entries": len(library_rows), "manifest_sha256": digest(manifest_path),
                      "spec_sha256": digest(spec), "manifest": str(manifest_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
