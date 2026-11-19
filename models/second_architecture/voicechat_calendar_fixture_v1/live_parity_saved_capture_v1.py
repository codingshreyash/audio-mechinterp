"""Lossless, hash-checked CPU capture format for future VoiceChat parity arms.

No NeMo import, model load, process launch or QA admission occurs here. The
format stores nested plain values and numeric torch/NumPy tensors without
pickle. A future separately reviewed worker must use it for the exact two
pass results, call kwargs, codec codes, and unspliced audio.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
from pathlib import Path
from typing import Any

SCHEMA = "voicechat_parity_saved_capture_v1"
MANIFEST = "capture_manifest.json"
MAX_NODES = 100_000
MAX_DEPTH = 64


def _require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def _digest(path: Path, cap: int) -> str:
    _require(path.is_file() and not path.is_symlink(), "Capture path is not a file")
    size = path.stat().st_size
    _require(size <= cap, "Capture file exceeds read cap")
    seen = 0
    h = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(min(65536, cap - seen + 1)):
            seen += len(block)
            _require(seen <= cap, "Capture file grew beyond read cap")
            h.update(block)
    _require(seen == size and path.stat().st_size == size,
             "Capture file changed while hashing")
    return h.hexdigest()


def _total(directory: Path, cap: int) -> int:
    total = 0
    for path in directory.iterdir():
        _require(path.is_file() and not path.is_symlink(),
                 "Capture has non-file or linked artifact")
        total += path.stat().st_size
        _require(total <= cap, "Capture exceeds byte cap")
    return total


def save_capture(value: Any, directory: Path, *, byte_cap: int) -> dict[str, Any]:
    """Save an exact nested capture; return a manifest hash/byte receipt."""
    _require(type(byte_cap) is int and byte_cap > 0, "Capture byte cap invalid")
    import numpy as np
    import torch

    directory.mkdir(parents=True, exist_ok=False)
    tensors: dict[str, dict[str, Any]] = {}
    nodes = 0

    def encode(item: Any, depth: int) -> Any:
        nonlocal nodes
        nodes += 1
        _require(nodes <= MAX_NODES and depth <= MAX_DEPTH,
                 "Capture structure exceeds limits")
        if isinstance(item, np.generic):
            item = np.asarray(item)
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            dtype = str(tensor.dtype)
            _require(dtype in {"torch.float16", "torch.float32", "torch.float64",
                               "torch.bfloat16", "torch.int8", "torch.int16",
                               "torch.int32", "torch.int64", "torch.uint8",
                               "torch.bool"}, "Capture torch dtype unsupported")
            array = (tensor.view(torch.uint16).numpy() if dtype == "torch.bfloat16"
                     else tensor.numpy())
            kind = "torch"
        elif isinstance(item, np.ndarray):
            _require(item.dtype.kind in "biuf" and not item.dtype.hasobject,
                     "Capture NumPy dtype unsupported")
            array = np.array(item, copy=True, order="C")
            dtype = str(item.dtype)
            kind = "numpy"
        else:
            array = None
        if array is not None:
            _require(array.nbytes + _total(directory, byte_cap) + 512 <= byte_cap,
                     "Capture tensor exceeds byte cap before write")
            name = f"tensor_{len(tensors):05d}.npy"
            path = directory / name
            with path.open("xb") as stream:
                np.save(stream, array, allow_pickle=False)
            _total(directory, byte_cap)
            tensors[name] = {"sha256": _digest(path, byte_cap),
                             "kind": kind, "dtype": dtype,
                             "shape": list(item.shape),
                             "storage_dtype": str(array.dtype)}
            return {"type": "tensor", "file": name}
        if item is None or type(item) in (bool, int, str):
            return {"type": "plain", "value": item}
        if type(item) is float:
            _require(math.isfinite(item), "Capture scalar is non-finite")
            return {"type": "plain", "value": item}
        if type(item) is bytes:
            return {"type": "bytes", "base64": base64.b64encode(item).decode("ascii")}
        if type(item) is list:
            return {"type": "list", "items": [encode(x, depth + 1) for x in item]}
        if type(item) is tuple:
            return {"type": "tuple", "items": [encode(x, depth + 1) for x in item]}
        if type(item) is dict:
            _require(all(type(key) is str for key in item),
                     "Capture mapping key is not text")
            return {"type": "dict", "items": [
                [key, encode(sub, depth + 1)] for key, sub in item.items()]}
        raise ValueError(f"Capture value type unsupported: {type(item).__name__}")

    tree = encode(value, 0)
    manifest = {"schema": SCHEMA, "tree": tree, "tensors": tensors,
                "nodes": nodes}
    payload = (json.dumps(manifest, ensure_ascii=True, sort_keys=True,
                          separators=(",", ":"), allow_nan=False) + "\n").encode("ascii")
    _require(len(payload) + _total(directory, byte_cap) <= byte_cap,
             "Capture manifest exceeds byte cap before write")
    manifest_path = directory / MANIFEST
    with manifest_path.open("xb") as stream:
        stream.write(payload)
    total = _total(directory, byte_cap)
    return {"schema": SCHEMA, "manifest_sha256": _digest(manifest_path, byte_cap),
            "total_bytes": total, "tensor_count": len(tensors)}


def load_capture(directory: Path, *, byte_cap: int,
                 expected_manifest_sha256: str) -> Any:
    """Verify every artifact, then reconstruct values for saved-data comparison."""
    _require(type(byte_cap) is int and byte_cap > 0, "Capture byte cap invalid")
    _total(directory, byte_cap)
    manifest_path = directory / MANIFEST
    _require(_digest(manifest_path, byte_cap) == expected_manifest_sha256,
             "Capture manifest hash differs")
    with manifest_path.open("rb") as stream:
        manifest_bytes = stream.read(byte_cap + 1)
    _require(len(manifest_bytes) <= byte_cap
             and hashlib.sha256(manifest_bytes).hexdigest() == expected_manifest_sha256,
             "Capture manifest changed during bounded read")
    manifest = json.loads(manifest_bytes.decode("ascii"))
    _require(manifest.get("schema") == SCHEMA
             and isinstance(manifest.get("tensors"), dict),
             "Capture manifest schema differs")
    tensor_rows = manifest["tensors"]
    _require(set(path.name for path in directory.iterdir()) ==
             {MANIFEST, *tensor_rows.keys()}, "Capture artifact set differs")
    import numpy as np
    import torch

    tensors: dict[str, Any] = {}
    for name, row in tensor_rows.items():
        _require(isinstance(name, str) and name.startswith("tensor_")
                 and name.endswith(".npy") and Path(name).name == name,
                 "Capture tensor filename invalid")
        path = directory / name
        _require(_digest(path, byte_cap) == row.get("sha256"),
                 "Capture tensor hash differs")
        with path.open("rb") as stream:
            array = np.load(stream, allow_pickle=False)
        _require(_digest(path, byte_cap) == row.get("sha256"),
                 "Capture tensor changed during load")
        _require(list(array.shape) == row.get("shape")
                 and str(array.dtype) == row.get("storage_dtype"),
                 "Capture tensor metadata differs")
        if row.get("kind") == "torch":
            tensor = torch.from_numpy(array.copy())
            if row.get("dtype") == "torch.bfloat16":
                _require(str(array.dtype) == "uint16", "Bfloat16 storage differs")
                tensor = tensor.view(torch.bfloat16)
            _require(str(tensor.dtype) == row.get("dtype"),
                     "Capture torch dtype differs")
            tensors[name] = tensor
        else:
            _require(row.get("kind") == "numpy"
                     and str(array.dtype) == row.get("dtype"),
                     "Capture NumPy dtype differs")
            tensors[name] = array

    nodes = 0
    used: set[str] = set()

    def decode(node: Any, depth: int) -> Any:
        nonlocal nodes
        nodes += 1
        _require(nodes <= MAX_NODES and depth <= MAX_DEPTH
                 and isinstance(node, dict), "Capture tree invalid")
        kind = node.get("type")
        if kind == "plain":
            value = node.get("value")
            _require(value is None or type(value) in (bool, int, float, str),
                     "Capture plain value invalid")
            _require(type(value) is not float or math.isfinite(value),
                     "Capture scalar is non-finite")
            return value
        if kind == "bytes":
            try:
                return base64.b64decode(node["base64"], validate=True)
            except (KeyError, ValueError) as error:
                raise ValueError("Capture byte encoding invalid") from error
        if kind == "tensor":
            name = node.get("file")
            _require(name in tensors and name not in used,
                     "Capture tensor reference invalid")
            used.add(name)
            return tensors[name]
        if kind in ("list", "tuple"):
            values = [decode(child, depth + 1) for child in node.get("items", [])]
            return values if kind == "list" else tuple(values)
        if kind == "dict":
            pairs = node.get("items")
            _require(isinstance(pairs, list), "Capture mapping invalid")
            result = {}
            for pair in pairs:
                _require(isinstance(pair, list) and len(pair) == 2
                         and type(pair[0]) is str and pair[0] not in result,
                         "Capture mapping entry invalid")
                result[pair[0]] = decode(pair[1], depth + 1)
            return result
        raise ValueError("Capture node type invalid")

    result = decode(manifest.get("tree"), 0)
    _require(nodes == manifest.get("nodes") and used == set(tensors),
             "Capture node/tensor inventory differs")
    return result
