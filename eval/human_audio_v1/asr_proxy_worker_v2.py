"""Require the reviewed GO and live bounded parent before V1 model routine."""
from __future__ import annotations

import argparse
import json
import os
import select
import threading
from pathlib import Path

from . import asr_proxy_gate_v2 as gate
from . import asr_proxy_worker_v1 as old


def need(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def line(fd: int, timeout: float) -> bytes:
    # Read only one byte at a time so a nonce and first heartbeat may share a pipe buffer.
    from time import monotonic
    deadline = monotonic() + timeout
    data = bytearray()
    while len(data) < 256:
        ready, _, _ = select.select([fd], [], [], max(0, deadline - monotonic()))
        need(bool(ready), "live parent channel timed out")
        char = os.read(fd, 1)
        need(bool(char), "live parent channel closed")
        if char == b"\n":
            return bytes(data)
        data.extend(char)
    raise ValueError("live parent line too long")


def watch(fd: int) -> None:
    while True:
        try:
            need(line(fd, 2.0) == b"beat", "parent heartbeat lost")
        except Exception:
            os._exit(78)


def run(attempt: Path, admission: Path, live_fd: int) -> None:
    need(live_fd >= 3 and attempt == old.ATTEMPT and admission == old.ADMISSION,
         "fixed V2 attempt and inherited channel required")
    go = gate.validate_go()  # first; no model weights loaded
    launch = json.loads((attempt / "launch.json").read_text())
    need(launch.get("schema") == "human_asr_proxy_parent_v1"
         and launch.get("schema_v2") == "human_asr_proxy_parent_v2"
         and launch.get("worker_parent_pid") == os.getppid()
         and launch.get("admission_sha256") == go["admission_v1_sha256"]
         and launch.get("admission_v2_sha256") == go["admission_v2_sha256"]
         and launch.get("go_sha256") == go["go_sha256"]
         and launch.get("source_identity_v2") == go["source_identity"],
         "V2 launch/GO binding invalid")
    nonce = line(live_fd, 2.0)
    need(len(nonce) == 64 and all(c in b"0123456789abcdef" for c in nonce)
         and gate.sha_bytes(nonce) == launch.get("parent_nonce_sha256"),
         "live parent nonce invalid")
    need(line(live_fd, 2.0) == b"beat", "initial parent heartbeat missing")
    ready = {"schema": "human_asr_proxy_worker_ready_v2",
             "worker_pid": os.getpid(), "parent_pid": os.getppid(),
             "go_sha256": go["go_sha256"],
             "admission_v2_sha256": go["admission_v2_sha256"],
             "parent_nonce_sha256": gate.sha_bytes(nonce)}
    with (attempt / "worker_ready_v2.json").open("x") as out:
        out.write(json.dumps(ready, sort_keys=True, separators=(",", ":")) + "\n")
        out.flush()
        os.fsync(out.fileno())
    thread = threading.Thread(target=watch, args=(live_fd,), daemon=True)
    thread.start()
    old.run(attempt, admission)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt", type=Path, required=True)
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--live-fd", type=int, required=True)
    args = parser.parse_args()
    run(args.attempt, args.admission, args.live_fd)


if __name__ == "__main__":
    main()
