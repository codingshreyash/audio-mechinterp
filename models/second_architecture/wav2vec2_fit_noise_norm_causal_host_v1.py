"""Standalone AC/lid/power-log evidence gate for one local CPU run."""
from __future__ import annotations

import gzip
import hashlib
import plistlib
import re
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

COMMANDS = {
    "power": ["pmset", "-g", "batt"],
    "clamshell": ["ioreg", "-a", "-r", "-c", "IOPMrootDomain", "-d", "1"],
    "battery": ["ioreg", "-a", "-r", "-c", "AppleSmartBattery", "-d", "1"],
    "power_log": ["pmset", "-g", "log"],
}
EVENT = re.compile(r"^(\d{4}-\d\d-\d\d)\s+(\d\d:\d\d:\d\d)\s+([+-]\d{4})\s+([^\t]+)\t", re.MULTILINE)
TERMINAL = re.compile(
    r"^(\d{4}-\d\d-\d\d)\s+(\d\d:\d\d:\d\d)\s+([+-]\d{4})\s+: Showing all currently held IOKit power assertions\s*$",
    re.MULTILINE)


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def command(argv: list[str]) -> str:
    result = subprocess.run(argv, capture_output=True, text=True, check=False, timeout=90)
    require(result.returncode == 0 and not result.stderr, f"host command failed: {argv}")
    return result.stdout


def timestamp(match) -> datetime:
    return datetime.strptime(" ".join(match.group(1, 2, 3)), "%Y-%m-%d %H:%M:%S %z")


def validate_log(log: str, start: datetime, end: datetime,
                 collected: datetime) -> None:
    events = list(EVENT.finditer(log))
    terminal = list(TERMINAL.finditer(log))
    require(events and terminal, "power log has no parseable coverage")
    require(min(timestamp(m) for m in events) <= start
            and timestamp(terminal[-1]) >= end-timedelta(seconds=1)
            and timestamp(terminal[-1]) >= collected-timedelta(seconds=5)
            and timestamp(terminal[-1]) <= collected+timedelta(seconds=1),
            "power log stale or truncated")
    suspect = [m.group(0) for m in events if start <= timestamp(m) <= end
               and (m.group(4).strip() in ("Sleep", "DarkWake", "Wake")
                    or "Thermal" in m.group(4))]
    require(not suspect, "sleep/wake/thermal event during required window")


def validate_power(probe: dict) -> None:
    outputs = probe["outputs"]
    require(probe.get("commands") == COMMANDS
            and "Now drawing from 'AC Power'" in outputs["power"]
            and plistlib.loads(outputs["battery"].encode())[0].get("ExternalConnected") is True
            and plistlib.loads(outputs["clamshell"].encode())[0].get("AppleClamshellState") is False,
            "host AC/lid state invalid")


def probe(*, run_start: datetime | None = None) -> dict:
    outputs = {name: command(argv) for name, argv in COMMANDS.items()}
    collected = datetime.now(timezone.utc)
    result = {"collected_utc": collected.isoformat(), "commands": COMMANDS,
              "outputs": outputs}
    validate_power(result)
    start = run_start or collected-timedelta(minutes=10)
    validate_log(outputs["power_log"], start, collected, collected)
    return result


def archive(attempt: Path, evidence: dict, name: str) -> None:
    require(name in ("pre", "post"), "host archive name invalid")
    raw = evidence["outputs"].pop("power_log").encode()
    require(len(raw) <= 128*1024**2, "raw host log cap exceeded")
    packed = gzip.compress(raw, mtime=0)
    require(len(packed) <= 16*1024**2, "compressed host log cap exceeded")
    path = attempt / f"{name}_power_log.txt.gz"
    with path.open("xb") as stream:
        stream.write(packed)
        stream.flush()
    evidence["power_log_archive"] = {
        "file": path.name,
        "compressed_sha256": hashlib.sha256(packed).hexdigest(),
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "raw_bytes": len(raw),
    }


def load_archive(attempt: Path, evidence: dict, name: str) -> str:
    item = evidence.get("power_log_archive")
    require(isinstance(item, dict) and item.get("file") == f"{name}_power_log.txt.gz",
            "host log reference invalid")
    path = attempt / item["file"]
    require(path.is_file() and not path.is_symlink()
            and path.stat().st_size <= 16*1024**2
            and hashlib.sha256(path.read_bytes()).hexdigest() == item["compressed_sha256"],
            "host log archive changed")
    raw = gzip.decompress(path.read_bytes())
    require(len(raw) == item["raw_bytes"] and len(raw) <= 128*1024**2
            and hashlib.sha256(raw).hexdigest() == item["raw_sha256"],
            "host raw log changed")
    return raw.decode()
