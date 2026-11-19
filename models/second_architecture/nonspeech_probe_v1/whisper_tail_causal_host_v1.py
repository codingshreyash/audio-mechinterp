"""Battery-aware local host gate for one bounded Whisper causal attempt."""
from __future__ import annotations

import plistlib
import re
import subprocess
import time
from datetime import datetime, timezone
from typing import Any

from models.second_architecture import wav2vec2_fit_noise_norm_causal_host_v1 as old

START_PERCENT = 80
STOP_PERCENT = 70
MAX_PROBE_GAP_SECONDS = 15.0
QUICK_COMMANDS = {
    "power": ("pmset", "-g", "batt"),
    "battery": ("ioreg", "-a", "-r", "-c", "AppleSmartBattery", "-d", "1"),
    "clamshell": ("ioreg", "-a", "-r", "-c", "IOPMrootDomain", "-d", "1"),
    "thermal": ("pmset", "-g", "therm"),
}


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def _command(argv: tuple[str, ...], *, timeout: float = 3.0) -> str:
    result = subprocess.run(argv, text=True, capture_output=True, timeout=timeout)
    require(result.returncode == 0 and not result.stderr,
            f"host quick probe failed: {argv[0]} {argv[1:]}")
    return result.stdout


def parse(outputs: dict[str, str], *, initial: bool) -> dict[str, Any]:
    """Reject unreadable power/lid/thermal state; avoid archiving battery serials."""
    match = re.search(r"\b(\d{1,3})%;", outputs["power"])
    require(bool(match), "battery percent unavailable")
    pmset_percent = int(match.group(1))
    battery = plistlib.loads(outputs["battery"].encode())[0]
    lid = plistlib.loads(outputs["clamshell"].encode())[0]
    percent = battery.get("CurrentCapacity")
    maximum = battery.get("MaxCapacity")
    require(type(percent) is int and type(maximum) is int and maximum == 100
            and 0 <= percent <= 100 and abs(percent-pmset_percent) <= 2,
            "battery readings disagree or unavailable")
    require(battery.get("AtCriticalLevel") is False
            and lid.get("AppleClamshellState") is False,
            "critical battery or closed lid")
    pmset_external = "'AC Power'" in outputs["power"]
    require(pmset_external == (battery.get("ExternalConnected") is True)
            and battery.get("ExternalConnected") in (True, False),
            "external power readings disagree or unavailable")
    thermal = outputs["thermal"]
    require("No thermal warning level has been recorded" in thermal
            and "No performance warning level has been recorded" in thermal,
            "thermal state unknown or warning")
    threshold = START_PERCENT if initial else STOP_PERCENT
    require(percent >= threshold if initial else percent > threshold,
            "battery below causal-run floor")
    return {"battery_percent": percent,"pmset_percent":pmset_percent,
            "external_connected":battery.get("ExternalConnected") is True,
            "lid_open":True,"thermal_ok":True,
            "power_text":outputs["power"],"thermal_text":thermal}


def quick_probe(*, initial: bool = False) -> dict[str, Any]:
    started = time.monotonic()
    outputs = {name:_command(argv) for name,argv in QUICK_COMMANDS.items()}
    result = parse(outputs,initial=initial)
    result["collected_utc"] = datetime.now(timezone.utc).isoformat()
    result["monotonic_seconds"] = time.monotonic()
    require(result["monotonic_seconds"]-started <= MAX_PROBE_GAP_SECONDS,
            "host quick probe exceeded cadence")
    return result


def validate_cadence(samples: list[dict], *, start: float, end: float) -> None:
    require(bool(samples) and 0 <= samples[0]["monotonic_seconds"]-start
            <= MAX_PROBE_GAP_SECONDS and 0 <= end-samples[-1]["monotonic_seconds"]
            <= MAX_PROBE_GAP_SECONDS and all(
                0 < b["monotonic_seconds"]-a["monotonic_seconds"]
                <= MAX_PROBE_GAP_SECONDS for a,b in zip(samples,samples[1:])),
            "battery/thermal probe coverage gap")
    for index,sample in enumerate(samples):
        require(sample["battery_percent"] >= START_PERCENT if index==0
                else sample["battery_percent"] > STOP_PERCENT,
                "saved battery floor violated")
        require(sample["lid_open"] is True and sample["thermal_ok"] is True,
                "saved lid or thermal warning")


def validate_saved_text(sample:dict)->None:
    """Recheck archived pmset/thermal text against the sanitized live summary."""
    power_text=sample.get("power_text")
    thermal_text=sample.get("thermal_text")
    match=re.search(r"\b(\d{1,3})%;",power_text) if isinstance(power_text,str) else None
    require(bool(match) and int(match.group(1))==sample.get("pmset_percent")
            and abs(sample["battery_percent"]-sample["pmset_percent"])<=2
            and sample.get("external_connected") is ("'AC Power'" in power_text)
            and isinstance(thermal_text,str)
            and "No thermal warning level has been recorded" in thermal_text
            and "No performance warning level has been recorded" in thermal_text,
            "saved host sample text/summary invalid")


def power_log(start: datetime, end: datetime) -> str:
    log = old.command(old.COMMANDS["power_log"])
    old.validate_log(log,start,end,datetime.now(timezone.utc))
    return log
