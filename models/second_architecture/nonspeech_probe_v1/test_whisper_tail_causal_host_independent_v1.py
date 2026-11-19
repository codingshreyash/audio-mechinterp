"""Independent, no-weight contract tests for the battery-aware causal host gate."""
from __future__ import annotations

import plistlib
import unittest
from unittest.mock import patch

from . import whisper_tail_causal_host_v1 as host


def fixture(*, percent: int = 99, external: bool = False,
            lid_closed: bool = False, thermal_ok: bool = True) -> dict[str, str]:
    return {
        "power": ("Now drawing from 'AC Power'\n" if external else
                  "Now drawing from 'Battery Power'\n")
                 + f" -InternalBattery-0 (id=1)\t{percent}%; discharging; 7:00 remaining present: true\n",
        "battery": plistlib.dumps([{
            "CurrentCapacity": percent, "MaxCapacity": 100,
            "AtCriticalLevel": False, "ExternalConnected": external,
        }], fmt=plistlib.FMT_XML).decode(),
        "clamshell": plistlib.dumps([{
            "AppleClamshellState": lid_closed,
        }], fmt=plistlib.FMT_XML).decode(),
        "thermal": ("Note: No thermal warning level has been recorded\n"
                    "Note: No performance warning level has been recorded\n"
                    if thermal_ok else "Note: Thermal warning level 1 has been recorded\n"),
    }


class HostGateIndependentTests(unittest.TestCase):
    def test_start_boundary_and_ac(self) -> None:
        self.assertEqual(host.parse(fixture(percent=80), initial=True)["battery_percent"], 80)
        self.assertTrue(host.parse(fixture(percent=80, external=True), initial=True)
                        ["external_connected"])
        with self.assertRaises(ValueError):
            host.parse(fixture(percent=79), initial=True)

    def test_monitor_floor(self) -> None:
        self.assertEqual(host.parse(fixture(percent=71), initial=False)["battery_percent"], 71)
        for percent in (70, 69):
            with self.subTest(percent=percent), self.assertRaises(ValueError):
                host.parse(fixture(percent=percent), initial=False)

    def test_fail_closed_power_lid_thermal_and_unknown(self) -> None:
        cases = []
        outputs = fixture()
        outputs["power"] = "battery status unknown"
        cases.append(outputs)
        outputs = fixture()
        outputs["power"] = outputs["power"].replace("Battery Power", "Unknown Power")
        cases.append(outputs)
        outputs = fixture()
        outputs["battery"] = "not plist"
        cases.append(outputs)
        cases.append(fixture(lid_closed=True))
        cases.append(fixture(thermal_ok=False))
        outputs = fixture()
        outputs["thermal"] = ""
        cases.append(outputs)
        outputs = fixture()
        outputs["battery"] = plistlib.dumps([{
            "CurrentCapacity": 99, "MaxCapacity": 100,
            "AtCriticalLevel": True, "ExternalConnected": False,
        }]).decode()
        cases.append(outputs)
        for index, outputs in enumerate(cases):
            with self.subTest(case=index), self.assertRaises((ValueError, IndexError,
                                                               plistlib.InvalidFileException)):
                host.parse(outputs, initial=False)

    def test_readings_must_agree(self) -> None:
        outputs = fixture(percent=99)
        outputs["power"] = outputs["power"].replace("99%;", "95%;")
        with self.assertRaises(ValueError):
            host.parse(outputs, initial=True)

        outputs = fixture(percent=99)
        outputs["power"] = outputs["power"].replace("Battery Power", "AC Power")
        with self.assertRaises(ValueError):
            host.parse(outputs, initial=True)

        outputs = fixture(percent=99, external=True)
        outputs["power"] = outputs["power"].replace("AC Power", "Battery Power")
        with self.assertRaises(ValueError):
            host.parse(outputs, initial=True)

    def test_probe_rejects_command_failure(self) -> None:
        with patch.object(host, "_command", side_effect=ValueError("unreadable")):
            with self.assertRaises(ValueError):
                host.quick_probe(initial=True)

    def test_cadence_boundaries_and_floor(self) -> None:
        def sample(t: float, percent: int) -> dict:
            return {"monotonic_seconds": t, "battery_percent": percent,
                    "lid_open": True, "thermal_ok": True}

        host.validate_cadence([sample(10.0, 80), sample(24.0, 71)],
                              start=0.0, end=30.0)
        bad = [
            ([sample(16.0, 80)], 0.0, 20.0),
            ([sample(10.0, 80), sample(26.0, 71)], 0.0, 30.0),
            ([sample(10.0, 80), sample(24.0, 71)], 0.0, 40.0),
            ([sample(10.0, 80), sample(24.0, 70)], 0.0, 30.0),
        ]
        for samples, start, end in bad:
            with self.subTest(samples=samples), self.assertRaises(ValueError):
                host.validate_cadence(samples, start=start, end=end)


if __name__ == "__main__":
    unittest.main()
