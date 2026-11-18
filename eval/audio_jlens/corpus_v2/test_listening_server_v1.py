"""Synthetic loopback HTTP and privacy checks; no human response or ASR run."""

from __future__ import annotations

import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
import wave
from pathlib import Path

import listening_server_v1 as listening


class BlindListeningServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frozen = listening.BlindCorpus.frozen()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        clips = root / "clips"
        clips.mkdir()
        self.ident = "a" * 20
        self.secret_name = "secret_target_otter.wav"
        self.wav = clips / self.secret_name
        with wave.open(str(self.wav), "wb") as stream:
            stream.setnchannels(1)
            stream.setsampwidth(2)
            stream.setframerate(16000)
            stream.writeframes(b"\x00\x00\x01\x00" * 100)
        case = {"opaque_id": self.ident, "target_word": "otter",
                "audio_path": "clips/" + self.secret_name,
                "wav_sha256": listening.sha(self.wav)}
        self.corpus = listening.BlindCorpus([self.ident], [case], root, clips,
                                            {"synthetic": "only"})
        self.log = listening.ResponseLog(root / "run", self.corpus.provenance, create=True)
        self.addCleanup(self.log.close)
        self.server = listening.ThreadingHTTPServer(
            ("127.0.0.1", 0), listening.handler_for(self.corpus, self.log))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.thread.join, 2)
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.base = f"http://127.0.0.1:{self.server.server_port}"

    def request(self, path: str, payload: dict | None = None,
                origin: str | None = None) -> tuple[int, bytes, object]:
        data = None if payload is None else json.dumps(payload).encode()
        headers = {"Content-Type": "application/json"} if data is not None else {}
        if origin is not None:
            headers["Origin"] = origin
        req = urllib.request.Request(self.base + path, data=data, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=3) as response:
                return response.status, response.read(), response.headers
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read(), exc.headers

    def response(self, **changes) -> dict:
        row = {"kind": "initial", "stage": "primary", "reviewer_id": "listener_a",
               "opaque_id": self.ident, "middle_word": "otter", "clear": True,
               "confidence": 5, "play_count": 1, "note": ""}
        row.update(changes)
        return row

    def test_frozen_480_case_preflight_is_byte_bound(self):
        self.assertEqual(len(self.frozen.ordered_ids), 480)
        self.assertEqual(len(self.frozen.cases), 480)
        self.assertEqual(list(self.frozen.provenance)[0], "private_sha256")

    def test_only_opaque_queue_and_audio_are_exposed(self):
        exposed = []
        for path in ("/", "/app.js", "/api/queue", "/audio/" + self.ident,
                     "/QA_CASES_PROSPECTIVE_V1.json", "/clips/" + self.secret_name):
            status, body, headers = self.request(path)
            exposed.append(body)
            if path.startswith("/audio/"):
                self.assertEqual((status, headers["Content-Type"]), (200, "audio/wav"))
                self.assertEqual(body, self.wav.read_bytes())
                self.assertNotIn("filename", str(headers).lower())
            elif path.startswith("/QA_") or path.startswith("/clips/"):
                self.assertEqual(status, 404)
            else:
                self.assertEqual(status, 200)
        joined = b"\n".join(exposed)
        self.assertNotIn(self.secret_name.encode(), joined)
        self.assertNotIn(b"target_word", joined)
        self.assertNotIn(b"paired_alternative", joined)
        self.assertNotIn(b"audio_path", joined)
        self.assertNotIn(b"otter", joined)
        self.assertIn(self.ident.encode(), joined)

    def test_initial_duplicate_and_append_only_correction(self):
        first = self.response()
        status, body, _ = self.request("/api/response", first)
        self.assertEqual((status, json.loads(body)["sequence"]), (201, 1))
        status, _body, _ = self.request("/api/response", first)
        self.assertEqual(status, 409)
        status, _body, _ = self.request("/api/response", self.response(
            stage="secondary", reviewer_id="listener_a"))
        self.assertEqual(status, 409)
        status, body, _ = self.request("/api/response", self.response(
            stage="secondary", reviewer_id="listener_b", middle_word="other"))
        self.assertEqual((status, json.loads(body)["sequence"]), (201, 2))
        correction = self.response(kind="correction", middle_word="otters",
                                   prior_sequence=1, reason="Typographical correction")
        status, body, _ = self.request("/api/response", correction)
        self.assertEqual((status, json.loads(body)["sequence"]), (201, 3))
        records = self.log._read_verified()
        self.assertEqual([r["kind"] for r in records],
                         ["initial", "initial", "correction"])
        self.assertEqual(records[0]["middle_word"], "otter")
        self.assertEqual(records[2]["prior_sequence"], 1)
        progress = self.request("/api/progress?stage=primary&reviewer_id=listener_a")
        self.assertEqual(json.loads(progress[1])["completed_ids"], [self.ident])
        self.assertNotIn(b"otter", progress[1])

    def test_cross_origin_and_changed_audio_fail_closed(self):
        status, _body, _ = self.request("/api/queue", origin="https://elsewhere.example")
        self.assertEqual(status, 403)
        with self.wav.open("ab") as stream:
            stream.write(b"tamper")
        status, body, _ = self.request("/audio/" + self.ident)
        self.assertEqual(status, 409)
        self.assertNotIn(b"tamper", body)

    def test_changed_response_log_fails_chain_verification(self):
        self.assertEqual(self.request("/api/response", self.response())[0], 201)
        raw = self.log.path.read_bytes()
        self.assertIn(b"otter", raw)
        self.log.path.write_bytes(raw.replace(b"otter", b"other", 1))
        with self.assertRaisesRegex(ValueError, "hash chain"):
            self.log._read_verified()
        self.assertEqual(self.request("/api/response", self.response(
            kind="correction", prior_sequence=1, reason="Correction"))[0], 409)


if __name__ == "__main__":
    unittest.main()
