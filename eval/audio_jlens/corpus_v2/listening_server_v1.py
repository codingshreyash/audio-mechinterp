"""Loopback-only blind listening collector for the frozen Qwen v2 WAV queue.

The server never sends the private case map, expected words, WAV paths, or
previous answers to a browser. It performs no ASR or candidate-model inference.
Human responses are append-only evidence, not an automatic quality verdict.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import threading
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import make_qa_queue_v1 as queue_builder


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CLIP_ROOT = (HERE / "rendered_v2/clips").resolve()
PRIVATE = HERE / "QA_CASES_PROSPECTIVE_V1.json"
QUEUE = HERE / "LISTENING_QUEUE_PROSPECTIVE_V1.json"
MANIFEST = HERE / "rendered_v2/manifest.json"
BINDING = HERE / "rendered_v2/TECHNICAL_BINDING.json"
HTML = HERE / "listening_v1.html"
APP = HERE / "listening_v1.js"
PINS = {
    "private": (PRIVATE, "d9a08a61b76e8ab56ab3a9b2d72a4aedc5871555487c7a33b712097fe3206ed8"),
    "queue": (QUEUE, "19319ac394a24f8d742285d4bb3ecfd78020b1d131fa57def5b3362fd0b73873"),
    "manifest": (MANIFEST, "15395c7d132eb3f1c1a6b2b7d555146faa83be4ee2b77fba4f481cf70f26761d"),
    "binding": (BINDING, "858f8f68fce65ee60e328b1536b34608ced6b3e84d725f229460c6376e14af93"),
}
STAGES = ("primary", "secondary", "tertiary")
KINDS = ("initial", "correction", "adjudication")
SERVER_VERSION = "BlindListening/1"
OPAQUE = re.compile(r"[0-9a-f]{20}\Z")
REVIEWER = re.compile(r"[A-Za-z0-9_.-]{3,64}\Z")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def canonical(value: dict) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class BlindCorpus:
    def __init__(self, ordered_ids: list[str], cases: list[dict], root: Path,
                 clip_root: Path, provenance: dict):
        require(len(ordered_ids) == len(set(ordered_ids)) == len(cases),
                "queue and case population differ")
        self.ordered_ids = list(ordered_ids)
        self.cases = {case["opaque_id"]: case for case in cases}
        require(set(self.ordered_ids) == set(self.cases)
                and all(OPAQUE.fullmatch(ident) for ident in self.ordered_ids),
                "queue IDs or private map changed")
        self.root = root.resolve()
        self.clip_root = clip_root.resolve()
        self.provenance = dict(provenance)
        for ident in self.ordered_ids:
            self.audio_path(ident)

    def audio_path(self, ident: str) -> Path:
        require(ident in self.cases, "unknown audio ID")
        case = self.cases[ident]
        relative = Path(case["audio_path"])
        require(not relative.is_absolute(), "absolute private audio path")
        path = (self.root / relative).resolve()
        require(path.is_relative_to(self.clip_root) and path.is_file()
                and sha(path) == case["wav_sha256"], "audio source hash/path changed")
        return path

    @classmethod
    def frozen(cls) -> "BlindCorpus":
        for name, (path, expected) in PINS.items():
            require(path.is_file() and sha(path) == expected,
                    f"frozen {name} SHA-256 changed")
        rebuilt_private, rebuilt_queue = queue_builder.build()
        private = json.loads(PRIVATE.read_text())
        public = json.loads(QUEUE.read_text())
        require(private == rebuilt_private and public == rebuilt_queue,
                "blind queue does not reconstruct from frozen metadata")
        require(private["planned_clips"] == public["planned_clips"] == 480,
                "frozen queue must have 480 cases")
        provenance = {name + "_sha256": expected for name, (_path, expected) in PINS.items()}
        return cls(public["opaque_ids_in_order"], private["cases"], ROOT,
                   CLIP_ROOT, provenance)


class ResponseLog:
    def __init__(self, run_dir: Path, provenance: dict, *, create: bool):
        self.run_dir = run_dir.resolve()
        self.meta_path = self.run_dir / "run_meta.json"
        self.path = self.run_dir / "responses.jsonl"
        self.lock_path = self.run_dir / "server.lock"
        self.mutex = threading.Lock()
        expected = {**provenance, "server_sha256": sha(Path(__file__)),
                    "html_sha256": sha(HTML), "app_sha256": sha(APP)}
        if create:
            self.run_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
            meta_fd = os.open(self.meta_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(meta_fd, "w", encoding="utf-8") as stream:
                json.dump({"status": "blind_listening_collection_no_quality_verdict",
                           "started_utc": utc_now(), "server_version": SERVER_VERSION,
                           "provenance": expected}, stream, indent=2)
                stream.write("\n")
            os.close(os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600))
        else:
            require(self.meta_path.is_file() and self.path.is_file(),
                    "listening run is incomplete")
            meta = json.loads(self.meta_path.read_text())
            require(meta["status"] == "blind_listening_collection_no_quality_verdict"
                    and meta["server_version"] == SERVER_VERSION
                    and meta["provenance"] == expected,
                    "listening run source/provenance changed")
        require(not self.run_dir.stat().st_mode & 0o077
                and not self.meta_path.stat().st_mode & 0o077
                and not self.path.stat().st_mode & 0o077,
                "listening response directory/files are not private")
        lock_fd = os.open(self.lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        self.lock_stream = os.fdopen(lock_fd, "a+")
        try:
            fcntl.flock(self.lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.lock_stream.close()
            raise ValueError("another listening server owns this run") from exc
        self.records = self._read_verified()

    def close(self) -> None:
        fcntl.flock(self.lock_stream.fileno(), fcntl.LOCK_UN)
        self.lock_stream.close()

    def _read_verified(self) -> list[dict]:
        records: list[dict] = []
        previous = "0" * 64
        with self.path.open("rb") as stream:
            for raw in stream:
                require(raw.endswith(b"\n"), "truncated append-only response line")
                record = json.loads(raw)
                saved_hash = record.pop("record_sha256")
                require(record["sequence"] == len(records) + 1
                        and record["previous_sha256"] == previous
                        and hashlib.sha256(canonical(record)).hexdigest() == saved_hash,
                        "response log hash chain changed")
                record["record_sha256"] = saved_hash
                previous = saved_hash
                records.append(record)
        return records

    def append(self, submission: dict, allowed_ids: set[str],
               browser_user_agent: str = "direct_test_client_no_browser") -> dict:
        require(isinstance(submission, dict), "response must be an object")
        required = {"kind", "stage", "reviewer_id", "opaque_id", "middle_word",
                    "clear", "confidence", "play_count"}
        optional = {"note", "reason", "prior_sequence"}
        require(required <= set(submission) and set(submission) <= required | optional,
                "response field set changed")
        kind, stage, reviewer, ident = (submission[k] for k in
                                       ("kind", "stage", "reviewer_id", "opaque_id"))
        require(kind in KINDS and stage in STAGES and ident in allowed_ids
                and isinstance(reviewer, str) and REVIEWER.fullmatch(reviewer) is not None,
                "response identity/stage invalid")
        answer = submission["middle_word"]
        require(isinstance(answer, str) and 0 < len(answer.strip()) <= 120
                and "\n" not in answer and "\r" not in answer,
                "middle-word answer invalid")
        require(type(submission["clear"]) is bool
                and type(submission["confidence"]) is int
                and 1 <= submission["confidence"] <= 5
                and type(submission["play_count"]) is int
                and 1 <= submission["play_count"] <= 100,
                "listening response values invalid")
        note = submission.get("note", "")
        reason = submission.get("reason", "")
        require(isinstance(note, str) and len(note) <= 500
                and isinstance(reason, str) and len(reason) <= 500,
                "response note/reason invalid")
        require(isinstance(browser_user_agent, str)
                and 1 <= len(browser_user_agent.strip()) <= 500
                and "\n" not in browser_user_agent and "\r" not in browser_user_agent,
                "browser user-agent/version missing or invalid")
        with self.mutex:
            self.records = self._read_verified()
            initials = [r for r in self.records if r["kind"] == "initial"]
            owners = {r["stage"]: r["reviewer_id"] for r in initials}
            if kind == "initial":
                require("prior_sequence" not in submission and not reason,
                        "initial response cannot revise a prior answer")
                require(not any(r["stage"] == stage and r["opaque_id"] == ident
                                for r in initials), "duplicate initial response")
                require(stage not in owners or owners[stage] == reviewer,
                        "review stage has a different blinded listener")
                require(not any(r["opaque_id"] == ident and r["reviewer_id"] == reviewer
                                for r in initials), "same listener cannot review a clip twice")
                if stage == "secondary":
                    require(any(r["stage"] == "primary" and r["opaque_id"] == ident
                                for r in initials), "secondary precedes primary")
                if stage == "tertiary":
                    require(any(r["stage"] == "secondary" and r["opaque_id"] == ident
                                for r in initials), "tertiary precedes secondary")
            else:
                require(kind != "adjudication" or stage == "tertiary",
                        "adjudication must be a tertiary record")
                prior = submission.get("prior_sequence")
                require(type(prior) is int and 1 <= prior <= len(self.records)
                        and reason.strip(), "correction/adjudication needs a prior row and reason")
                referenced = self.records[prior - 1]
                require(referenced["opaque_id"] == ident and referenced["stage"] == stage,
                        "correction/adjudication references another clip or stage")
                if kind == "correction":
                    require(referenced["reviewer_id"] == reviewer,
                            "correction must come from original listener")
                else:
                    require(referenced["kind"] == "initial"
                            and any(r["stage"] == "primary" and r["opaque_id"] == ident
                                    for r in initials)
                            and any(r["stage"] == "secondary" and r["opaque_id"] == ident
                                    for r in initials)
                            and reviewer not in {r["reviewer_id"] for r in initials
                                                 if r["opaque_id"] == ident},
                            "adjudication requires an independent third listener")
            record = {k: submission[k] for k in required}
            record.update({"note": note, "reason": reason,
                           "prior_sequence": submission.get("prior_sequence"),
                           "sequence": len(self.records) + 1,
                           "timestamp_utc": utc_now(),
                           "browser_user_agent": browser_user_agent,
                           "previous_sha256": (self.records[-1]["record_sha256"]
                                               if self.records else "0" * 64)})
            record["record_sha256"] = hashlib.sha256(canonical(record)).hexdigest()
            with self.path.open("ab") as stream:
                stream.write(canonical(record) + b"\n")
                stream.flush()
                os.fsync(stream.fileno())
            self.records.append(record)
            return {"status": "recorded", "sequence": record["sequence"]}

    def completed(self, stage: str, reviewer: str) -> list[str]:
        require(stage in STAGES and REVIEWER.fullmatch(reviewer) is not None,
                "progress identity invalid")
        with self.mutex:
            self.records = self._read_verified()
            return [r["opaque_id"] for r in self.records
                    if r["kind"] == "initial" and r["stage"] == stage
                    and r["reviewer_id"] == reviewer]


def handler_for(corpus: BlindCorpus, responses: ResponseLog):
    # The bytes served remain exactly those bound in the run metadata.
    html_bytes = HTML.read_bytes()
    app_bytes = APP.read_bytes()
    provenance = json.loads(responses.meta_path.read_text())["provenance"]
    require(hashlib.sha256(html_bytes).hexdigest() == provenance["html_sha256"]
            and hashlib.sha256(app_bytes).hexdigest() == provenance["app_sha256"],
            "listening interface changed during initialization")

    class Handler(BaseHTTPRequestHandler):
        server_version = SERVER_VERSION
        sys_version = ""

        def _headers(self, status: int, content_type: str, length: int) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(length))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy",
                             "default-src 'self'; script-src 'self'; media-src 'self'; "
                             "connect-src 'self'; style-src 'self' 'unsafe-inline'; "
                             "object-src 'none'; base-uri 'none'")
            self.end_headers()

        def _send(self, status: int, payload: bytes, content_type: str) -> None:
            self._headers(status, content_type, len(payload))
            self.wfile.write(payload)

        def _error(self, status: int) -> None:
            self._send(status, b'{"error":"request rejected"}', "application/json")

        def _same_origin(self) -> bool:
            host = self.headers.get("Host", "")
            if not re.fullmatch(r"(?:127[.]0[.]0[.]1|localhost):[0-9]{1,5}", host):
                return False
            origin = self.headers.get("Origin")
            return (origin is None or origin == "http://" + host) and (
                self.headers.get("Sec-Fetch-Site") != "cross-site")

        def do_GET(self) -> None:
            if not self._same_origin():
                self._error(HTTPStatus.FORBIDDEN)
                return
            request = urlsplit(self.path)
            path = request.path
            if path == "/":
                self._send(HTTPStatus.OK, html_bytes, "text/html; charset=utf-8")
            elif path == "/app.js":
                self._send(HTTPStatus.OK, app_bytes, "text/javascript; charset=utf-8")
            elif path == "/api/queue":
                self._send(HTTPStatus.OK, canonical({
                    "opaque_ids_in_order": corpus.ordered_ids,
                    "planned_clips": len(corpus.ordered_ids)}), "application/json")
            elif path == "/api/progress":
                params = parse_qs(request.query, strict_parsing=True)
                try:
                    require(set(params) == {"stage", "reviewer_id"}, "progress query invalid")
                    done = responses.completed(params["stage"][0], params["reviewer_id"][0])
                except (ValueError, KeyError, IndexError):
                    self._error(HTTPStatus.BAD_REQUEST)
                    return
                self._send(HTTPStatus.OK, canonical({"completed_ids": done}),
                           "application/json")
            elif path.startswith("/audio/"):
                ident = path.removeprefix("/audio/")
                if not OPAQUE.fullmatch(ident):
                    self._error(HTTPStatus.NOT_FOUND)
                    return
                try:
                    data = corpus.audio_path(ident).read_bytes()
                except ValueError:
                    self._error(HTTPStatus.CONFLICT)
                    return
                self._send(HTTPStatus.OK, data, "audio/wav")
            else:
                self._error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            if not self._same_origin() or self.path != "/api/response":
                self._error(HTTPStatus.FORBIDDEN)
                return
            if self.headers.get("Content-Type") != "application/json":
                self._error(HTTPStatus.UNSUPPORTED_MEDIA_TYPE)
                return
            try:
                length = int(self.headers.get("Content-Length", "-1"))
                require(0 < length <= 4096, "response length invalid")
                payload = json.loads(self.rfile.read(length))
                browser = self.headers.get("User-Agent", "")
                require(browser and browser != "direct_test_client_no_browser",
                        "browser user-agent/version missing")
                result = responses.append(payload, set(corpus.ordered_ids), browser)
            except (ValueError, KeyError, TypeError, json.JSONDecodeError):
                self._error(HTTPStatus.CONFLICT)
                return
            self._send(HTTPStatus.CREATED, canonical(result), "application/json")

        def log_message(self, _format: str, *_args) -> None:
            # No path, IDs, answers, or private fields in server logs.
            return

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--serve", action="store_true")
    parser.add_argument("--run-dir", type=Path,
                        default=HERE / "listening_run_v1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    corpus = BlindCorpus.frozen()
    if args.validate_only:
        print(json.dumps({"status": "validated_blind_audio_sources_no_listening",
                          "clips": len(corpus.ordered_ids),
                          "private_sha256": PINS["private"][1],
                          "queue_sha256": PINS["queue"][1]}, indent=2))
        return
    require(0 <= args.port <= 65535, "port outside TCP range")
    responses = ResponseLog(args.run_dir, corpus.provenance,
                            create=not args.run_dir.exists())
    try:
        with ThreadingHTTPServer(("127.0.0.1", args.port),
                                 handler_for(corpus, responses)) as server:
            print(f"Blinded listening service: http://127.0.0.1:{server.server_port}/")
            print("Responses are local append-only records; no quality verdict is produced.")
            server.serve_forever(poll_interval=.5)
    finally:
        responses.close()


if __name__ == "__main__":
    main()
