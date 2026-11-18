# Local blinded listening interface

This interface supports the prospective [480-WAV intelligibility protocol](ASR_LISTENING_QA_PROTOCOL_PROSPECTIVE_V1.md). It collects observations only. **No listening responses, ASR transcripts, or quality verdict have been produced by the code and tests in this package.** A real human who has not seen the internal case map must supply each listening response.

## Start and resume

From `eval/audio_jlens/corpus_v2`, run:

```sh
python3 listening_server_v1.py --validate-only
python3 listening_server_v1.py --serve --run-dir listening_run_v1 --port 8765
```

Open `http://127.0.0.1:8765/` in a local browser. The server binds only to loopback. The first launch creates a private run directory. Restart the same command to resume: the code rehashes the 480 WAVs and verifies the queue, interface files, run metadata, and entire response hash chain before accepting another row. A second server cannot own the same run at the same time. Do not expose the private case map, repository, or response directory to listeners; this local-only service does not provide remote authentication.

The listener enters an ID, selects first, second, or third review, plays the clip, and enters one heard middle word, clarity, and confidence. The browser receives only opaque IDs and audio bytes from `/audio/{opaque_id}`. Its URL and response headers do not contain the source filename or expected word. The private map stays server-side; no static file directory is served. The public queue remains in its deterministic shuffled order.

Initial answers are unique for each clip and review stage. The same listener cannot provide a second independent initial answer for that clip. A correction or adjudication is a **new** row with a prior record number and a reason; it cannot overwrite the first answer. The response file is append-only JSONL with a numbered SHA-256 chain and private file permissions. The interface records the browser's play count; it is a self-reported interaction count, **not proof that a human understood the audio**.

The response log at `listening_run_v1/responses.jsonl` contains what people heard and should be handled as research data. The source and queue hashes are in `run_meta.json`. Retain that run directory unchanged for the later independent QA auditor; a truncated or changed response line fails on resume. The final quality rule also needs two complete, independently pinned ASR passes and any second/third human reviews specified by the protocol. This interface alone never authorizes Qwen model inference.

## Local verification

```sh
python3 -m unittest -v test_listening_server_v1.py
node --check listening_v1.js
```

The HTTP tests create synthetic WAVs and synthetic response rows in temporary directories. They verify that labels and filenames are absent from served pages and JSON, duplicate first answers fail, corrections append, cross-origin requests fail, changed audio is refused, and a changed response log fails its hash chain. They do not submit actual corpus responses.
