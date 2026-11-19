"""FIT-only V2 source/runner for the acoustic-noise pilot.

The no-weight ``preflight`` and ``validate-only`` modes do not load models.
``run`` requires a separately committed independent GO receipt. See the V2
protocol; do not treat this engineering pilot as a security study.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
import wave
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PROTOCOL = HERE / "ACOUSTIC_FIT_NOISE_PILOT_PROSPECTIVE_V2.md"
MANIFEST = ROOT / "eval/audio_jlens/corpus_v1/rendered/manifest.json"
MANIFEST_SHA = "adf8cd02d24a43ef576bcb993dc54bd2527440a62798e12fc788def3c063971f"
WORDS = ("silver", "teal")
ENGINES = ("piper", "kokoro")
ARMS = ("clean", "clean_repeat", "full_noise", "word_noise", "outside_noise")
SEED_BASE = 60520261118
SNR_DB = 20.0
MAX_SECONDS = 1800
MAX_RSS_BYTES = 4 * 1024**3
MAX_OUTPUT_BYTES = 32 * 1024**2
EXPECTED = {
    "piper": {
        "slot": (30527, 49847), "frames": 86876,
        "noise": "c10e891886b119df5982bc7fb1f6795fe4963a35fca910c3dd5b7cf3d778448d",
        "silver": ("fe554972f68b470fb0f4a1c844b874e70758279c281bba47e963792bef2634a9", "79fc004685c31d196182f02deee103ec73b031917e88111338db853739e6cdc6", "7bcebcf0cb5bbe8d5555d5c31f790bd0a0ecea21f4ab7356e2cc3ee26ac538c4", "d13023361ab5a4cdd48becf8de5821ebe9a4f2e9eec02975a9690188bef06eb8", "7a80e54c5ba4c4746bb86b19c0b72047fd810e8d1a38b177fdc93d89beb641aa"),
        "teal": ("112ed36abf5fb2be1ce77dcfffa9d1bb6deba10b4ecb2d14fc5960b8106626e9", "dde8bdd8bb927aab4c4cc60925ba9a518e59d0d18d3995e36bffb6b8f331dd85", "c7d1ac5d7fc16c1a80efeaf2b853c22d412ead885292982b54cf481c6a0ab9a1", "48d6e7f316df523f20a097357a3fcb198baaaeed878e5c56cf44338a61c75d3b", "8fc55fa944178032334775a2b20b5413f09c1720d7458332efc6c596179e9e38"),
    },
    "kokoro": {
        "slot": (33920, 57120), "frames": 91840,
        "noise": "b010976dc8ef9db6bbbd6cccbe7a053cfe1ef0b7dac2b99a3b7e8be3dac44eb5",
        "silver": ("bcad702fa8b55c453286cb0e83ee3336a4bb6ade5e5ac9d4a42dc82c1a97c26e", "96bff0e029bd9f033b76e84258bc8ea75fe0f8c23bf965fbfe8f68a673f16386", "c431301ee124b9cc14fd256beee675a994f3eab8ebf931775f56a37ff20f61f8", "6573844e6acf720b1eedd0086060e81784ab552d0c1a35b24c3fc7f93eb9c032", "8ff8b229c397cd50c6d42d1c3a3af77b1dea4ada37d6ca6616359296ce4ef753"),
        "teal": ("5038dbe167ae2ba6c9317f74c17bd9bb8ef7d07bbdee9da49f6dc0731e119365", "84e07ded47cb6410b0840e7d502f10810018319b5d55b50fe22aaed890848006", "8d4fc06d0658e99190a97f61ff4ac0cb9a8c9c38073cbdf6af672f235f32d161", "f9a48dcfb965605774aec3ec375a9dacc33fe7b4ed95153430f116122ebedc63", "1ca226bef3e45e57fb04817b41364f2bff708eaefdba365ede6a9c714c5743fb"),
    },
}


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def source_commit(path: Path) -> str:
    rel = str(path.relative_to(ROOT))
    for args in (("git", "diff", "--quiet", "--", rel),
                 ("git", "diff", "--cached", "--quiet", "--", rel)):
        if subprocess.run(args, cwd=ROOT, check=False).returncode != 0:
            raise ValueError(f"dirty source: {rel}")
    commit = subprocess.check_output(("git", "log", "-1", "--format=%H", "--", rel), cwd=ROOT, text=True).strip()
    if len(commit) != 40:
        raise ValueError(f"uncommitted source: {rel}")
    return commit


def stimuli():
    import numpy as np

    if np.__version__ != "2.3.5":
        raise ValueError("NumPy version differs from frozen transform")
    if sha_file(MANIFEST) != MANIFEST_SHA:
        raise ValueError("FIT manifest changed")
    manifest = json.loads(MANIFEST.read_text())
    result = {}
    for engine in ENGINES:
        spec = EXPECTED[engine]
        rows = [r for r in manifest["pairs"] if r["pair_id"] == "color_05"
                and r["phrasing_id"] == "fit_chosen" and r["engine"] == engine]
        if len(rows) != 1:
            raise ValueError(f"missing/duplicate FIT pair: {engine}")
        pair = rows[0]
        slot = (pair["word_slot_start_sample"], pair["word_slot_end_sample_exclusive"])
        if pair["split"] != "fit_pool" or pair["map_fit"] is not True or pair["words"] != list(WORDS) or slot != spec["slot"]:
            raise ValueError("FIT pair identity/slot changed")
        originals = {}
        for word in WORDS:
            output = pair["outputs"][word]
            expected = spec[word]
            path = ROOT / output["path"]
            if path.is_symlink() or not path.is_file() or sha_file(path) != expected[0] or output["file_sha256"] != expected[0]:
                raise ValueError(f"source WAV changed: {engine}/{word}")
            with wave.open(str(path), "rb") as wav:
                format_tuple = (wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getnframes())
                if format_tuple != (1, 2, 16000, spec["frames"]):
                    raise ValueError("WAV format changed")
                raw = wav.readframes(wav.getnframes())
            if sha_bytes(raw) != expected[1] or output["pcm_int16_sha256"] != expected[1]:
                raise ValueError("source PCM changed")
            originals[word] = np.frombuffer(raw, dtype="<i2").astype(np.int64)
        start, end = slot
        a, b = (originals[w] for w in WORDS)
        if not np.array_equal(a[:start], b[:start]) or not np.array_equal(a[end:], b[end:]) or np.array_equal(a[start:end], b[start:end]):
            raise ValueError("FIT paired-PCM invariant failed")
        mean_rms = sum(np.sqrt(np.mean(x.astype(np.float64)**2)) for x in (a, b)) / 2
        rng = np.random.default_rng(SEED_BASE)
        noise = rng.standard_normal(spec["frames"])
        noise = np.rint(noise / np.sqrt(np.mean(noise**2)) * (mean_rms / 10)).astype(np.int64)
        if sha_bytes(noise.astype("<i2").tobytes()) != spec["noise"]:
            raise ValueError("frozen noise differs")
        for word in WORDS:
            key = f"{engine}_{word}"
            x = originals[word]
            result[key] = {}
            for arm in ARMS:
                if arm in ("clean", "clean_repeat"):
                    y = x.copy()
                else:
                    edit = noise.copy()
                    if arm == "word_noise":
                        edit[:start] = 0
                        edit[end:] = 0
                    elif arm == "outside_noise":
                        edit[start:end] = 0
                    y = x + edit
                if np.any(y < -32768) or np.any(y > 32767):
                    raise ValueError("noise clipped source PCM")
                pcm = y.astype("<i2")
                hash_index = {"full_noise": 2, "word_noise": 3, "outside_noise": 4}.get(arm)
                if hash_index is not None and sha_bytes(pcm.tobytes()) != spec[word][hash_index]:
                    raise ValueError(f"frozen transformed PCM differs: {key}/{arm}")
                result[key][arm] = pcm
    if len(result) != 4 or any(set(arms) != set(ARMS) for arms in result.values()):
        raise ValueError("planned matrix incomplete")
    return result


def preflight():
    import numpy as np
    import torch
    import transformers
    import whisper_fit_word_window_causal_v1 as whisper_source
    import wav2vec2_fit_baseline_stage_capture_v1 as wav_source

    clips = stimuli()
    if transformers.__version__ != "5.20.2" or torch.__version__ != "2.15.0":
        raise ValueError("V2 Wav2Vec2 capture requires Transformers 5.20.2 and Torch 2.15.0")
    whisper_receipt = whisper_source.preflight()
    wav_receipt = wav_source.preflight()
    return {
        "schema": "acoustic_fit_noise_preflight_v2", "protocol_sha256": sha_file(PROTOCOL),
        "runner_sha256": sha_file(Path(__file__)), "runner_commit": source_commit(Path(__file__)),
        "protocol_commit": source_commit(PROTOCOL), "manifest_sha256": MANIFEST_SHA,
        "whisper_source_receipt_sha256": sha_bytes(canonical(whisper_receipt).encode()),
        "wav2vec2_source_receipt_sha256": sha_bytes(canonical(wav_receipt).encode()),
        "numpy_version": np.__version__, "torch_version": torch.__version__,
        "transformers_version": transformers.__version__, "snr_db": SNR_DB,
        "arms": list(ARMS), "cases": {k: {a: sha_bytes(v.tobytes()) for a, v in arms.items()}
                                      for k, arms in clips.items()},
        "model_revisions": {"whisper": whisper_source.MODEL_REVISION,
                            "wav2vec2": wav_source.MODEL_REVISION},
        "wav2vec2_capture_contract": {
            "feature_projection": "tuple_first_tensor",
            "post_block0": "tensor", "post_block5": "tensor", "post_block11": "tensor"},
        "limits": {"threads": 4, "seconds_per_model": MAX_SECONDS,
                   "rss_bytes": MAX_RSS_BYTES, "output_bytes": MAX_OUTPUT_BYTES},
    }


def require_review(review_path: Path, receipt_path: Path, current: dict):
    if canonical(json.loads(receipt_path.read_text())) != canonical(current):
        raise ValueError("preflight receipt differs")
    review = json.loads(review_path.read_text())
    if (review.get("schema") != "acoustic_fit_noise_review_go_v2"
            or review.get("status") != "GO"
            or review.get("runner_sha256") != current["runner_sha256"]
            or review.get("protocol_sha256") != current["protocol_sha256"]
            or review.get("preflight_sha256") != sha_file(receipt_path)):
        raise ValueError("missing/mismatched independent GO")
    review_commit = source_commit(review_path)
    if review_commit in (current["runner_commit"], current["protocol_commit"]):
        raise ValueError("review must be a later separate commit")
    for prior in (current["runner_commit"], current["protocol_commit"]):
        if subprocess.run(("git", "merge-base", "--is-ancestor", prior, review_commit),
                          cwd=ROOT, check=False).returncode != 0:
            raise ValueError("review commit must descend from frozen sources")


def select_wav_capture_v2(stage: str, kind: str, result):
    """Enforce the installed 5.20.2 module return type before old shape checks."""
    import torch

    if stage == "feature_projection":
        if (kind != "post_first" or not isinstance(result, tuple) or len(result) != 2
                or not all(torch.is_tensor(part) for part in result)):
            raise ValueError("feature_projection must return a two-tensor tuple")
        tensor = result[0]
    elif stage in ("post_block0", "post_block5", "post_block11"):
        if kind != "post_tensor" or not torch.is_tensor(result):
            raise ValueError(f"{stage} must return a batch-preserving Tensor")
        tensor = result
    else:
        if kind != "post" or not torch.is_tensor(result):
            raise ValueError(f"{stage} must return a Tensor")
        tensor = result
    if not torch.is_tensor(tensor):
        raise ValueError(f"{stage} selected non-Tensor output")
    return tensor


def wav2vec2_model_forward_v2(model, input_values, frames):
    """V1 capture with only the confirmed block-output contract changed."""
    import torch
    import wav2vec2_fit_baseline_stage_capture_v1 as wav_source

    if torch.__version__ != "2.15.0":
        raise ValueError("V2 capture requires pinned Torch 2.15.0")
    captures = {"input_values": input_values.detach().clone()}
    counts = {stage: (1 if stage == "input_values" else 0) for stage in wav_source.STAGES}
    modules = wav_source.capture_modules(model)
    expected_block = {"post_block0", "post_block5", "post_block11"}
    if set(modules) != set(wav_source.STAGES) - {"input_values"}:
        raise ValueError("Wav2Vec2 capture site inventory changed")
    handles = []
    for stage, (module, kind) in modules.items():
        if stage in expected_block:
            if kind != "post_first":
                raise ValueError("old block capture topology changed")
            kind = "post_tensor"
        if kind == "pre":
            def pre_hook(_module, args, stage=stage):
                counts[stage] += 1
                if counts[stage] != 1:
                    raise ValueError(f"{stage} capture fired more than once")
                if not args or not torch.is_tensor(args[0]):
                    raise ValueError(f"{stage} pre-hook did not receive Tensor")
                captures[stage] = args[0].detach().clone()
            handles.append(module.register_forward_pre_hook(pre_hook))
        else:
            def post_hook(_module, _args, result, stage=stage, kind=kind):
                counts[stage] += 1
                if counts[stage] != 1:
                    raise ValueError(f"{stage} capture fired more than once")
                captures[stage] = select_wav_capture_v2(stage, kind, result).detach().clone()
            handles.append(module.register_forward_hook(post_hook))
    try:
        with torch.inference_mode():
            result = model(input_values=input_values, attention_mask=None,
                           output_hidden_states=False, return_dict=True)
    finally:
        for handle in handles:
            handle.remove()
    if set(captures) != set(wav_source.STAGES) or any(n != 1 for n in counts.values()):
        raise ValueError("Wav2Vec2 capture hook coverage changed")
    if not torch.equal(result.logits, captures["logits"]):
        raise ValueError("captured logits differ from forward result")
    arrays = {stage: wav_source.validate_tensor(stage, captures[stage], frames)
              for stage in wav_source.STAGES}
    return arrays, counts


def run_model(model_name: str, receipt_path: Path, review_path: Path, output: Path):
    import gc
    import numpy as np
    import torch
    import whisper_fit_word_window_causal_v1 as whisper_source
    import wav2vec2_fit_baseline_stage_capture_v1 as wav_source

    current = preflight()
    require_review(review_path, receipt_path, current)
    if output.exists():
        raise FileExistsError("attempt directory exists; no retry/overwrite")
    output.mkdir(parents=True)
    records = output / "records.jsonl"
    start = time.monotonic()

    def check():
        if time.monotonic() - start > MAX_SECONDS:
            raise TimeoutError("model time cap")
        rss = whisper_source.current_rss_bytes()
        if rss > MAX_RSS_BYTES:
            raise MemoryError("RSS cap")
        if records.exists() and records.stat().st_size > MAX_OUTPUT_BYTES:
            raise RuntimeError("output cap")

    def write(obj):
        with records.open("a") as f:
            f.write(canonical(obj) + "\n")
            f.flush()
            os.fsync(f.fileno())

    write({"type": "start", "model": model_name, "preflight_sha256": sha_file(receipt_path),
           "review_sha256": sha_file(review_path), "utc": time.time()})
    try:
        torch.set_num_threads(4)
        clips = stimuli()
        if model_name == "whisper":
            from transformers import WhisperForConditionalGeneration, WhisperProcessor
            processor = WhisperProcessor.from_pretrained(whisper_source.MODEL_ID, revision=whisper_source.MODEL_REVISION, local_files_only=True)
            model = WhisperForConditionalGeneration.from_pretrained(whisper_source.MODEL_ID, revision=whisper_source.MODEL_REVISION, local_files_only=True).to("cpu", dtype=torch.float32).eval()
            whisper_cases, _ = whisper_source.check_case_manifest()
            case_map = {c.key: c for c in whisper_cases}
        elif model_name == "wav2vec2":
            from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
            processor = Wav2Vec2Processor.from_pretrained(wav_source.SNAPSHOT, local_files_only=True)
            model, loading = Wav2Vec2ForCTC.from_pretrained(wav_source.SNAPSHOT, local_files_only=True, output_loading_info=True)
            if (set(loading["missing_keys"]) != {"wav2vec2.masked_spec_embed"}
                    or loading["unexpected_keys"] or loading["mismatched_keys"] or loading["error_msgs"]):
                raise ValueError("Wav2Vec2 checkpoint load report changed")
            model = model.to("cpu", dtype=torch.float32).eval()
            wav_cases, _ = wav_source.trusted_cases()
            case_map = {f"{c.engine}_{c.word}": c for c in wav_cases if c.pair_id == "color_05"}
        else:
            raise ValueError("model must be whisper or wav2vec2")
        if any(p.device.type != "cpu" or p.dtype != torch.float32 for p in model.parameters()):
            raise ValueError("model is not wholly CPU float32")
        for key in sorted(clips):
            clean_states = None
            clean_result = None
            for arm in ARMS:
                check()
                pcm = clips[key][arm]
                samples = pcm.astype(np.float32) / np.float32(32768.0)
                if model_name == "whisper":
                    batch = processor(samples, sampling_rate=16000, return_tensors="pt", return_attention_mask=True)
                    if tuple(batch["input_features"].shape) != (1, 80, 3000):
                        raise ValueError("Whisper mel geometry changed")
                    captures, counts = {}, {s: 0 for s in whisper_source.CAPTURE_SITES}
                    handles = whisper_source.install_capture_hooks(model, captures, counts)
                    try:
                        result = whisper_source.run_model(model, processor, batch)
                    finally:
                        for handle in handles:
                            handle.remove()
                    if set(captures) != set(counts) or any(v != 1 for v in counts.values()):
                        raise ValueError("Whisper capture coverage failed")
                    states = {"mel": batch["input_features"].numpy(),
                              **{s: t.numpy() for s, t in captures.items()}}
                    output_value = {"token_ids": result["token_ids"], "transcript": result["transcript"],
                                    "eos_seen": result["eos_seen"], "censored": result["censored"]}
                    window = case_map[key].window
                else:
                    values = wav_source.normalized_input(pcm, mode="S", stats={}, processor=processor)
                    states, counts = wav2vec2_model_forward_v2(
                        model, torch.from_numpy(values.copy())[None], len(pcm))
                    ids = states["logits"].argmax(axis=-1)[0].astype(np.int64).tolist()
                    output_value = {"argmax_ids": ids, "transcript": processor.batch_decode([ids])[0]}
                    window = None
                if arm == "clean":
                    clean_states, clean_result = states, output_value
                if arm == "clean_repeat" and (output_value != clean_result or
                        any(not np.array_equal(states[s], clean_states[s]) for s in states)):
                    raise ValueError("clean repeat failed exact output/state parity")
                metrics = {}
                for stage, state in states.items():
                    if state.dtype != np.float32 or not np.isfinite(state).all():
                        raise ValueError(f"invalid state: {stage}")
                    if state.shape != clean_states[stage].shape:
                        raise ValueError(f"state shape changed: {stage}")
                    difference = state.astype(np.float64) - clean_states[stage].astype(np.float64)
                    if model_name == "wav2vec2":
                        positions = state.shape[1] if stage not in ("first_conv_raw", "first_conv_norm", "final_feature_conv") else state.shape[2]
                        lo, hi = wav_source.stage_window(case_map[key], stage, positions)
                        view = difference[0] if positions == state.shape[1] else difference[0].T
                    elif stage == "mel":
                        lo, hi = window[0] * 2, window[1] * 2
                        view = difference[0].T
                    else:
                        lo, hi = window
                        view = difference[0]
                    if not 0 <= lo < hi <= len(view):
                        raise ValueError(f"stage window invalid: {stage}")
                    outside = np.concatenate((view[:lo], view[hi:]), axis=0)
                    metrics[stage] = {"shape": list(state.shape), "sha256": sha_bytes(state.tobytes()),
                                      "whole_l2": float(np.linalg.norm(difference)),
                                      "word_l2": float(np.linalg.norm(view[lo:hi])),
                                      "outside_l2": float(np.linalg.norm(outside)),
                                      "outside_maxabs": float(np.max(np.abs(outside))) if outside.size else 0.0,
                                      "nominal_window": [lo, hi]}
                write({"type": "case", "model": model_name, "case": key, "arm": arm,
                       "pcm_sha256": sha_bytes(pcm.tobytes()), "output": output_value,
                       "metrics": metrics, "hook_counts": counts,
                       "elapsed_seconds": time.monotonic() - start,
                       "rss_bytes": whisper_source.current_rss_bytes()})
                check()
        write({"type": "complete", "planned_cases": 20, "elapsed_seconds": time.monotonic() - start})
    except BaseException as exc:
        write({"type": "failure", "error": type(exc).__name__, "detail": str(exc),
               "elapsed_seconds": time.monotonic() - start})
        raise
    finally:
        gc.collect()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("validate-only", "preflight", "run"))
    parser.add_argument("--model", choices=("whisper", "wav2vec2"))
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--review", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "validate-only":
        print(canonical({"status": "source_inputs_valid", "pcm_sha256":
                         {k: {a: sha_bytes(v.tobytes()) for a, v in arms.items()}
                          for k, arms in stimuli().items()}}))
    elif args.command == "preflight":
        print(canonical(preflight()))
    else:
        if not all((args.model, args.receipt, args.review, args.output)):
            parser.error("run requires --model, --receipt, --review, --output")
        run_model(args.model, args.receipt, args.review, args.output)


if __name__ == "__main__":
    main()
