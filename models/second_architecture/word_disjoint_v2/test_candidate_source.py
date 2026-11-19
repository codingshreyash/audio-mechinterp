"""No-weight Track B admission, geometry and hook tests."""

from __future__ import annotations

import types
import unittest
import copy
import json
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

import wav2vec2_dev_word_disjoint_audit_v1 as dev_audit
import wav2vec2_dev_word_disjoint_v1 as dev
import wav2vec2_fit_complement_patch_v1 as fit_w2v

from . import candidate_common as common
from . import audit_whisper
from . import run_wav2vec2
from . import run_whisper


class CandidateSourceTests(unittest.TestCase):
    def test_old_source_go_rejects_post_review_auditor_common_helper_edits(self) -> None:
        reviewed = {role: {"path": f"{role}.py", "sha256": role * 8,
                           "commit": "a" * 40}
                    for role in ("runner", "auditor", "common", "helper", "test")}
        text = ("Decision: GO\n```json\n" + json.dumps({
            "schema": "word_disjoint_source_go_v1", "decision": "GO",
            "source_bindings": reviewed}) + "\n```\n")
        common.require_source_go_text(text, reviewed)
        for role in ("auditor", "common", "helper"):
            changed = copy.deepcopy(reviewed)
            changed[role]["sha256"] = "f" * 64
            changed[role]["commit"] = "b" * 40
            with self.assertRaisesRegex(ValueError, "exact reviewed sources"):
                common.require_source_go_text(text, changed)
        sixth = "local:models/second_architecture/wav2vec2_fit_baseline_stage_capture_v1.py"
        reviewed[sixth] = {"path": sixth.removeprefix("local:"),
                           "sha256": "a" * 64, "commit": "a" * 40}
        text = ("Decision: GO\n```json\n" + json.dumps({
            "schema": "word_disjoint_source_go_v1", "decision": "GO",
            "source_bindings": reviewed}) + "\n```\n")
        changed = copy.deepcopy(reviewed)
        changed[sixth]["sha256"] = "b" * 64
        changed[sixth]["commit"] = "b" * 40
        with self.assertRaisesRegex(ValueError, "exact reviewed sources"):
            common.require_source_go_text(text, changed)

    def test_local_import_closure_covers_transitive_candidate_helpers(self) -> None:
        local = common.ROOT / "models" / "second_architecture"
        common_source = common.HERE / "candidate_common.py"
        w2v = common.local_import_closure([
            common.HERE / "run_wav2vec2.py", common.HERE / "audit_wav2vec2.py",
            common_source])
        whisper = common.local_import_closure([
            common.HERE / "run_whisper.py", common.HERE / "audit_whisper.py",
            common_source])
        for closure in (w2v, whisper):
            names = {path.name for path in closure}
            self.assertTrue({"audit.py", "render.py", "queue.py", "__init__.py",
                             "wav2vec2_fit_baseline_stage_capture_v1.py",
                             "wav2vec2_dev_word_disjoint_v1.py",
                             "wav2vec2_dev_word_disjoint_audit_v1.py"} <= names)
            self.assertTrue(all(path.is_relative_to(local) for path in closure))
        self.assertIn(local / "wav2vec2_fit_complement_patch_v1.py", whisper)
        self.assertIn(local / "whisper_fit_word_window_causal_v1.py", whisper)

    def test_relative_import_closure_resolves_parent_and_rejects_escape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            local = root / "models" / "second_architecture"
            package = local / "word_disjoint_v2"
            package.mkdir(parents=True)
            helper = local / "helper.py"
            helper.write_text("VALUE = 1\n")
            module = package / "module.py"
            module.write_text("from .. import helper\n")
            with patch.object(common, "ROOT", root):
                self.assertIn(helper, common.local_import_closure([module]))
                module.write_text("from ... import helper\n")
                with self.assertRaisesRegex(ValueError, "escapes local source tree"):
                    common.local_import_closure([module])

    def test_old_go_cannot_freeze_post_review_sixth_helper_commit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            local = root / "models" / "second_architecture"
            local.mkdir(parents=True)

            def git(*args: str) -> str:
                return subprocess.check_output(["git", "-C", str(root), *args],
                                               text=True, stderr=subprocess.DEVNULL).strip()

            def commit(message: str) -> None:
                git("add", ".")
                git("-c", "user.name=Test", "-c", "user.email=test@example.org",
                    "commit", "-qm", message)

            git("init", "-q")
            paths = {role: local / f"{role}.py" for role in
                     ("runner", "auditor", "common", "helper", "test")}
            sixth = "local:models/second_architecture/transitive.py"
            paths[sixth] = local / "transitive.py"
            for path in paths.values():
                path.write_text("VALUE = 1\n")
            commit("source")
            with patch.object(common, "ROOT", root):
                reviewed = common.source_bindings(paths)
                preflight = root / "preflight.json"
                preflight.write_text("{}\n")
                commit("preflight")
                review = root / "review.md"
                review.write_text("Decision: GO\n```json\n" + json.dumps({
                    "schema": "word_disjoint_source_go_v1", "decision": "GO",
                    "source_bindings": reviewed}) + "\n```\n")
                commit("independent GO")
                paths[sixth].write_text("VALUE = 2\n")
                commit("post GO helper change")
                freeze = root / "freeze.json"
                freeze.write_text("{}\n")
                commit("freeze")
                changed = common.source_bindings(paths)
                with self.assertRaisesRegex(ValueError, "exact reviewed sources"):
                    common.reviewed_freeze_paths(preflight, freeze, review, changed)

    def test_whisper_patch_mask_identity_with_unchanged_mel(self) -> None:
        import torch
        mel = np.zeros((1, 80, 3000), dtype=np.float32)
        mask = np.ones((1, 3000), dtype=np.int64)
        clean = {"attention_mask_shape": list(mask.shape),
                 "attention_mask_sha256": dev.tensor_sha(mask)}
        batch = {"input_features": torch.from_numpy(mel.copy()),
                 "attention_mask": torch.from_numpy(mask.copy())}
        run_whisper.require_recipient_input(batch, mel, clean)
        batch["attention_mask"][0, 0] = 0
        with self.assertRaisesRegex(ValueError, "attention mask differs"):
            run_whisper.require_recipient_input(batch, mel, clean)

    def test_human_review_raw_disagreement_requires_full_third_queue(self) -> None:
        cases = [types.SimpleNamespace(opaque_id=f"clip-{i:02d}",
                 wav_sha256=f"hash-{i:02d}", word="banana") for i in range(32)]
        order = [case.opaque_id for case in cases]
        queue = {"opaque_ids_in_order": order}

        def export(reviewer: str) -> dict:
            return {
                "schema": "word_disjoint_blind_review_v1",
                "reviewer_id": reviewer,
                "queue_sha256": common.QUEUE_SHA,
                "planned_clips": 32,
                "opaque_ids_in_order": order,
                "records": [{
                    "opaque_id": case.opaque_id,
                    "reviewer_id": reviewer,
                    "wav_sha256": case.wav_sha256,
                    "submitted_at_utc": "2026-11-18T12:00:00Z",
                    "verbatim_transcript": "I said banana and then stopped.",
                    "middle_word": "banana",
                    "clear_or_unclear": "clear",
                    "confidence": 5,
                } for case in cases],
            }

        primary, secondary, tertiary = export("a"), export("b"), export("c")
        secondary["records"][0]["verbatim_transcript"] = \
            "I said banana and then stopped!"
        with self.assertRaisesRegex(ValueError, "third review coverage"):
            common.evaluate_human_reviews([primary, secondary], cases, queue)
        tertiary["records"] = tertiary["records"][:1]
        with self.assertRaisesRegex(ValueError, "exact 32-clip queue"):
            common.evaluate_human_reviews([primary, secondary, tertiary], cases, queue)
        tertiary = export("c")
        result = common.evaluate_human_reviews(
            [primary, secondary, tertiary], cases, queue)
        self.assertEqual(result["quality_confirmed"], 32)
        self.assertEqual(result["disagreement_count"], 1)

    def test_exact_parser_and_full_reference(self) -> None:
        expected = ["i", "said", "banana", "and", "then", "stopped"]
        self.assertEqual(common.reference_tokens("banana"), expected)
        self.assertEqual(common.tokens("Ｉ said banana, and then stopped!"), expected)
        self.assertEqual(common.tokens("I_said_banana_and_then_stopped"), expected)
        for value in ("I said banana banana and then stopped",
                      "I said banana tomato and then stopped",
                      "I said banana then stopped",
                      "I said banana and then stopped 2", ""):
            self.assertNotEqual(common.tokens(value), expected)

    def test_exact_fixture_and_bounded_matrices_without_weights(self) -> None:
        cases, _ = common.fixture_cases()
        self.assertEqual(len(cases), 32)
        self.assertEqual(len({c.word for c in cases}), 16)
        self.assertEqual({c.engine for c in cases}, {"piper", "samantha"})
        self.assertEqual(len(common.directions(cases, ("S", "C"))), 64)
        self.assertEqual(len(common.directions(cases, ("stock",))), 32)
        self.assertEqual(len(run_wav2vec2.shared_stats(cases)), 16)
        self.assertLess(run_wav2vec2.byte_ledger(cases)["total_bound"],
                        run_wav2vec2.MAX_OUTPUT_BYTES)
        self.assertLess(run_whisper.byte_ledger(cases)["total_bound"],
                        run_whisper.MAX_OUTPUT_BYTES)
        for case in cases:
            n = run_whisper.active_frames(case)
            a, b = run_whisper.window(case)
            self.assertTrue(0 < a < b < n)
            t = (case.frames - 400) // 320 + 1
            x, y = dev.frame_window(case.slot, t)
            self.assertTrue(0 < x < y < t)

    def test_human_qa_absence_closes_inference_gate(self) -> None:
        cases, _ = common.fixture_cases()
        with patch.object(common, "QA_RECEIPT", Path("/definitely-absent/QA_GO.json")):
            with self.assertRaisesRegex(ValueError, "human listening QA receipt"):
                common.qa_gate(cases)

    def test_whisper_regions_and_independent_patch_reconstruction(self) -> None:
        r = np.zeros((1, 1500, 384), dtype=np.float32)
        d = np.ones((1, 1500, 384), dtype=np.float32)
        n, w = 120, (40, 65)
        for arm in common.ARMS:
            seed = 713 if arm.startswith("random") else None
            candidate, norms = run_whisper.patch_state(r, d, n, w, arm, seed)
            independent = audit_whisper.patch_independent(r, d, n, w, arm, seed)
            self.assertTrue(np.array_equal(candidate, independent), arm)
            self.assertEqual(dev.tensor_sha(candidate), dev_audit.tensor_sha(independent))
            self.assertTrue(np.array_equal(candidate[:, n:], r[:, n:]))
            self.assertGreaterEqual(norms["realized_l2"], 0)
            if arm in ("W", "random_W"):
                self.assertTrue(np.array_equal(candidate[:, :w[0]], r[:, :w[0]]))
                self.assertTrue(np.array_equal(candidate[:, w[1]:], r[:, w[1]:]))
            if arm in ("O", "random_O"):
                self.assertTrue(np.array_equal(candidate[:, w[0]:w[1]], r[:, w[0]:w[1]]))
        self.assertTrue(np.array_equal(run_whisper.patch_state(r, d, n, w, "full", None)[0][:, n:],
                                       r[:, n:]))

    def test_whisper_fake_hook_changes_full_ids_and_is_removed(self) -> None:
        import torch

        class FakeProcessor:
            def batch_decode(self, sequences, skip_special_tokens=True):
                ids = sequences[0].tolist() if hasattr(sequences[0], "tolist") else sequences[0]
                return ["donor" if ids[1] == 3 else "recipient"]

        class FakeModel:
            def __init__(self):
                encoder = types.SimpleNamespace(
                    layers=[torch.nn.Identity()], layer_norm=torch.nn.Identity())
                self.model = types.SimpleNamespace(encoder=encoder)
                self.generation_config = types.SimpleNamespace(eos_token_id=2)

            def generate(self, **kwargs):
                x = torch.zeros((1, 1500, 384), dtype=torch.float32)
                x = self.model.encoder.layers[0](x)
                x = self.model.encoder.layer_norm(x)
                token = 3 if x[0, 50, 0] > 0 else 4
                return types.SimpleNamespace(sequences=torch.tensor([[1, token, 2]]))

        model = FakeModel()
        processor = FakeProcessor()
        clean = np.zeros((1, 1500, 384), dtype=np.float32)
        donor = clean.copy()
        donor[0, 50, 0] = 1.0
        for site in common.SITES_WHISPER:
            result, counts = run_whisper.generate(
                model, processor, {}, patch=(site, clean, donor))
            self.assertEqual(result["ids"], [1, 3, 2])
            self.assertEqual(counts[site], 1)
            self.assertEqual(len(run_whisper._hook_module(model, site)._forward_hooks), 0)

    def test_wav2vec2_fake_two_site_hooks_and_live_identity(self) -> None:
        import torch

        class TupleLayer(torch.nn.Module):
            def forward(self, x):
                return (x, None)

        class FakeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.projection = TupleLayer()
                self.final = TupleLayer()
                self.wav2vec2 = types.SimpleNamespace(
                    feature_projection=self.projection,
                    encoder=types.SimpleNamespace(layers=[torch.nn.Identity() for _ in range(11)]
                                                  + [self.final]))

            def forward(self, input_values, **kwargs):
                t = (input_values.shape[1] - 400) // 320 + 1
                x = torch.zeros((1, t, 768), dtype=torch.float32)
                x = self.projection(x)[0]
                x = self.final(x)[0]
                logits = torch.zeros((1, t, 32), dtype=torch.float32)
                logits[:, :, 1] = x[:, :, 0]
                return types.SimpleNamespace(logits=logits)

        model = FakeModel()
        frames = 16000
        tensor = torch.zeros((1, frames), dtype=torch.float32)
        states, logits = run_wav2vec2.capture_clean(model, tensor, frames)
        self.assertEqual(set(states), set(common.SITES_W2V))
        self.assertTrue(np.array_equal(states["feature_projection"],
                                       states["post_block11"]))
        for site in common.SITES_W2V:
            edited = states[site].copy()
            edited[:, 20:25, 0] = 1.0
            patched, count = fit_w2v.patched_forward(
                model, tensor, site=site, clean_state=states[site],
                patched_state=edited)
            self.assertEqual(count, 1)
            self.assertTrue(np.all(patched[:, 20:25, 1] == 1))
            self.assertEqual(len((model.projection if site == "feature_projection"
                                  else model.final)._forward_hooks), 0)


if __name__ == "__main__":
    unittest.main()
