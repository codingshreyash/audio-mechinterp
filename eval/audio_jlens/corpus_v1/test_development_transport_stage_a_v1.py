"""CPU toy checks only; no development candidate model output is loaded."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from development_transport_metrics_v1 import (bootstrap_pairs, summarize_primary)
from development_transport_stage_a_v1 import (comparator_coordinates,
                                              predictions, score_ids,
                                              classify_generated,
                                              generation_result)
from fit_uniform_baselines_v1 import ridge_fit


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cpu")
        self.lm_head = torch.nn.Linear(2048, 4, bias=False)
        with torch.no_grad():
            self.lm_head.weight.zero_()
            self.lm_head.weight[0, 0] = 1
            self.lm_head.weight[1, 0] = -1
        self.model = torch.nn.Module()
        self.model.language_model = torch.nn.Module()
        self.model.language_model.norm = torch.nn.Identity()


class DevelopmentTransportTests(unittest.TestCase):
    def test_generation_patch_hook_changes_complete_toy_answer(self):
        class Tokenizer:
            all_special_ids = [9]
            def decode(self, ids, skip_special_tokens=True):
                return "".join({0: "cat", 1: "dog", 9: "" if skip_special_tokens else "<eos>"}[i]
                               for i in ids)
        class Generator(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.source = torch.nn.Identity()
                self.source_state = torch.zeros(6, 2)
            def generate(self, input_ids, **kwargs):
                state = self.source(self.source_state)
                token = 1 if float(state[-1, 0]) > .5 else 0
                return SimpleNamespace(sequences=torch.cat([
                    input_ids, torch.tensor([[token, 9]], dtype=input_ids.dtype)
                ], dim=1))
        model = Generator()
        processor = SimpleNamespace(tokenizer=Tokenizer())
        case = {"word": "cat", "foil_word": "dog",
                "expected_aliases": [{"token_id": 0}]}
        policy = {"eos_token_id": 9, "words": {
            "cat": {"first_token_ids": [0]},
            "dog": {"first_token_ids": [1]},
        }}
        batch = {"input_ids": torch.tensor([[99]])}
        clean = generation_result(model, processor, batch, case, policy)
        self.assertEqual(clean["token_ids"], [0, 9])
        delta = torch.zeros(6, 2)
        delta[-1, 0] = 1
        changed = generation_result(model, processor, batch, case, policy,
                                    model.source, [[i] for i in range(6)],
                                    delta, 1.0, (6, 2))
        self.assertEqual(changed["token_ids"], [1, 9])
        self.assertEqual(changed["audio_source_hook_calls"], 1)
        self.assertEqual(changed["prefix_class"], "foil")
        self.assertTrue(changed["prefix_class_complete_answer_match"])
        with self.assertRaisesRegex(ValueError, "shape changed"):
            generation_result(model, processor, batch, case, policy,
                              model.source, [[i] for i in range(6)],
                              delta, 1.0, (7, 2))

    def test_hidden_special_and_early_eos_fail_complete_answer(self):
        class Tokenizer:
            all_special_ids = [9, 10]
        aliases = [{"token_id": 0}]
        good = classify_generated("cat", [0, 9], Tokenizer(), "cat", aliases, 9)
        self.assertTrue(good["strict_word_plus_eos"])
        hidden = classify_generated("cat", [0, 10, 9], Tokenizer(), "cat", aliases, 9)
        self.assertTrue(hidden["strict_exact_word"])
        self.assertFalse(hidden["strict_word_plus_eos"])
        self.assertEqual(hidden["nonterminal_special_token_ids"], [10])
        early = classify_generated("cat", [0, 9, 9], Tokenizer(), "cat", aliases, 9)
        self.assertTrue(early["early_eos"])
        self.assertFalse(early["semantic_word_plus_eos"])

    def test_score_orientation(self):
        policy = {"words": {
            "cat": {"first_token_ids": [0]},
            "dog": {"first_token_ids": [1]},
        }}
        spec = {"aliases": {"cat": [{"token_id": 0}],
                            "dog": [{"token_id": 1}]}}
        ids = score_ids(policy, spec, "cat", "dog")
        model = ToyModel()
        basis = torch.eye(2048, 16)
        clean = torch.zeros(2048)
        q = {"map": torch.tensor([1.] + [0.] * 15),
             "global": torch.zeros(16), "six_bin": torch.zeros(16)}
        out = predictions(model, clean, basis, q, 0.1, ids)
        # Increasing expected-cat logit decreases donor-dog log odds.
        self.assertAlmostEqual(out["map"], -0.2, places=6)
        self.assertEqual(out["zero"], 0)

    def test_same_rank_global_and_six_bin_comparator(self):
        torch.manual_seed(9)
        x_global = torch.randn(48, 2, dtype=torch.float64)
        x_six = torch.randn(48, 12, dtype=torch.float64)
        y = torch.randn(48, 16, dtype=torch.float64)
        baseline = {"models": {"connector": {
            "global": ridge_fit(x_global, y, 1.0),
            "six_bin": ridge_fit(x_six, y, 1.0),
        }}}
        lens = {"jacobians": {"connector": torch.zeros(6, 16, 2)}}
        delta = torch.tensor([[1., 2.], [3., 4.], [5., 6.],
                              [7., 8.], [9., 10.], [11., 12.]])
        rows = [[0], [1, 2], [3], [4], [5], [6]]
        q = comparator_coordinates(lens, baseline, "connector", delta, rows)
        self.assertEqual(set(q), {"map", "global", "six_bin"})
        self.assertTrue(torch.equal(q["map"], torch.zeros(16, dtype=torch.float64)))
        self.assertEqual(tuple(q["global"].shape), (16,))

    def test_all_planned_pair_bootstrap_gate(self):
        records = []
        for pair in range(12):
            for j in range(8):
                alphas = [{"alpha": a, "measured_v2_delta_score": 1.0,
                           "predicted_v2_delta_score": {
                               "map": .9, "zero": 0.0,
                               "global": .5, "six_bin": .6,
                           }} for a in (.005, .01, .025)]
                records.append({
                    "case_id": f"{pair}/{j}", "pair_id": f"pair_{pair}",
                    "category": f"category_{pair//3}",
                    "engine": "piper" if j < 4 else "kokoro",
                    "template": "template_0" if j % 4 < 2 else "template_1",
                    "clean": {"semantic_word_plus_eos": True,
                              "prefix_class": "expected",
                              "prefix_class_complete_answer_match": True},
                    "sites": {site: {"absolute_anchor_improved": True,
                                     "alphas": alphas}
                              for site in ("connector", "encoder31")},
                })
        summary = summarize_primary(records, {"connector": .01, "encoder31": .01},
                                    bootstrap=False)
        self.assertTrue(summary["development_primary_gate_passed"])
        self.assertEqual(summary["sites"]["connector"]["determinate_conditions"], 288)
        intervals = bootstrap_pairs(records, "connector", .01, repetitions=20)
        self.assertEqual(intervals["sign"]["n_valid_replicates"], 20)
        records[0]["clean"]["prefix_class"] = "unclassified"
        self.assertEqual(summarize_primary(records, {"connector": .01,
                                                    "encoder31": .01},
                                           bootstrap=False)["clean_task"]["prefix_classified"], 95)


if __name__ == "__main__":
    unittest.main()
