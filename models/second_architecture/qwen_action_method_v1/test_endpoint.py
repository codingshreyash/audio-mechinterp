"""No-weight fake protocol and endpoint tests."""
from __future__ import annotations

import copy
import unittest

from . import endpoint, plan


class FakeTokenizer:
    eos_token_id = endpoint.EOS_ID
    vocab = {101: "LIST", 102: "ASK", 103: "CANCEL", 104: " CANCEL",
             105: " and LIST", 106: "<|im_start|>", 107: "\n",
             endpoint.EOS_ID: "<|im_end|>"}

    def decode(self, ids, *, skip_special_tokens):
        assert not skip_special_tokens
        return "".join(self.vocab[x] for x in ids)


class SourceOnlyTests(unittest.TestCase):
    def setUp(self):
        self.spec = endpoint.load_spec()
        self.tok = FakeTokenizer()

    def test_frozen_census_and_split(self):
        self.assertEqual(len(self.spec["scripts"]), 30)
        self.assertEqual(sum(row["split"] == "fit_text" for row in self.spec["scripts"]), 18)
        self.assertEqual(sum(row["split"] == "development" for row in self.spec["scripts"]), 6)
        self.assertEqual(sum(row["split"] == "locked" for row in self.spec["scripts"]), 6)
        self.assertEqual(6 + 6 * 2, 18)  # future development/locked WAV count
        mutated = copy.deepcopy(self.spec)
        mutated["scripts"][-1]["text"] = mutated["scripts"][0]["text"]
        with self.assertRaisesRegex(ValueError, "wording reused"):
            endpoint.validate_spec(mutated)

    def test_pinned_source_bytes_without_model_import(self):
        actual = endpoint.validate_local_source(self.spec)
        self.assertEqual(actual["modeling"], self.spec["installed_modeling_sha256"])
        mutated = copy.deepcopy(self.spec)
        mutated["installed_modeling_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "source/config bytes changed"):
            endpoint.validate_local_source(mutated)

    def test_typed_input_has_no_audio(self):
        rows = endpoint.make_typed_conversation(self.spec["system_prompt"],
                                                self.spec["scripts"][0]["text"])
        self.assertEqual([r["role"] for r in rows], ["system", "user"])
        self.assertEqual([piece["type"] for row in rows for piece in row["content"]],
                         ["text", "text"])

    def test_frozen_causal_control_census(self):
        cases = plan.audio_cases(self.spec)
        self.assertEqual(len(cases), 18)
        self.assertEqual(sum(c["split"] == "development" for c in cases), 6)
        self.assertEqual(sum(c["split"] == "locked" for c in cases), 12)
        rows = plan.causal_schedule(self.spec)
        self.assertEqual(len(rows), 90)
        self.assertEqual(sum(r["split"] == "development" for r in rows), 30)
        self.assertEqual(sum(r["split"] == "locked" for r in rows), 60)
        self.assertEqual(len({(r["recipient_id"], r["donor_id"], r["arm"])
                              for r in rows}), 90)
        seeds = [r["seed"] for r in rows if str(r["arm"]).startswith("random_")]
        self.assertEqual(len(seeds), len(set(seeds)))
        label_by_case = {c["id"]: c["label"] for c in cases}
        for row in rows:
            self.assertNotEqual(label_by_case[row["recipient_id"]],
                                label_by_case[row["donor_id"]])
            self.assertEqual(label_by_case[row["recipient_id"]],
                             label_by_case[row["same_label_donor_id"]])

    def test_complete_label_requires_terminal_eos(self):
        for token, label in ((101, "LIST"), (102, "ASK"), (103, "CANCEL"),
                             (104, "CANCEL")):
            got = endpoint.score_complete_answer([token, endpoint.EOS_ID], self.tok)
            self.assertEqual((got["status"], got["label"]), ("complete", label))
        self.assertEqual(endpoint.score_complete_answer([103], self.tok)["status"],
                         "censored")
        self.assertEqual(endpoint.score_complete_answer([103, 105, endpoint.EOS_ID],
                                                        self.tok)["status"], "malformed")
        self.assertEqual(endpoint.score_complete_answer([106, 103, endpoint.EOS_ID],
                                                        self.tok)["status"], "malformed")
        self.assertEqual(endpoint.score_complete_answer([103, endpoint.EOS_ID, 107],
                                                        self.tok)["status"], "censored")

    def test_exact_controls_require_full_eos_path(self):
        a = endpoint.score_complete_answer([103, endpoint.EOS_ID], self.tok)
        b = endpoint.score_complete_answer([104, endpoint.EOS_ID], self.tok)
        c = endpoint.score_complete_answer([103], self.tok)
        self.assertTrue(endpoint.exact_complete_parity(a, copy.deepcopy(a)))
        self.assertFalse(endpoint.exact_complete_parity(a, b))
        self.assertFalse(endpoint.exact_complete_parity(c, c))


if __name__ == "__main__":
    unittest.main()
