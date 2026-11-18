"""CPU checks of the v2 candidate proposal's gates; no audio/model outputs."""

import copy
import json
import unittest

import validate_candidates_unfrozen as v


class CandidateProposalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = json.loads(v.PROPOSAL.read_text())
        cls.old, _ = v.historical_words()

    def test_full_candidate_counts_and_whole_pair_split(self):
        words, assigned = v.validate_structure(self.spec, self.old)
        self.assertEqual((len(words), len(set(words))), (144, 144))
        self.assertEqual({k: len(x) for k, x in assigned.items()},
                         {"fit_pool": 36, "development": 12, "locked_test": 12})
        for category in v.CATEGORIES:
            fit = [x for x in assigned["fit_pool"] if x["category"] == category]
            self.assertEqual(len(fit), 9)
            self.assertEqual(sum(x["fit_role"] == "basis" for x in fit), 4)
            self.assertEqual(sum(x["fit_role"] == "map" for x in fit), 5)
            self.assertEqual(sum(x["category"] == category for x in assigned["development"]), 3)
            self.assertEqual(sum(x["category"] == category for x in assigned["locked_test"]), 3)
        ids = [x["pair_id"] for rows in assigned.values() for x in rows]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(assigned, v.assign_pairs(self.spec))

    def test_old_locked_word_cannot_enter_any_split(self):
        changed = copy.deepcopy(self.spec)
        changed["pairs_by_category"]["animal"][0][0] = "panda"
        with self.assertRaisesRegex(ValueError, "prior word metadata"):
            v.validate_structure(changed, self.old)

    def test_reserve_words_are_disjoint_too(self):
        changed = copy.deepcopy(self.spec)
        changed["ordered_reserve_pairs_by_category"]["color"][0][0] = "azure"
        with self.assertRaisesRegex(ValueError, "word reuse"):
            v.validate_structure(changed, self.old)

    def test_homophone_screen_rejects_missing_or_collision(self):
        with self.assertRaisesRegex(ValueError, "missing"):
            v.check_phonemes(["one"], set(), {})
        with self.assertRaisesRegex(ValueError, "homophones"):
            v.check_phonemes(["one", "won"], set(),
                             {"one": {("W", "AH", "N")},
                              "won": {("W", "AH", "N")}})

    def test_templates_remain_split_exclusive(self):
        changed = copy.deepcopy(self.spec)
        changed["templates_by_split"]["locked_test"][0]["id"] = "v2_dev_listen"
        with self.assertRaises(ValueError):
            v.validate_structure(changed, self.old)


if __name__ == "__main__":
    unittest.main()
