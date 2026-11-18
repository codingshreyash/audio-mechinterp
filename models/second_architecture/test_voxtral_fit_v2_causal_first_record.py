"""CPU-only checks for frozen first-record Voxtral causal benchmark plumbing."""

import unittest

import torch

import voxtral_fit_v2_causal_first_record as assay


class FirstRecordTests(unittest.TestCase):
    def test_strict_classification_precedence(self):
        special = {2, 24, 99}
        args = (special, 2, "beaver", "monkey")
        self.assertEqual(assay.classify([55, 24, 2], "Beaver", *args), "malformed_special")
        self.assertEqual(assay.classify([55], "Beaver", *args), "missing_eos")
        self.assertEqual(assay.classify([55, 2], "Beaver!", *args), "malformed_surface")
        self.assertEqual(assay.classify([55, 2], "  BEAVER  ", *args), "recipient")
        self.assertEqual(assay.classify([55, 2], "Monkey", *args), "opposite")
        self.assertEqual(assay.classify([55, 2], "Sparrow", *args), "other")

    def test_edit_masks_and_controls(self):
        source = torch.zeros((12, 8), dtype=torch.bfloat16)
        donor = source.clone()
        donor[2:6] = 2
        donor[6:10] = 1
        whole = [2, 3, 4, 5]
        edited, info = assay.prepare_edit(source, donor, whole, whole, "whole", "case", "projector", "loss")
        self.assertTrue(info["valid"])
        self.assertTrue(torch.equal(edited, donor[whole]))
        same, same_info = assay.prepare_edit(source, source, whole, whole, "self", "case", "projector", "loss")
        self.assertTrue(same_info["valid"])
        self.assertTrue(torch.equal(same, source[whole]))
        control, control_info = assay.prepare_edit(source, donor, [6, 7, 8, 9], whole,
                                                    "post", "case", "projector", "loss")
        self.assertTrue(control_info["valid"])
        self.assertLessEqual(abs(control_info["effective_bf16_norm"] - info["target_bf16_norm"]),
                             0.05 * info["target_bf16_norm"])
        random1, rinfo1 = assay.prepare_edit(source, donor, whole, whole, "random", "case", "projector", "loss")
        random2, rinfo2 = assay.prepare_edit(source, donor, whole, whole, "random", "case", "projector", "loss")
        self.assertEqual(rinfo1["random_seed"], rinfo2["random_seed"])
        self.assertTrue(torch.equal(random1, random2))

    def test_invalid_zero_norm_does_not_create_edit(self):
        base = torch.zeros((8, 4), dtype=torch.bfloat16)
        edit, info = assay.prepare_edit(base, base, [0, 1], [0, 1], "pre", "case", "projector", "restore")
        self.assertIsNone(edit)
        self.assertFalse(info["valid"])

    def test_projection(self):
        self.assertTrue(assay.project_resource(100, 200, 10)["passes"])
        self.assertFalse(assay.project_resource(100, 400, 10)["passes"])
        self.assertFalse(assay.project_resource(100, 200, assay.base.MEMORY_LIMIT_BYTES)["passes"])
        full = assay.gate_projection_on_workload(assay.project_resource(100, 200, 10), 100, 8, 0)
        self.assertTrue(full["passes"])
        incomplete = assay.gate_projection_on_workload(assay.project_resource(100, 200, 10), 99, 8, 1)
        self.assertFalse(incomplete["passes"])
        self.assertIsNone(incomplete["projected_eight_record_seconds"])

    def test_hook_sites_and_edited_shape_precondition(self):
        self.assertEqual(assay.expected_hook_shape("encoder31"), (1, 1500, 1280))
        self.assertEqual(assay.expected_hook_shape("projector"), (375, 3072))
        with self.assertRaises(ValueError):
            assay.answer_and_scores(object(), None, {}, "beaver", "monkey", {},
                                    "projector", [1, 2], torch.zeros((1, 3072), dtype=torch.bfloat16))
        with self.assertRaises(ValueError):
            assay.answer_and_scores(object(), None, {}, "beaver", "monkey", {},
                                    "projector", [1], torch.zeros((1, 3072), dtype=torch.float32))


if __name__ == "__main__":
    unittest.main()
