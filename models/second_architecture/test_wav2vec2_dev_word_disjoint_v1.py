"""No-weight tests for prospective development geometry and saved-data rules."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import wav2vec2_dev_word_disjoint_audit_v1 as audit
import wav2vec2_dev_word_disjoint_v1 as run


class NoWeightDevTests(unittest.TestCase):
    def test_exact_parser_and_template_gate(self)->None:
        expected=["i","said","banana","and","then","stopped"]
        self.assertEqual(run.tokens("Ｉ said banana, and then stopped!"),expected)
        self.assertEqual(audit.token_list("I_said_banana_and_then_stopped."),expected)
        for bad in ("I said banana banana and then stopped",
                    "I said banana tomato and then stopped",
                    "I said banana then stopped",
                    "I said banana and then stopped 2",""):
            self.assertNotEqual(run.tokens(bad),expected)
        self.assertEqual(run.tokens("Listen for my chosen word: rabbit. Tell me the chosen word."),
                         ["listen","for","my","chosen","word","rabbit","tell","me",
                          "the","chosen","word"])

    def test_complete_matrix_and_pcm(self)->None:
        cases,stats,refs=run.trusted_dev_cases()
        cells=run.directions(cases)
        self.assertEqual((len(cases),len(stats),len(refs),len(cells)),(48,24,48,96))
        self.assertEqual({c.engine for c in cases},{"piper","kokoro"})
        self.assertEqual(len({c.word for c in cases}),24)
        self.assertEqual(len({(d["index"],a) for d in cells for a in run.ARMS}),576)
        self.assertLess(run.estimate_bytes(cases)["total_bound"],run.MAX_OUTPUT_BYTES)
        self.assertTrue(all(refs[c.key].count(c.word)==1 for c in cases))
        for d in cells:
            self.assertEqual(tuple(d["slot"]),next(c.slot for c in cases if c.key==d["recipient_key"]))
        # Recompute shared C mode on a real development pair; no model weights.
        pair=[c for c in cases if c.pair_id==cases[0].pair_id and c.engine==cases[0].engine]
        self.assertEqual(len(pair),2)
        x=[run.base.normalized_input(run.base.load_pcm(c),mode="C",
                                      stats=stats[(c.pair_id,c.engine)]) for c in pair]
        s,e=pair[0].slot
        self.assertTrue(np.array_equal(x[0][:s],x[1][:s]))
        self.assertTrue(np.array_equal(x[0][e:],x[1][e:]))

    def test_patch_support_and_independent_reconstruction(self)->None:
        r=np.zeros((1,9,768),dtype=np.float32)
        d=np.ones((1,9,768),dtype=np.float32)
        w=(3,6)
        for arm in run.ARMS:
            seed=71 if arm.startswith("random") else None
            patch,norms=run.edit_state(r,d,w,arm,seed)
            independent=audit.patch_independent(r,d,w,arm,seed)
            self.assertTrue(np.array_equal(patch,independent),arm)
            self.assertEqual(run.tensor_sha(patch),audit.tensor_sha(independent))
            self.assertGreaterEqual(norms["realized_l2"],0)
            if arm in ("W","random_W"):
                self.assertTrue(np.array_equal(patch[:,:3],r[:,:3]))
                self.assertTrue(np.array_equal(patch[:,6:],r[:,6:]))
            if arm in ("O","random_O"):
                self.assertTrue(np.array_equal(patch[:,3:6],r[:,3:6]))
        self.assertEqual(run.frame_window((30527,42788),249),(95,134))
        with self.assertRaisesRegex(ValueError,"invalid"):
            run.frame_window((0,320),2)

    def test_hash_checked_sidecar(self)->None:
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)
            path=output/"captures/test.npy"
            path.parent.mkdir()
            value=np.ones((1,3,32),dtype=np.float32)
            np.save(path,value,allow_pickle=False)
            self.assertTrue(np.array_equal(audit.array(output,"captures/test.npy",audit.sha(path),value.shape),value))
            with self.assertRaisesRegex(ValueError,"hash"):
                audit.array(output,"captures/test.npy","0"*64,value.shape)
            with self.assertRaisesRegex(ValueError,"reference"):
                audit.array(output,"captures/../escape.npy","0"*64,value.shape)

if __name__=="__main__":unittest.main()
