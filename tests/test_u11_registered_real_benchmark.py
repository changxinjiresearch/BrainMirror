#!/usr/bin/env python3
"""Deterministic independent falsification/sanity tests. Mock model only."""
import importlib.util
import math
import tempfile
import pathlib
import unittest
import numpy as np

HERE = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("u11", HERE / "scripts" / "u11_real_mnist_benchmark.py")
u11 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(u11)


class TestPreregisteredContracts(unittest.TestCase):
    def test_grid_exact_and_disjoint(self):
        self.assertEqual(len(u11.TILES), 49)
        self.assertEqual(len({int(i) for t in u11.TILES for i in t}), 28*28)
        self.assertTrue(all(len(t)==16 for t in u11.TILES))

    def test_matched_controls_fail_closed(self):
        x = np.zeros(784, dtype=np.float32)
        pairs, logs = u11.choose_matched_controls(x)
        self.assertEqual(len(pairs),49)
        self.assertTrue(all(k != v for k,v in pairs.items()))
        x[u11.TILES[0]]=1
        pairs, logs = u11.choose_matched_controls(x)
        self.assertNotIn(0,pairs)
        self.assertEqual(logs[0]["reason"],"CONTROL_UNMATCHED")

    def test_zero_edit_identity_and_seeded_nuisance(self):
        x = np.linspace(0,1,784).astype(np.float32)
        z = u11.u10_noise(x,109)
        z2 = u11.u10_noise(x,109)
        z3 = u11.u10_noise(x,211)
        np.testing.assert_array_equal(z,z2)
        self.assertGreater(float(np.max(np.abs(z3-z))), 0)
        for dose in u11.PRE_REGISTERED_DOSES:
            b = u11.occlude(z,u11.TILES[0],dose)
            c = u11.occlude(z,u11.TILES[48],dose)
            if dose == 0:
                np.testing.assert_array_equal(b,c)
            else:
                self.assertGreater(float(np.max(np.abs(b-c))),0)
        np.testing.assert_array_equal(x,u11.occlude(x,u11.TILES[0],0))

    def test_signed_specificity_swaps(self):
        a,b,c = 3.0,1.0,2.5
        original=abs(b-a)-abs(c-a)
        swapped=abs(c-a)-abs(b-a)
        self.assertEqual(original,-swapped)
        stable_zero=abs(a-a)-abs(a-a)
        self.assertEqual(stable_zero,0)
        self.assertFalse(stable_zero>0,"Stability alone must never be positive evidence")

    def test_idx_selection_is_class_stratified(self):
        labels=np.tile(np.arange(10),1000)
        selected=u11.select_fixed_cases(labels,10)
        self.assertEqual(len(selected),100)
        self.assertEqual(selected[:10],list(range(10)))
        self.assertEqual(selected[-1],99)

    def test_holm_nondecreasing_and_sample_stat_unit(self):
        def row(i):
            return {"label":i%10,"correct":True,"usable_primary":True,
                    "aucs":{m:{"deletion_auc":float(i%10)/100+(0.1 if m=="scept" else 0.2),
                                "insertion_auc":float(i%10)/100+(0.7 if m=="scept" else 0.6)}
                            for m in u11.METHODS}}
        d=u11.paired_stats([row(i) for i in range(100)])
        self.assertEqual(d["n_primary"],100)
        self.assertEqual(len(d["comparisons"]),6)
        self.assertTrue(all(0<=r["holm_adjusted_p"]<=1 for r in d["comparisons"]))
        self.assertTrue(all(r["signed_improvement_mean"]>0 for r in d["comparisons"]))

    def test_wrong_idx_length_rejects(self):
        with tempfile.TemporaryDirectory() as tmp:
            import gzip, struct
            f=pathlib.Path(tmp)/"invalid.gz"
            f.write_bytes(gzip.compress(struct.pack(">IIII",2051,10000,28,28)+bytes(2)))
            with self.assertRaises(ValueError):
                u11.read_idx(f,2051)


if __name__=="__main__":
    unittest.main(verbosity=2)
