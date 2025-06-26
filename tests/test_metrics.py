from collections import defaultdict
from rmet.metrics import (
    coverage,
    precision,
    recall,
    dcg,
    ndcg,
    hitrate,
    f_score,
    average_precision,
    reciprocal_rank,
    calculate,
)
import unittest
import math
import torch
import numpy as np


class TestRecommenderMetricsTorch(unittest.TestCase):

    def setUp(self):
        self.logits = [[0.9, 0.8, 0.1, 0.4, 0.2], [0.1, 0.2, 0.3, 0.9, 0.8]]
        # relevant: [0, 2, 3], [3, 4]
        self.targets = [[1, 0, 1, 1, 0], [0, 0, 0, 1, 1]]

        self.logits_torch = torch.tensor(self.logits)
        self.targets_torch = torch.tensor(self.targets)

        self.logits_np = np.array(self.logits)
        self.targets_np = np.array(self.targets)

        self.k = 3
        self.best_logit_indices = [[0, 1, 3], [3, 4, 2]]
        self.best_logit_indices_torch = torch.tensor(self.best_logit_indices)
        self.best_logit_indices_np = np.array(self.best_logit_indices)

        self.precision1 = 2 / 3
        self.precision2 = 2 / 3
        self.precision = [self.precision1, self.precision2]

        self.recall1 = 2 / 3
        self.recall2 = 1.0
        self.recall = [self.recall1, self.recall2]

        self.dcg1 = 1 + 0 + 1 / math.log2(4)
        self.dcg2 = 1 + 1 / math.log2(3)
        self.dcg = [self.dcg1, self.dcg2]

        self.ideal_dcg1 = 1 + 1 / math.log2(3) + 1 / math.log2(4)
        self.ideal_dcg2 = 1 + 1 / math.log2(3)
        self.ideal_dcg = [self.ideal_dcg1, self.ideal_dcg2]

        self.ndcg1 = self.dcg1 / self.ideal_dcg1
        self.ndcg2 = self.dcg2 / self.ideal_dcg2
        self.ndcg = [self.ndcg1, self.ndcg2]

        self.hitrate1 = 1.0
        self.hitrate2 = 1.0
        self.hitrate = [self.hitrate1, self.hitrate2]

        self.f1_user1 = (
            2 * self.precision1 * self.recall1 / (self.precision1 + self.recall1)
        )
        self.f1_user2 = (
            2 * self.precision2 * self.recall2 / (self.precision2 + self.recall2)
        )
        self.f1_user = [self.f1_user1, self.f1_user2]

        self.ap1 = (1 / 1 + 2 / 3) / 3
        self.ap2 = (1 / 1 + 2 / 2) / 2
        self.ap = [self.ap1, self.ap2]

        # first relevant at rank 1 for both users
        self.rr1 = 1.0
        self.rr2 = 1.0
        self.rr = [self.rr1, self.rr2]

        self.coverage = 5 / 5

        self.user_computation_results = {
            "dcg": self.dcg,
            "ndcg": self.ndcg,
            "recall": self.recall,
            "precision": self.precision,
            "hitrate": self.hitrate,
            "f_score": self.f1_user,
            "ap": self.ap,
            "rr": self.rr,
        }
        self.metrics = list(self.user_computation_results.keys()) + ["coverage"]

    def test_precision(self):
        expected = torch.tensor(self.precision)
        result = precision(self.logits_torch, self.targets_torch, self.k)
        self.assertTrue(torch.allclose(result, expected, atol=1e-4))

    def test_precision_numpy(self):
        expected = np.array(self.precision)
        result = precision(self.logits_np, self.targets_np, self.k)
        self.assertTrue(np.allclose(result, expected, atol=1e-4))

    def test_recall(self):
        expected = torch.tensor(self.recall)
        result = recall(self.logits_torch, self.targets_torch, self.k)
        self.assertTrue(torch.allclose(result, expected, atol=1e-4))

    def test_recall_numpy(self):
        expected = np.array(self.recall)
        result = recall(self.logits_np, self.targets_np, self.k)
        self.assertTrue(np.allclose(result, expected, atol=1e-4))

    def test_dcg(self):
        expected = torch.tensor(self.dcg)
        result = dcg(self.logits_torch, self.targets_torch, self.k)
        self.assertTrue(torch.allclose(result, expected, atol=1e-4))

    def test_dcg_numpy(self):
        expected = np.array(self.dcg)
        result = dcg(self.logits_np, self.targets_np, self.k)
        self.assertTrue(np.allclose(result, expected, atol=1e-4))

    def test_ndcg(self):
        expected = torch.tensor(self.ndcg)
        result = ndcg(self.logits_torch, self.targets_torch, self.k)
        self.assertTrue(torch.allclose(result, expected, atol=1e-4))

    def test_ndcg_numpy(self):
        expected = np.array(self.ndcg)
        result = ndcg(self.logits_np, self.targets_np, self.k)
        self.assertTrue(np.allclose(result, expected, atol=1e-4))

    def test_hitrate(self):
        expected = torch.tensor(self.hitrate)
        result = hitrate(self.logits_torch, self.targets_torch, self.k)
        self.assertTrue(torch.allclose(result, expected))

    def test_hitrate_numpy(self):
        expected = np.array(self.hitrate)
        result = hitrate(self.logits_np, self.targets_np, self.k)
        self.assertTrue(np.allclose(result, expected))

    def test_f_score(self):
        expected = torch.tensor(self.f1_user)
        result = f_score(self.logits_torch, self.targets_torch, self.k)
        self.assertTrue(torch.allclose(result, expected, atol=1e-4))

    def test_f_score_numpy(self):
        expected = np.array(self.f1_user)
        result = f_score(self.logits_np, self.targets_np, self.k)
        self.assertTrue(np.allclose(result, expected, atol=1e-4))

    def test_average_precision(self):
        expected = torch.tensor(self.ap)
        result = average_precision(self.logits_torch, self.targets_torch, self.k)
        self.assertTrue(torch.allclose(result, expected, atol=1e-4))

    def test_average_precision_numpy(self):
        expected = np.array(self.ap)
        result = average_precision(self.logits_np, self.targets_np, self.k)
        self.assertTrue(np.allclose(result, expected, atol=1e-4))

    def test_reciprocal_rank(self):
        expected = torch.tensor(self.rr)
        result = reciprocal_rank(self.logits_torch, self.targets_torch, self.k)
        self.assertTrue(torch.allclose(result, expected, atol=1e-4))

    def test_reciprocal_rank_numpy(self):
        expected = np.array(self.rr)
        result = reciprocal_rank(self.logits_np, self.targets_np, self.k)
        self.assertTrue(np.allclose(result, expected, atol=1e-4))

    def test_all_zero_targets(self):
        logits = self.logits_torch
        targets = torch.zeros_like(self.targets_torch)
        k = self.k

        zero = torch.tensor([0.0, 0.0])
        self.assertTrue(torch.allclose(precision(logits, targets, k), zero))
        self.assertTrue(torch.allclose(recall(logits, targets, k), zero))
        self.assertTrue(torch.allclose(hitrate(logits, targets, k), zero))
        self.assertTrue(torch.allclose(f_score(logits, targets, k), zero))
        self.assertTrue(torch.allclose(average_precision(logits, targets, k), zero))
        self.assertTrue(torch.allclose(reciprocal_rank(logits, targets, k), zero))

    def test_coverage(self):
        self.assertAlmostEqual(coverage(self.logits_torch, self.k), self.coverage)
        self.assertAlmostEqual(coverage(self.logits_torch, 2), 4 / 5)

    def test_coverage(self):
        self.assertAlmostEqual(coverage(self.logits_np, self.k), self.coverage)
        self.assertAlmostEqual(coverage(self.logits_np, 2), 4 / 5)

    def test_compute_nested(self):

        result = calculate(
            metrics=self.metrics,
            logits=self.logits_torch,
            targets=self.targets_torch,
            k=self.k,
            return_aggregated=True,
            return_individual=True,
            calculate_std=True,
            flatten_results=False,
            flattened_parts_separator=None,
            flattened_results_prefix=None,
            n_items=None,
            best_logit_indices=None,
            return_best_logit_indices=False,
        )

        expected = defaultdict(lambda: dict())
        for m, r in self.user_computation_results.items():
            expected[f"{m}_individual"][self.k] = torch.tensor(r)
            expected[m][self.k] = np.mean(r)
            # ddof to adjust to degrees of freedom = 1 for std computation in PyTorch
            expected[f"{m}_std"][self.k] = np.std(r, ddof=1)
        expected.update({"coverage": {self.k: self.coverage}})

        # check matching keys
        self.assertSetEqual(set(result.keys()), set(expected.keys()))
        for k in result:
            # check matching subkeys
            self.assertSetEqual(set(result[k].keys()), set(expected[k].keys()))
            for sk in result[k]:
                # check matching results
                if k.endswith("individual"):
                    self.assertTrue(torch.allclose(result[k][sk], expected[k][sk]))
                else:
                    self.assertAlmostEqual(result[k][sk], expected[k][sk])

    def test_compute_flattened(self):

        separator = "_|_"
        prefix = "<my-prefix>"

        result = calculate(
            metrics=self.metrics,
            logits=self.logits_torch,
            targets=self.targets_torch,
            k=self.k,
            return_aggregated=True,
            return_individual=True,
            calculate_std=True,
            flatten_results=True,
            flattened_parts_separator=separator,
            flattened_results_prefix=prefix,
            n_items=None,
            best_logit_indices=None,
            return_best_logit_indices=False,
        )

        expected = dict()
        for m, k in self.user_computation_results.items():
            expected[f"{prefix}{separator}{m}_individual@{self.k}"] = torch.tensor(k)
            expected[f"{prefix}{separator}{m}@{self.k}"] = np.mean(k)
            # ddof to adjust to degrees of freedom = 1 for std computation in PyTorch
            expected[f"{prefix}{separator}{m}_std@{self.k}"] = np.std(k, ddof=1)
        expected.update({f"{prefix}{separator}coverage@{self.k}": self.coverage})

        # check matching keys
        self.assertSetEqual(set(result.keys()), set(expected.keys()))
        for k in result:
            # check matching results
            if "individual" in k:
                self.assertTrue(torch.allclose(result[k], expected[k]))
            else:
                self.assertAlmostEqual(result[k], expected[k])

    def test_best_indices(self):

        result, best_indices = calculate(
            metrics=self.metrics,
            logits=self.logits_torch,
            targets=self.targets_torch,
            k=self.k,
            return_best_logit_indices=True,
            return_individual=True,
            flatten_results=True,
        )

        self.assertTrue(torch.allclose(best_indices, self.best_logit_indices_torch))

        result_on_best_logits = calculate(
            metrics=self.metrics,
            logits=self.logits_torch,
            targets=self.targets_torch,
            k=self.k,
            best_logit_indices=best_indices,
            return_best_logit_indices=False,
            return_individual=True,
            flatten_results=True,
        )

        # check matching keys
        self.assertSetEqual(set(result.keys()), set(result_on_best_logits.keys()))
        for k in result:
            # check matching results
            if "individual" in k:
                self.assertTrue(torch.allclose(result[k], result_on_best_logits[k]))
            else:
                self.assertAlmostEqual(result[k], result_on_best_logits[k])


if __name__ == "__main__":
    unittest.main()
