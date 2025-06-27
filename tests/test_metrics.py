import math
import torch
import unittest
import numpy as np
import scipy.sparse as sp
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


class TestRecommenderMetricsTorch(unittest.TestCase):

    def setUp(self):
        self.logits = [[0.9, 0.8, 0, 0.4, 0], [0.0, 0.0, 0.3, 0.9, 0.8]]
        # relevant: [0, 2, 3], [3, 4]
        self.targets = [[1, 0, 1, 1, 0], [0, 0, 0, 1, 1]]

        self.supported_types = {
            "torch": {
                "cast": torch.tensor,
                "cast-result": torch.tensor,
                "all_close": torch.allclose,
                "zeros_like": torch.zeros_like,
            },
            "np": {
                "cast": np.array,
                "cast-result": np.array,
                "all_close": np.allclose,
                "zeros_like": np.zeros_like,
            },
            "sparse-csr": {
                "cast": sp.csr_array,
                "cast-result": np.array,
                "all_close": np.allclose,
                "zeros_like": np.zeros_like,
            },
        }

        self.k = 3
        self.best_logit_indices = [[0, 1, 3], [3, 4, 2]]

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
        for _, fn_lookup in self.supported_types.items():
            expected = fn_lookup["cast-result"](self.precision)
            result = precision(
                logits=fn_lookup["cast"](self.logits),
                targets=fn_lookup["cast"](self.targets),
                k=self.k,
            )
            self.assertTrue(fn_lookup["all_close"](result, expected, atol=1e-4))

    def test_recall(self):
        for _, fn_lookup in self.supported_types.items():
            expected = fn_lookup["cast-result"](self.recall)
            result = recall(
                logits=fn_lookup["cast"](self.logits),
                targets=fn_lookup["cast"](self.targets),
                k=self.k,
            )
            self.assertTrue(fn_lookup["all_close"](result, expected, atol=1e-4))

    def test_dcg(self):
        for _, fn_lookup in self.supported_types.items():
            expected = fn_lookup["cast-result"](self.dcg)
            result = dcg(
                logits=fn_lookup["cast"](self.logits),
                targets=fn_lookup["cast"](self.targets),
                k=self.k,
            )
            self.assertTrue(fn_lookup["all_close"](result, expected, atol=1e-4))

    def test_ndcg(self):
        for _, fn_lookup in self.supported_types.items():
            expected = fn_lookup["cast-result"](self.ndcg)
            result = ndcg(
                logits=fn_lookup["cast"](self.logits),
                targets=fn_lookup["cast"](self.targets),
                k=self.k,
            )
            self.assertTrue(fn_lookup["all_close"](result, expected, atol=1e-4))

    def test_hitrate(self):
        for _, fn_lookup in self.supported_types.items():
            expected = fn_lookup["cast-result"](self.hitrate)
            result = hitrate(
                logits=fn_lookup["cast"](self.logits),
                targets=fn_lookup["cast"](self.targets),
                k=self.k,
            )
            self.assertTrue(fn_lookup["all_close"](result, expected, atol=1e-4))

    def test_f_score(self):
        for _, fn_lookup in self.supported_types.items():
            expected = fn_lookup["cast-result"](self.f1_user)
            result = f_score(
                logits=fn_lookup["cast"](self.logits),
                targets=fn_lookup["cast"](self.targets),
                k=self.k,
            )
            self.assertTrue(fn_lookup["all_close"](result, expected, atol=1e-4))

    def test_average_precision(self):
        for _, fn_lookup in self.supported_types.items():
            expected = fn_lookup["cast-result"](self.ap)
            result = average_precision(
                logits=fn_lookup["cast"](self.logits),
                targets=fn_lookup["cast"](self.targets),
                k=self.k,
            )
            self.assertTrue(fn_lookup["all_close"](result, expected, atol=1e-4))

    def test_reciprocal_rank(self):
        for _, fn_lookup in self.supported_types.items():
            expected = fn_lookup["cast-result"](self.rr)
            result = reciprocal_rank(
                logits=fn_lookup["cast"](self.logits),
                targets=fn_lookup["cast"](self.targets),
                k=self.k,
            )
            self.assertTrue(fn_lookup["all_close"](result, expected, atol=1e-4))

    def test_all_zero_targets(self):
        for _, fn_lookup in self.supported_types.items():
            logits = fn_lookup["cast"](self.logits)
            targets = fn_lookup["zeros_like"](fn_lookup["cast-result"](self.targets))
            k = self.k

            zero = fn_lookup["cast-result"]([0.0, 0.0])
            self.assertTrue(fn_lookup["all_close"](precision(logits, targets, k), zero))
            self.assertTrue(fn_lookup["all_close"](recall(logits, targets, k), zero))
            self.assertTrue(fn_lookup["all_close"](hitrate(logits, targets, k), zero))
            self.assertTrue(fn_lookup["all_close"](f_score(logits, targets, k), zero))
            self.assertTrue(
                fn_lookup["all_close"](average_precision(logits, targets, k), zero)
            )
            self.assertTrue(
                fn_lookup["all_close"](reciprocal_rank(logits, targets, k), zero)
            )

    def test_coverage(self):
        for _, fn_lookup in self.supported_types.items():
            logits = fn_lookup["cast"](self.logits)
            self.assertAlmostEqual(coverage(logits, self.k), self.coverage)
            self.assertAlmostEqual(coverage(logits, 2), 4 / 5)

    def test_compute_nested(self):

        for _, fn_lookup in self.supported_types.items():
            result = calculate(
                metrics=self.metrics,
                logits=fn_lookup["cast"](self.logits),
                targets=fn_lookup["cast"](self.targets),
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
                expected[f"{m}_individual"][self.k] = fn_lookup["cast-result"](r)
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
                        self.assertTrue(
                            fn_lookup["all_close"](result[k][sk], expected[k][sk])
                        )
                    else:
                        self.assertAlmostEqual(result[k][sk], expected[k][sk])

    def test_compute_flattened(self):

        separator = "_|_"
        prefix = "<my-prefix>"

        for _, fn_lookup in self.supported_types.items():
            result = calculate(
                metrics=self.metrics,
                logits=fn_lookup["cast"](self.logits),
                targets=fn_lookup["cast"](self.targets),
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
                expected[f"{prefix}{separator}{m}_individual@{self.k}"] = fn_lookup[
                    "cast-result"
                ](k)
                expected[f"{prefix}{separator}{m}@{self.k}"] = np.mean(k)
                # ddof to adjust to degrees of freedom = 1 for std computation in PyTorch
                expected[f"{prefix}{separator}{m}_std@{self.k}"] = np.std(k, ddof=1)
            expected.update({f"{prefix}{separator}coverage@{self.k}": self.coverage})

            # check matching keys
            self.assertSetEqual(set(result.keys()), set(expected.keys()))
            for k in result:
                # check matching results
                if "individual" in k:
                    self.assertTrue(fn_lookup["all_close"](result[k], expected[k]))
                else:
                    self.assertAlmostEqual(result[k], expected[k])

    def test_best_indices(self):

        for _, fn_lookup in self.supported_types.items():
            result, best_indices = calculate(
                metrics=self.metrics,
                logits=fn_lookup["cast"](self.logits),
                targets=fn_lookup["cast"](self.targets),
                k=self.k,
                return_best_logit_indices=True,
                return_individual=True,
                flatten_results=True,
            )

            self.assertTrue(
                fn_lookup["all_close"](
                    best_indices, fn_lookup["cast-result"](self.best_logit_indices)
                )
            )

            result_on_best_logits = calculate(
                metrics=self.metrics,
                logits=fn_lookup["cast"](self.logits),
                targets=fn_lookup["cast"](self.targets),
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
                    self.assertTrue(
                        fn_lookup["all_close"](result[k], result_on_best_logits[k])
                    )
                else:
                    self.assertAlmostEqual(result[k], result_on_best_logits[k])


if __name__ == "__main__":
    unittest.main()
