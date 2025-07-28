import math
import torch
import unittest
import numpy as np
import scipy.sparse as sp
from collections import defaultdict

from rmet.metrics import (
    average_rank,
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
    supported_metrics,
)

from rmet.batch_evaluator import BatchEvaluator, EvaluatorResults
from rmet.type_helpers import std


class TestRecommenderMetrics(unittest.TestCase):

    def setUp(self):
        # top-k: [0, 1, 3], [3, 4, 2]
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

        self.item_ranks = [5, 3, 2, 4, 1]
        self.ar1 = (5 + 3 + 4) / 3
        self.ar2 = (2 + 4 + 1) / 3
        self.ar = [self.ar1, self.ar2]

        # computed item ranks should be [2, 3, 2, 1, 2],
        # as interaction counts are [1, 0, 1, 2, 1]
        # (same values are assigned same ranks, without gaps between ranks)
        self.ar1_from_targets = (2 + 3 + 1) / 3
        self.ar2_from_targets = (1 + 2 + 2) / 3
        self.ar_from_targets = [self.ar1_from_targets, self.ar2_from_targets]

        self.user_computation_results = {
            "dcg": self.dcg,
            "ndcg": self.ndcg,
            "recall": self.recall,
            "precision": self.precision,
            "hitrate": self.hitrate,
            "f_score": self.f1_user,
            "ap": self.ap,
            "rr": self.rr,
            "average_rank": self.ar,
        }
        self.metrics = list(self.user_computation_results.keys()) + ["coverage"]

    def _assert_dictionary_equality(self, d1, d2, is_close_fn):
        # check matching keys
        self.assertEqual(type(d1), type(d2))
        self.assertSetEqual(set(d1.keys()), set(d2.keys()))
        for k in d1:
            # dict might be nested
            if isinstance(d1[k], dict):
                self._assert_dictionary_equality(d1[k], d2[k], is_close_fn)
            else:
                # check matching results
                self.assertEqual(type(d1[k]), type(d2[k]))
                if isinstance(d1[k], (np.ndarray, torch.Tensor)):
                    self.assertTrue(is_close_fn(d1[k], d2[k]))
                else:
                    self.assertAlmostEqual(d1[k], d2[k], delta=1e-4)

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

    def test_average_rank(self):
        for _, fn_lookup in self.supported_types.items():
            # compute WITHOUT item ranks
            expected = fn_lookup["cast-result"](self.ar_from_targets)
            result = average_rank(
                logits=fn_lookup["cast"](self.logits),
                targets=fn_lookup["cast"](self.targets),
                k=self.k,
            )
            self.assertTrue(fn_lookup["all_close"](result, expected, atol=1e-4))

            # compute WITH item ranks
            expected = fn_lookup["cast-result"](self.ar)
            result = average_rank(
                logits=fn_lookup["cast"](self.logits),
                targets=fn_lookup["cast"](self.targets),
                k=self.k,
                item_ranks=fn_lookup["cast"](self.item_ranks),
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

            self.assertAlmostEqual(coverage(logits, self.k), self.coverage, delta=1e-4)
            self.assertAlmostEqual(coverage(logits, 2), 4 / 5, delta=1e-4)

    def test_compute_nested(self):
        for _, fn_lookup in self.supported_types.items():
            result = calculate(
                metrics=self.metrics,
                logits=fn_lookup["cast"](self.logits),
                targets=fn_lookup["cast"](self.targets),
                k=self.k,
                return_aggregated=True,
                return_per_user=True,
                calculate_std=True,
                flatten_results=False,
                n_items=None,
                best_logit_indices=None,
                return_best_logit_indices=False,
                item_ranks=fn_lookup["cast"](self.item_ranks),
            )

            expected = defaultdict(lambda: defaultdict(lambda: dict()))
            for m, r in self.user_computation_results.items():
                expected[m][self.k]["user"] = fn_lookup["cast-result"](r)
                expected[m][self.k]["mean"] = np.mean(r).item()
                # ddof to adjust to degrees of freedom = 1 for std computation in PyTorch
                expected[m][self.k]["std"] = np.std(r, ddof=1).item()
            expected.update({"coverage": {self.k: {"global": self.coverage}}})
            # transform to normal dictionary
            expected = {k: dict(v) for k, v in expected.items()}

            self._assert_dictionary_equality(result, expected, fn_lookup["all_close"])

    def test_compute_flattened(self):

        prefix = "<my-prefix>/"
        suffix = "/<my-suffix>"

        for _, fn_lookup in self.supported_types.items():
            result = calculate(
                metrics=self.metrics,
                logits=fn_lookup["cast"](self.logits),
                targets=fn_lookup["cast"](self.targets),
                k=self.k,
                return_aggregated=True,
                return_per_user=True,
                calculate_std=True,
                flatten_results=True,
                flatten_prefix=prefix,
                flatten_suffix=suffix,
                n_items=None,
                best_logit_indices=None,
                return_best_logit_indices=False,
                item_ranks=fn_lookup["cast"](self.item_ranks),
            )

            expected = dict()
            for m, k in self.user_computation_results.items():
                metric_name = f"{prefix}{m}@{self.k}"
                expected[f"{metric_name}_user" + suffix] = fn_lookup["cast-result"](k)
                expected[metric_name + suffix] = np.mean(k).item()
                # ddof to adjust to degrees of freedom = 1 for std computation in PyTorch
                expected[f"{metric_name}_std" + suffix] = np.std(k, ddof=1).item()
            expected.update({f"{prefix}coverage@{self.k}{suffix}": self.coverage})

            self._assert_dictionary_equality(result, expected, fn_lookup["all_close"])

    def test_best_indices(self):
        for _, fn_lookup in self.supported_types.items():
            result, best_indices = calculate(
                metrics=self.metrics,
                logits=fn_lookup["cast"](self.logits),
                targets=fn_lookup["cast"](self.targets),
                k=self.k,
                return_best_logit_indices=True,
                return_per_user=True,
                flatten_results=True,
                item_ranks=fn_lookup["cast"](self.item_ranks),
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
                return_per_user=True,
                flatten_results=True,
                item_ranks=fn_lookup["cast"](self.item_ranks),
            )

            self._assert_dictionary_equality(
                result, result_on_best_logits, fn_lookup["all_close"]
            )

    def test_single_metric_vs_list(self):
        # test string metrics
        for m in supported_metrics:
            for _, fn_lookup in self.supported_types.items():
                r_single_metric = calculate(
                    metrics=m,
                    logits=fn_lookup["cast"](self.logits),
                    targets=fn_lookup["cast"](self.targets),
                    k=self.k,
                )

                r_metrics_list = calculate(
                    metrics=m,
                    logits=fn_lookup["cast"](self.logits),
                    targets=fn_lookup["cast"](self.targets),
                    k=self.k,
                )

                self._assert_dictionary_equality(
                    r_single_metric, r_metrics_list, fn_lookup["all_close"]
                )

    def test_accepted_metrics(self):
        # test enum metrics
        for m in supported_metrics:
            for _, fn_lookup in self.supported_types.items():
                calculate(
                    metrics=[m],
                    logits=fn_lookup["cast"](self.logits),
                    targets=fn_lookup["cast"](self.targets),
                    k=self.k,
                )

        # test string metrics
        for m in supported_metrics:
            for _, fn_lookup in self.supported_types.items():
                calculate(
                    metrics=[str(m)],
                    logits=fn_lookup["cast"](self.logits),
                    targets=fn_lookup["cast"](self.targets),
                    k=self.k,
                )

        # test mix between string and Enum metrics
        for i, m in enumerate(supported_metrics):
            for _, fn_lookup in self.supported_types.items():
                m = m if i % 2 == 0 else str(m)
                print(m)
                calculate(
                    metrics=[str(m)],
                    logits=fn_lookup["cast"](self.logits),
                    targets=fn_lookup["cast"](self.targets),
                    k=self.k,
                )

        # test some random metrics that should fail
        not_supported_metrics = ["foo", "bar", "foobar", "barfoo"]
        for m in not_supported_metrics:
            for _, fn_lookup in self.supported_types.items():
                with self.assertRaises(ValueError):
                    calculate(
                        metrics=[m],
                        logits=fn_lookup["cast"](self.logits),
                        targets=fn_lookup["cast"](self.targets),
                        k=self.k,
                    )

    def test_batch_evaluator(self):
        for _, fn_lookup in self.supported_types.items():
            batch_evaluator = BatchEvaluator(
                metrics=self.metrics,
                top_k=self.k,
                calculate_std=True,
                n_items=None,
                **{"item_ranks": fn_lookup["cast"](self.item_ranks)},
            )

            for i in range(len(self.logits)):
                batch_evaluator.eval_batch(
                    user_indices=fn_lookup["cast-result"]([i]),
                    # slice to maintain batch
                    logits=fn_lookup["cast"](self.logits[i : i + 1]),
                    targets=fn_lookup["cast"](self.targets[i : i + 1]),
                )
            result = batch_evaluator.get_results()

            expected = EvaluatorResults(
                user_level_metrics={},
                aggregated_metrics={},
                user_indices=None,
                user_top_k=None,
            )

            for m, m_result in self.user_computation_results.items():
                metric_name = f"{m}@{self.k}"
                m_result = fn_lookup["cast-result"](m_result)
                expected.user_level_metrics[metric_name] = m_result
                expected.aggregated_metrics[metric_name] = m_result.mean().item()
                # ddof to adjust to degrees of freedom = 1 for std computation in PyTorch,
                # as we want to have the same results between NumPy and PyTorch
                expected.aggregated_metrics[metric_name + "_std"] = std(m_result)
            expected.aggregated_metrics.update({f"coverage@{self.k}": self.coverage})

            self._assert_dictionary_equality(
                result.user_level_metrics, expected.user_level_metrics, np.allclose
            )
            self._assert_dictionary_equality(
                result.aggregated_metrics, expected.aggregated_metrics, np.allclose
            )


if __name__ == "__main__":
    unittest.main()
