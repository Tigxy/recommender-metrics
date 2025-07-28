import math
import torch
import unittest
import numpy as np
import scipy.sparse as sp

from rmet.metrics import (
    calculate,
    supported_metrics,
)

from rmet.groups import calculate_per_group
from rmet.batch_evaluator import BatchEvaluator, EvaluatorResults


class TestRecommenderGroupMetrics(unittest.TestCase):

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

        self.k = [3, 5]

        rng = np.random.default_rng(42)
        self.n_users, self.n_items = 5, 10
        self.logits = rng.random(size=(self.n_users, self.n_items))
        self.targets = rng.integers(low=0, high=2, size=(self.n_users, self.n_items))
        self.item_ranks = np.argsort(self.targets.sum(0))

        self.group_assignments = {
            "G1": rng.choice(["A", "B", "C"], size=self.n_users),
            "G2": rng.choice(["X", "Y", "Z"], size=self.n_users),
        }

        self.metrics = supported_metrics

    def _extract_at_level(
        self, d: dict, depth: int, key: any, keep_level: bool = False
    ):
        """
        Extracts the dictionary values of a given depth that match a certain key
        of a nested dictionary
        """
        if isinstance(d, dict):
            new_dict = {}
            for k, v in d.items():
                if depth == 0:
                    if k == key:
                        if keep_level:
                            new_dict[k] = v
                        else:
                            return v
                elif depth > 0:
                    new_dict[k] = self._extract_at_level(
                        v, depth - 1, key, keep_level=keep_level
                    )
            return new_dict
        return d

    def _extract_at_levels(
        self, d: dict, depth_keys: dict[int, any], keep_levels: bool = False
    ):
        """
        Extracts the values of a nested dictionary based on the given depths and corresponding keys.
        """
        # sort by depths so that we go from most shallow to deepest depths for efficiency
        sorted_depths = sorted(depth_keys.items(), key=lambda x: x[0], reverse=False)

        for i, (depth, key) in enumerate(sorted_depths):
            if not keep_levels:
                # -i to account for dropped depths
                depth -= i
            d = self._extract_at_level(d, depth, key, keep_level=keep_levels)

        return d

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
                    # ignore values that are NaN (equality comparison doesn't work for them)
                    if math.isnan(d1[k]) and math.isnan(d2[k]):
                        pass
                    else:
                        self.assertAlmostEqual(d1[k], d2[k], delta=1e-4)

    def test_compute_nested(self):
        for _, fn_lookup in self.supported_types.items():
            for group_name, group_assignment in self.group_assignments.items():
                results = calculate_per_group(
                    metrics=self.metrics,
                    group_name=group_name,
                    group_assignment=group_assignment,
                    logits=fn_lookup["cast"](self.logits),
                    targets=fn_lookup["cast"](self.targets),
                    k=self.k,
                    return_aggregated=True,
                    return_per_user=True,
                    calculate_std=True,
                    flatten_results=False,
                )

                # test results
                for group in np.unique(group_assignment):
                    mask = group_assignment == group

                    # we can use this function as it's already tested elsewhere
                    expected = calculate(
                        metrics=self.metrics,
                        logits=fn_lookup["cast"](self.logits)[mask],
                        targets=fn_lookup["cast"](self.targets)[mask],
                        k=self.k,
                        return_aggregated=True,
                        return_per_user=True,
                        calculate_std=True,
                        flatten_results=False,
                    )

                    self._assert_dictionary_equality(
                        self._extract_at_levels(results, {2: group_name, 3: group}),
                        expected,
                        fn_lookup["all_close"],
                    )

    def test_compute_flattened(self):
        for _, fn_lookup in self.supported_types.items():
            for group_name, group_assignment in self.group_assignments.items():
                results = calculate_per_group(
                    metrics=self.metrics,
                    group_name=group_name,
                    group_assignment=group_assignment,
                    logits=fn_lookup["cast"](self.logits),
                    targets=fn_lookup["cast"](self.targets),
                    k=self.k,
                    return_aggregated=True,
                    return_per_user=True,
                    calculate_std=True,
                    flatten_results=True,
                )

                # test results
                for group in np.unique(group_assignment):
                    mask = group_assignment == group
                    group_result_suffix = f"/{group_name}_{group}"

                    # we can use this function as it's already tested elsewhere
                    expected = calculate(
                        metrics=self.metrics,
                        logits=fn_lookup["cast"](self.logits)[mask],
                        targets=fn_lookup["cast"](self.targets)[mask],
                        k=self.k,
                        return_aggregated=True,
                        return_per_user=True,
                        calculate_std=True,
                        flatten_results=True,
                        flatten_suffix=group_result_suffix,
                    )

                    self._assert_dictionary_equality(
                        {
                            k: v
                            for k, v in results.items()
                            if k.endswith(group_result_suffix)
                        },
                        expected,
                        fn_lookup["all_close"],
                    )

    def test_compute_with_best_logits_input(self):
        for _, fn_lookup in self.supported_types.items():
            for group_name, group_assignment in self.group_assignments.items():
                _, best_logits_indices = calculate(
                    metrics=self.metrics,
                    logits=fn_lookup["cast"](self.logits),
                    targets=fn_lookup["cast"](self.targets),
                    k=self.k,
                    return_best_logit_indices=True,
                )

                results = calculate_per_group(
                    metrics=self.metrics,
                    group_name=group_name,
                    group_assignment=group_assignment,
                    best_logit_indices=fn_lookup["cast-result"](best_logits_indices),
                    targets=fn_lookup["cast"](self.targets),
                    k=self.k,
                    return_aggregated=True,
                    return_per_user=True,
                    calculate_std=True,
                    flatten_results=True,
                    n_items=self.n_items,
                )

                # test results
                for group in np.unique(group_assignment):
                    mask = group_assignment == group
                    group_result_suffix = f"/{group_name}_{group}"

                    # we can use this function as it's already tested elsewhere
                    expected = calculate(
                        metrics=self.metrics,
                        best_logit_indices=fn_lookup["cast-result"](
                            best_logits_indices
                        )[mask],
                        targets=fn_lookup["cast"](self.targets)[mask],
                        k=self.k,
                        return_aggregated=True,
                        return_per_user=True,
                        calculate_std=True,
                        flatten_results=True,
                        flatten_suffix=group_result_suffix,
                        n_items=self.n_items,
                    )

                    self._assert_dictionary_equality(
                        {
                            k: v
                            for k, v in results.items()
                            if k.endswith(group_result_suffix)
                        },
                        expected,
                        fn_lookup["all_close"],
                    )

    def test_group_batch_evaluator(self):
        for _, fn_lookup in self.supported_types.items():
            batch_evaluator = BatchEvaluator(
                metrics=self.metrics,
                top_k=self.k,
                calculate_std=True,
                n_items=None,
                item_ranks=fn_lookup["cast-result"](self.item_ranks),
            )

            for i in range(len(self.logits)):
                batch_evaluator.eval_batch(
                    user_indices=fn_lookup["cast-result"]([i]),
                    # slice to maintain batch
                    logits=fn_lookup["cast"](self.logits[i : i + 1]),
                    targets=fn_lookup["cast"](self.targets[i : i + 1]),
                    group_assignments={
                        k: v[i : i + 1] for k, v in self.group_assignments.items()
                    },
                )
            result = batch_evaluator.get_results()

            expected = EvaluatorResults(
                user_level_metrics={},
                aggregated_metrics={},
                user_indices=None,
                user_top_k=None,
            )

            # compute expected results
            overall_results = calculate(
                metrics=self.metrics,
                logits=fn_lookup["cast"](self.logits),
                targets=fn_lookup["cast"](self.targets),
                k=self.k,
                calculate_std=True,
                return_per_user=True,
                return_aggregated=True,
                flatten_results=True,
                item_ranks=fn_lookup["cast-result"](self.item_ranks),
            )
            expected.aggregated_metrics.update(
                {k: v for k, v in overall_results.items() if "_user" not in k}
            )
            expected.user_level_metrics.update(
                {
                    k.replace("_user", ""): v
                    for k, v in overall_results.items()
                    if "_user" in k
                }
            )

            for group_name, group_assignment in self.group_assignments.items():
                for group in np.unique(group_assignment):
                    mask = group_assignment == group
                    group_result_suffix = f"/{group_name}_{group}"

                    # we can use this function as it's already tested elsewhere
                    expected_per_group_value = calculate(
                        metrics=self.metrics,
                        logits=fn_lookup["cast-result"](self.logits)[mask],
                        targets=fn_lookup["cast"](self.targets)[mask],
                        k=self.k,
                        return_aggregated=True,
                        return_per_user=True,
                        calculate_std=True,
                        flatten_results=True,
                        flatten_suffix=group_result_suffix,
                        n_items=self.n_items,
                        item_ranks=fn_lookup["cast-result"](self.item_ranks),
                    )

                    expected.aggregated_metrics.update(
                        {
                            k: v
                            for k, v in expected_per_group_value.items()
                            if "_user" not in k
                        }
                    )
                    expected.user_level_metrics.update(
                        {
                            k.replace("_user", ""): v
                            for k, v in expected_per_group_value.items()
                            if "_user" in k
                        }
                    )

            self._assert_dictionary_equality(
                result.user_level_metrics,
                expected.user_level_metrics,
                fn_lookup["all_close"],
            )
            self._assert_dictionary_equality(
                result.aggregated_metrics,
                expected.aggregated_metrics,
                fn_lookup["all_close"],
            )


if __name__ == "__main__":
    unittest.main()
