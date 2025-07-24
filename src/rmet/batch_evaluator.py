import torch
import numpy as np
import scipy.sparse as sp
from typing import Iterable
from natsort import natsorted
from dataclasses import dataclass
from collections import defaultdict

from .metrics import (
    calculate,
    supported_metrics,
    supported_user_accuracy_metrics,
    supported_global_metrics,
    supported_user_beyond_accuracy_metrics,
)

from .groups import calculate_per_group
from .type_helpers import std


@dataclass
class EvaluatorResults:
    aggregated_metrics: dict[str, torch.Tensor]
    user_level_metrics: dict[str, torch.Tensor]
    user_indices: torch.Tensor
    user_top_k: torch.Tensor


class BatchEvaluator:
    """
    Helper class that supports evaluating recommendations batch after batch, stores these
    internally and can aggregate the results after the final batch.
    """

    def __init__(
        self,
        metrics: Iterable[str],
        top_k: Iterable[int],
        calculate_std: bool = True,
        n_items: int = None,
        metric_prefix: str = "",
        **kwargs,
    ):
        """
        Initializes a BatchEvaluator instance based on the provided config.

        :param metrics:         Metric name or list of metrics to compute. Check out 'supported_metrics'
                                for a list of all available metrics.
        :param k:               Top k items to consider
        :param calculate_std:   Whether to calculate the standard deviation for the aggregated results
        :param n_items:         Number of items in dataset, in case targets do not contain 'labels' for all items
        :param metric_prefix:   Prefix for the computed metrics
        :param kwargs:          Additional parameters that are passed to metric computations, e.g., item_ranks
        """
        self.metrics = metrics
        self.top_k = top_k
        self.calculate_std = calculate_std

        self.n_items = n_items
        self.metric_prefix = metric_prefix
        self.calculation_kwargs = kwargs

        # ensure that only valid metrics are supplied
        invalid_metrics = set(self.metrics) - set(supported_metrics)
        if len(invalid_metrics) > 0:
            raise ValueError(
                f"Metric(s) {invalid_metrics} are not supported. "
                f"Select metrics from {supported_metrics}."
            )

        # determine to which kind of metrics the different metrics belong to
        # need to know whether we can compute the metrics per batch, or only
        # after the final batch has been processed
        # Note: We don't want to compute everything at the end, as we might not be able
        #       to make use GPU computations like that.
        self._user_metrics = set(self.metrics).intersection(
            set(supported_user_accuracy_metrics).union(
                supported_user_beyond_accuracy_metrics
            )
        )
        self._dist_metrics = set(self.metrics).intersection(supported_global_metrics)

        self._are_results_available = False
        self._user_indices = None
        self._user_top_k = None
        self._user_group_assignments = None

        # internal storage for the results
        self._user_level_results = None
        self._user_group_level_results = None
        self._reset_internal_dict()

    def _reset_internal_dict(self):
        """
        Resets the internal memory on computed and gathered metrics.
        """
        self._user_level_results = defaultdict(lambda: list())
        self._user_group_level_results = defaultdict(lambda: list())
        self._user_indices = list()
        self._user_top_k = list()
        self._user_group_assignments = defaultdict(lambda: list())
        self._are_results_available = False

    def _detach_and_to_cpu(self, a: any):
        if isinstance(a, dict):
            return {k: self._detach_and_to_cpu(v) for k, v in a.items()}
        elif isinstance(a, torch.Tensor):
            return a.detach().cpu()
        else:
            return a

    def _concat_list_elements(self, a: tuple | list):
        if len(a) > 0:
            if isinstance(a[0], torch.Tensor):
                return torch.cat(a)
            elif isinstance(a[0], np.ndarray):
                return np.concatenate(a)
        return a

    def _concat_dict_list_values(self, d: dict):
        return {k: self._concat_list_elements(v) for k, v in d.items()}

    @staticmethod
    def _natsort_dict(d: dict):
        return {k: d[k] for k in natsorted(d.keys())}

    def _calculate_user_metrics(self, logits: torch.Tensor, y_true: torch.Tensor):
        """
        Wrapper function to compute user-based metrics, e.g., recall and precision
        """
        user_level_results, top_k_indices = calculate(
            metrics=self._user_metrics,
            logits=logits,
            targets=y_true,
            k=self.top_k,
            return_aggregated=False,
            return_per_user=True,
            flatten_results=True,
            flatten_prefix=self.metric_prefix,
            n_items=self.n_items,
            return_best_logit_indices=True,
            **self.calculation_kwargs,
        )

        # drop the "_user" part as it's not relevant for us
        user_level_results = {
            k.replace("_user", ""): v
            for k, v in user_level_results.items()
            if k.endswith("_user")
        }
        return user_level_results, top_k_indices

    def _calculate_user_group_metrics(
        self,
        top_k_indices: torch.Tensor,
        y_true: torch.Tensor,
        group_assignments: dict[str, Iterable],
    ):
        """
        Wrapper function to compute user-based group metrics, e.g., recall and precision per group
        """
        results = {}
        for group_name, group_assignment in group_assignments.items():
            user_level_group_results = calculate_per_group(
                group_name=group_name,
                group_assignment=group_assignment,
                metrics=self._user_metrics,
                targets=y_true,
                k=self.top_k,
                return_aggregated=False,
                return_per_user=True,
                flatten_results=True,
                flatten_prefix=self.metric_prefix,
                n_items=self.n_items,
                best_logit_indices=top_k_indices,
                **self.calculation_kwargs,
            )

            # drop the "_user" part as it's not relevant for us
            for k, v in user_level_group_results.items():
                results[k.replace("_user", "")] = v
        return results

    def _calculate_distribution_metrics(self, user_top_k):
        """
        Wrapper function to compute distribution-based metrics, e.g., coverage.
        """
        results = calculate(
            metrics=self._dist_metrics,
            k=self.top_k,
            return_aggregated=True,
            return_per_user=False,
            flatten_results=True,
            flatten_prefix=self.metric_prefix,
            best_logit_indices=user_top_k,
            n_items=self.n_items,
            **self.calculation_kwargs,
        )
        return results

    def _calculate_group_distribution_metrics(
        self, user_top_k, group_assignments: dict[str, Iterable]
    ):
        """
        Wrapper function to compute group-based distribution metrics, e.g., coverage per group.
        """
        results = {}
        for group_name, group_assignment in group_assignments.items():
            group_results = calculate_per_group(
                group_name=group_name,
                group_assignment=group_assignment,
                metrics=self._dist_metrics,
                k=self.top_k,
                return_aggregated=True,
                return_per_user=False,
                flatten_results=True,
                flatten_prefix=self.metric_prefix,
                best_logit_indices=user_top_k,
                n_items=self.n_items,
                **self.calculation_kwargs,
            )
            results.update(group_results)
        return results

    @torch.no_grad()
    def eval_batch(
        self,
        user_indices: torch.Tensor | np.ndarray,
        logits: torch.Tensor | np.ndarray | sp.csr_array,
        targets: torch.Tensor | np.ndarray | sp.csr_array,
        group_assignments: dict[str, Iterable] = None,
    ):
        """
        Evaluates a batch of logits and their true targets and stores the results internally.
        To retrieve the results, call `get_results`.

        :param user_indices:    indices of users in batch
        :param logits:          predicted ratings, expected shape is (batch_size, n_items)
        :param targets:         target ratings, expected  (batch_size, n_items)
        :param group_assignments: if given, metrics will also be computed on group-level
        """
        if logits.shape != targets.shape:
            raise ValueError(
                f"Logits and true labels must have the same shape ({logits.shape} != {targets.shape})"
            )
        if logits.ndim != 2:
            raise ValueError(
                f"Logits and targets are expected to have 2 dimensions "
                f"instead of {logits.ndim}."
            )

        if self.n_items is None:
            self.n_items = targets.shape[-1]

        # compute and store user-based metrics
        user_metrics, top_k_indices = self._calculate_user_metrics(logits, targets)
        user_metrics = self._detach_and_to_cpu(user_metrics)
        for metric_name, metric_values in user_metrics.items():
            self._user_level_results[metric_name].append(metric_values)

        # compute and store user group-level metrics
        if group_assignments is not None:
            user_group_metrics = self._calculate_user_group_metrics(
                top_k_indices, targets, group_assignments
            )
            user_group_metrics = self._detach_and_to_cpu(user_group_metrics)
            for metric_name, metric_values in user_group_metrics.items():
                self._user_group_level_results[metric_name].append(metric_values)

            for g, assignments in group_assignments.items():
                # for simplicity we assume assignmets are convertable to numpy arrays
                self._user_group_assignments[g].append(np.array(assignments))

        user_indices = self._detach_and_to_cpu(user_indices)
        top_k_indices = self._detach_and_to_cpu(top_k_indices)

        self._user_indices.append(user_indices)
        self._user_top_k.append(top_k_indices)

        self._are_results_available = True

    def get_results(self, reset_state: bool = True) -> EvaluatorResults:
        """
        Retrieves all user- and distribution-based results.

        :param reset_state: Whether to reset internal memory. If true, `get_results` can only
                            be called once.
        """
        if not self._are_results_available:
            raise RuntimeError(
                "No results have yet been calculated. Call `eval_batch` before "
                "calling this method."
            )

        user_top_k = self._concat_list_elements(self._user_top_k)
        user_indices = self._concat_list_elements(self._user_indices)

        aggregated_results, user_level_results = {}, {}
        if len(self._user_metrics) > 0:
            # join user metrics across all batches
            all_user_results = self._user_level_results | self._user_group_level_results
            user_level_results = self._concat_dict_list_values(all_user_results)

            # aggregate the results
            aggregated_results = {
                k: v.mean().item() for k, v in user_level_results.items()
            }

            if self.calculate_std:
                # determine all flattened metrics based on user-level
                # (group-level results have groups appended)
                # e.g., ndcg@10 -> ndcg@10_std
                user_metrics = list(self._user_level_results.keys())
                # first try replace longest, then shortest, so that
                # we don't replace ndcg@100 with the value for ndcg@10
                user_metrics = sorted(user_metrics, key=len, reverse=True)
                std_replacement_strings = {m: m + "_std" for m in user_metrics}

                for k, v in user_level_results.items():
                    for original, replacement in std_replacement_strings.items():
                        if original in k:
                            # limit replacement to a single time
                            k = k.replace(original, replacement)
                            break
                    aggregated_results[k] = std(v)

        # calculate distribution metrics
        if len(self._dist_metrics) > 0:
            distribution_results = self._calculate_distribution_metrics(user_top_k)
            aggregated_results.update(distribution_results)

            # calculate group-level distribution metrics
            if len(self._user_group_assignments) > 0:
                # first join all the gathered groups
                group_assignments = self._concat_dict_list_values(
                    self._user_group_assignments
                )
                group_distribution_results = self._calculate_group_distribution_metrics(
                    user_top_k, group_assignments
                )
                aggregated_results.update(group_distribution_results)

        if reset_state:
            self._reset_internal_dict()

        return EvaluatorResults(
            # employ natural sorting on metrics, so that, e.g., @20 comes before @100
            aggregated_metrics=self._natsort_dict(aggregated_results),
            user_level_metrics=self._natsort_dict(user_level_results),
            user_indices=user_indices,
            user_top_k=user_top_k,
        )
