from typing import Iterable
import torch
import numpy as np
import scipy.sparse as sp
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

from .type_helpers import (
    std,
)


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

        Params:
        - metrics:          Metric name or list of metrics to compute. Check out 'supported_metrics'
                            for a list of all available metrics.
        - k:                Top k items to consider
        - calculate_std:    Whether to calculate the standard deviation for the aggregated results
        - n_items:          Number of items in dataset, in case targets do not contain 'labels' for all items
        - metric_prefix:    Prefix for the computed metrics
        - kwargs:           Additional parameters that are passed to metric computations, e.g., item_ranks
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

        # internal storage for the results
        self._user_level_results = None
        self._reset_internal_dict()

    def _reset_internal_dict(self):
        """
        Resets the internal memory on computed and gathered metrics.
        """
        self._user_level_results = defaultdict(lambda: list())
        self._user_indices = list()
        self._user_top_k = list()
        self._are_results_available = False

    def _store_user_metrics(
        self, results: dict[str, float | np.ndarray | torch.Tensor]
    ):
        """
        Updates internal memory based on new results.
        """
        for metric_name, metric_values in results.items():
            if isinstance(metric_values, torch.Tensor):
                metric_values = metric_values.detach().cpu().numpy()
            self._user_level_results[metric_name].append(metric_values)

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

    @torch.no_grad()
    def eval_batch(
        self,
        user_indices: torch.Tensor | np.ndarray,
        logits: torch.Tensor | np.ndarray | sp.csr_array,
        targets: torch.Tensor | np.ndarray | sp.csr_array,
    ):
        """
        Evaluates a batch of logits and their true targets and stores the results internally.
        To retrieve the results, call `get_results`.

        Params:
        - user_indices: indices of users in batch
        - logits: predicted ratings, expected shape is (batch_size, n_items)
        - targets: target ratings, expected  (batch_size, n_items)
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

        user_metrics, top_k_indices = self._calculate_user_metrics(logits, targets)
        self._store_user_metrics(user_metrics)

        if isinstance(user_indices, torch.Tensor):
            user_indices = user_indices.detach().cpu()
        if isinstance(top_k_indices, torch.Tensor):
            top_k_indices = top_k_indices.detach().cpu()

        self._user_indices.append(user_indices)
        self._user_top_k.append(top_k_indices)

        self._are_results_available = True

    def get_results(self, reset_state: bool = True) -> EvaluatorResults:
        """
        Retrieves all user- and distribution-based results.

        Params:
        - reset_state: Whether to reset internal memory. If true, `get_results` can only
                       be called once.
        """
        if not self._are_results_available:
            raise RuntimeError(
                "No results have yet been calculated. Call `eval_batch` before "
                "calling this method."
            )

        if isinstance(self._user_top_k[0], np.ndarray):
            user_top_k = np.concatenate(self._user_top_k)
            user_indices = np.concatenate(self._user_indices)
        else:
            user_top_k = torch.cat(self._user_top_k)
            user_indices = torch.cat(self._user_indices)

        aggregated_results, user_level_results = {}, {}
        if len(self._user_metrics) > 0:
            # join user metrics across all batches
            user_level_results = {
                k: np.concatenate(v) for k, v in self._user_level_results.items()
            }
            # aggregate the results
            aggregated_results = {
                k: v.mean().item() for k, v in user_level_results.items()
            }
            if self.calculate_std:
                aggregated_results.update(
                    {f"{k}_std": std(v) for k, v in user_level_results.items()}
                )

        # calculate distribution metrics
        if len(self._dist_metrics) > 0:
            distribution_results = self._calculate_distribution_metrics(user_top_k)
            aggregated_results.update(distribution_results)

        # employ natural sorting on metrics, so that, e.g., @20 comes before @100
        aggregated_results = {
            k: aggregated_results[k] for k in natsorted(aggregated_results.keys())
        }

        if reset_state:
            self._reset_internal_dict()

        return EvaluatorResults(
            aggregated_metrics=aggregated_results,
            user_level_metrics=user_level_results,
            user_indices=user_indices,
            user_top_k=user_top_k,
        )
