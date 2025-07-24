import torch
import numpy as np
import scipy.sparse as sp
from collections import defaultdict
from .metrics import calculate
from .type_helpers import get_unique_values


def _try_masking(mask, a: any = None):
    """
    Helper function to try and mask a given array, otherwise
    return None.
    """
    if a is None:
        return None
    return a[mask]


def _nested_default_to_normal_dict(d: defaultdict):
    if isinstance(d, (dict, defaultdict)):
        return {k: _nested_default_to_normal_dict(v) for k, v in d.items()}
    return d


def _create_nested_defaultdict(depth: int):
    if depth < 1:
        raise ValueError("Depth must be at least 1")
    if depth == 1:
        return defaultdict()
    return defaultdict(lambda: _create_nested_defaultdict(depth - 1))


def calculate_per_group(
    group_name: str,
    group_assignment: any,
    logits: torch.Tensor | np.ndarray | sp.csr_array = None,
    targets: torch.Tensor | np.ndarray | sp.csr_array = None,
    flatten_results: bool = False,
    flatten_prefix: str = "",
    best_logit_indices: torch.Tensor | np.ndarray = None,
    return_best_logit_indices: bool = False,
    **kwargs,
):
    """
    :param group_name:          for labeling the group
    :param group_assignment:    assignment which user belongs to which group to compute
                                group-based metrics
    :param logits:              prediction matrix about item relevance
    :param targets:             0/1 matrix encoding true item relevance, same shape as logits
    :param flatten_results:     whether to flatten the results' dictionary.
    :param flatten_prefix:      prefix to use for flattened results, e.g., to differentiate between
                                different splits
    :param best_logit_indices:  previously computed indices of the best logits in sorted order
    :param return_best_logit_indices:   whether to return the indices of the best logits
    :param **kwargs:            see documentation of `rmet.metrics.calculate()`

    :return: a nested dictionary depending of structure {
            `metric`: {
                `k`: {
                    `key`: {
                        `group_name`: {
                            `group_value`: `value`
                        }
                    }
                }
            },
             where `key` depends on the metric and parameters:
             - "user":   ..., # for user-based metrics if 'return_per_user'
             - "mean":   ..., # for user-based metrics if 'return_aggregated'
             - "std":    ..., # for user-based metrics if 'return_aggregated' and 'calculate_std'
             - "global": ..., # for global metrics
             or a nested dictionary with keys in the format
             - for "mean" and "global" keys due to their importance:
               `{flatten_prefix}/{metric}@{k}/{group_name}_{group_value}`
             - for "std" and "user" keys
               `{flatten_prefix}/{metric}@{k}_{key}/{group_name}_{group_value}`
    """

    # we'd like to process groups as either torch tensors or
    # numpy arrays for simpler handling
    if not isinstance(group_assignment, torch.Tensor):
        group_assignment = np.array(group_assignment)
    # unique_values = get_unique_values(group_assignment)
    unique_values = set(list(group_assignment))

    # iterate over all groups and collect results
    # metric -> k -> group_name -> group_value -> result
    results_per_group = _create_nested_defaultdict(4)
    # group_name -> group_value -> logits
    logits_per_group = _create_nested_defaultdict(2)
    for v in unique_values:
        mask_matching_samples = group_assignment == v

        # compute results for given group
        group_results = calculate(
            logits=_try_masking(mask_matching_samples, logits),
            targets=_try_masking(mask_matching_samples, targets),
            best_logit_indices=_try_masking(mask_matching_samples, best_logit_indices),
            flatten_results=flatten_results,
            # we append prefix before the group results if flattened, not per group
            flatten_prefix=flatten_prefix,
            flatten_suffix=f"/{group_name}_{v}",
            return_best_logit_indices=return_best_logit_indices,
            **kwargs,
        )

        # split results based on supplied arguments
        if return_best_logit_indices:
            metrics_results, best_logits = group_results
        else:
            metrics_results, best_logits = group_results, None

        # store results
        if flatten_results:
            # for simplicity we do the flattening in calculate()
            results_per_group.update(metrics_results)
        else:
            for m, m_results in metrics_results.items():
                for ki, ki_results in m_results.items():
                    results_per_group[m][ki][group_name][v] = ki_results
        logits_per_group[group_name][v] = best_logits

    results_per_group = _nested_default_to_normal_dict(results_per_group)
    logits_per_group = _nested_default_to_normal_dict(logits_per_group)

    if return_best_logit_indices:
        return results_per_group, logits_per_group
    return results_per_group
