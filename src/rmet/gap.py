from collections import defaultdict

import torch
import itertools
from typing import Iterable
from .user_feature import UserFeature
from .metrics import calculate, MetricEnum


def __mean(v):
    return torch.mean(v).item() if isinstance(v, torch.Tensor) else v


# def calculate(metrics: tuple, logits: torch.Tensor, targets=None, k: Union[int, list] = 10,
#               return_aggregated: bool = True, return_individual: bool = False,
#               flatten_results: bool = False, flattened_results_prefix: str = ""):
#     """
#     Computes the values for a given list of metrics.
#
#     :param metrics: The list of metrics to compute. Check out 'supported_metrics' for a list of names.
#     :param logits: prediction matrix about item relevance
#     :param targets: 0/1 matrix encoding true item relevance, same shape as logits
#     :param k: top k items to consider
#     :param return_aggregated: Whether aggregated metric results should be returned.
#     :param return_individual: Whether the results for individual users should be returned
#     :param flatten_results: Whether to flatten the results' dictionary. Key is of format "{prefix}{metric}@{k}"
#     :param flattened_results_prefix: Prefix to prepend to the flattened results key.
#     :return: a dictionary containing ...
#         {metric_name: value} if 'return_aggregated=True', and/or
#         {<metric_name>_individual: list_of_values} if 'return_individual=True'
#     """


def _calculate_for_feature(
    group: UserFeature,
    metrics: Iterable[MetricEnum],
    logits: torch.Tensor,
    targets: torch.Tensor = None,
    k: int = 10,
    return_individual=False,
):
    """
    Computes the values for a given list of metrics for the users with different demographic features.
    Moreover, pairwise differences between the group metrics are also calculated.

    In the context of gender, the metrics would be computed for male users and female users individually.

    :param group: A user feature for which to compute the metrics.
    :param metrics: The list of metrics to compute. Check out 'supported_metrics' for a list of names.
    :param logits: prediction matrix about item relevance
    :param targets: 0/1 matrix encoding true item relevance, same shape as logits
    :param k: top k items to consider
    :param return_individual: Whether the results for individual users should also be returned.
    :return: A dictionary of metrics computed for the individual user groups, and their pairwise differences
     in the form of {metric_name: value} pairs.
    """

    results = {}

    # calculate metrics for users of a single feature
    for lbl, indices in group:
        t = targets[indices] if targets is not None else None
        results[f"{group.name}_{lbl}"] = calculate(
            metrics, logits[indices], t, k, return_individual=return_individual
        )

    pairs = list(itertools.combinations(group.unique_labels, 2))
    for a, b in pairs:
        pair_results = dict()
        for m in metrics:
            pair_results[str(m)] = (
                results[f"{group.name}_{a}"][m] - results[f"{group.name}_{b}"][m]
            )
        results[f"{group.name}_{a}-{b}"] = pair_results
    return results


def _dict_difference(first: any, second: any):
    if not isinstance(first, type(second)):
        raise ValueError(f"Types mismatch! ({type(first)=} vs {type(second)=})")

    if isinstance(first, dict):
        if set(first.keys()) != set(second.keys()):
            raise ValueError("Dictionaries do not have the same keys!")
        return {k: _dict_difference(first[k], second[k]) for k in first.keys()}

    else:
        return first - second


def calculate_for_feature(
    group: UserFeature,
    metrics: Iterable[MetricEnum | str],
    logits: torch.Tensor,
    targets: torch.Tensor = None,
    k: int | Iterable[int] = 10,
    return_individual: bool = False,
    flatten_results: bool = False,
    flattened_parts_separator: str = "/",
    flattened_results_prefix: str = "",
):
    """
    Computes the values for a given list of metrics for the users with different demographic features.
    Moreover, pairwise differences between the group metrics are also calculated.

    In the context of gender, the metrics would be computed for male users and female users individually.

    :param group: A user feature for which to compute the metrics.
    :param metrics: The list of metrics to compute. Check out 'supported_metrics' for a list of names.
    :param logits: prediction matrix about item relevance
    :param targets: 0/1 matrix encoding true item relevance, same shape as logits
    :param k: top k items to consider
    :param return_individual: Whether the results for individual users should also be returned.
    :param flatten_results: Whether to flatten the results' dictionary.
                            Key is of format "{prefix}/{metric}@{k}/{feature_group}" for separator "/"
    :param flattened_parts_separator: How to separate the individual parts of the flattened key
    :param flattened_results_prefix: Prefix to prepend to the flattened results key.
    :return: A dictionary of metrics computed for the individual user groups, and their pairwise differences
     in the form of {metric_name: value} pairs.
    """
    results = dict()

    # calculate metrics for users of a single feature
    for lbl, indices in group:
        t = targets[indices] if targets is not None else None
        results[f"{group.name}_{lbl}"] = calculate(
            metrics, logits[indices], t, k, return_individual=return_individual
        )

    # calculate the differences between features
    pairs = list(itertools.combinations(group.unique_labels, 2))
    for a, b in pairs:
        # only iterate over metrics, and not all values in results, as it might contain
        # results on the user-level as well
        results[f"{group.name}_{a}-{b}"] = {
            str(m): _dict_difference(
                results[f"{group.name}_{a}"][m], results[f"{group.name}_{b}"][m]
            )
            for m in metrics
        }

    if flatten_results:
        flattened_results = dict()
        for group, group_results in results.items():
            for metric, metric_results in group_results.items():
                for k, results_for_k in metric_results.items():
                    # collect parts out of which final key should be generated
                    key_parts = (
                        [flattened_results_prefix]
                        if flattened_results_prefix
                        else list()
                    )
                    key_parts.extend([f"{metric}@{k}", group])

                    key = flattened_parts_separator.join(key_parts)
                    flattened_results[key] = results_for_k
        return flattened_results

    return results
