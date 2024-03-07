import torch
from typing import Iterable
from enum import StrEnum, auto
from collections import defaultdict


class MetricEnum(StrEnum):
    DCG = auto()
    NDCG = auto()
    Precision = auto()
    Recall = auto()
    F_Score = auto()
    Hitrate = auto()
    Coverage = auto()

    def __str__(self):
        return self.value


def _get_top_k(logits: torch.Tensor, k=10, logits_are_top_indices: bool = False, sorted: bool = True):
    """
    Gets the top-k indices for the logits

    :param logits: prediction matrix about item relevance
    :param k: top k items to consider
    :param logits_are_top_indices: whether logits are already top-k sorted indices
    :param sorted: whether indices should be returned in sorted order
    """
    return logits[:, :k] if logits_are_top_indices else logits.topk(k, dim=-1, sorted=sorted).indices


def dcg(logits: torch.Tensor, targets: torch.Tensor, k=10, logits_are_top_indices: bool = False):
    """
    Computes the Discounted Cumulative Gain (DCG) for items.

    :param logits: prediction matrix about item relevance
    :param targets: 0/1 matrix encoding true item relevance, same shape as logits
    :param k: top k items to consider
    :param logits_are_top_indices: whether logits are already top-k sorted indices
    """
    top_indices = _get_top_k(logits, k, logits_are_top_indices)
    discount = 1 / torch.log2(torch.arange(1, k + 1) + 1)
    discount = discount.to(device=logits.device)
    relevancy_scores = torch.gather(targets, dim=-1, index=top_indices)
    return relevancy_scores.float() @ discount


def ndcg(logits: torch.Tensor, targets: torch.Tensor, k: int = 10, logits_are_top_indices: bool = False):
    """
    Computes the Normalized Discounted Cumulative Gain (nDCG) for items.

    :param logits: prediction matrix about item relevance
    :param targets: 0/1 matrix encoding true item relevance, same shape as logits
    :param k: top k items to consider
    :param logits_are_top_indices: whether logits are already top-k sorted indices
    """
    if k <= 0:
        raise ValueError("k is required to be positive!")

    normalization = dcg(targets, targets, k)
    normalization = normalization.to(device=logits.device)
    ndcg = dcg(logits, targets, k, logits_are_top_indices) / normalization

    return ndcg


def precision(logits: torch.Tensor, targets: torch.Tensor, k: int = 10, logits_are_top_indices: bool = False):
    """
    Computes the Precision@k (P@k) for items.
    In short, this is the proportion of relevant items in the retrieved items.

    :param logits: prediction matrix about item relevance
    :param targets: 0/1 matrix encoding true item relevance, same shape as logits
    :param k: top k items to consider
    :param logits_are_top_indices: whether logits are already top-k sorted indices
    """
    if k <= 0:
        raise ValueError("k is required to be positive!")

    top_indices = _get_top_k(logits, k, logits_are_top_indices, sorted=False)
    n_relevant_items = torch.gather(targets, dim=-1, index=top_indices).sum(dim=-1)
    return n_relevant_items / k


def recall(logits: torch.Tensor, targets: torch.Tensor, k: int = 10, logits_are_top_indices: bool = False):
    """
    Computes the Recall@k (R@k) for items.
    In short, this is the proportion of relevant retrieved items of all relevant items.

    :param logits: prediction matrix about item relevance
    :param targets: 0/1 matrix encoding true item relevance, same shape as logits
    :param k: top k items to consider
    :param logits_are_top_indices: whether logits are already top-k sorted indices
    """
    top_indices = _get_top_k(logits, k, logits_are_top_indices, sorted=False)
    n_relevant_items = torch.gather(targets, dim=-1, index=top_indices).sum(dim=-1)
    n_total_relevant = targets.sum(dim=-1)

    # may happen that there are no relevant true items, cover this possible DivisionByZero case.
    mask = n_total_relevant != 0
    recall = torch.zeros_like(n_relevant_items, dtype=torch.float, device=logits.device)
    recall[mask] = n_relevant_items[mask] / n_total_relevant[mask]

    return recall


def f_score(logits: torch.Tensor, targets: torch.Tensor, k: int = 10, logits_are_top_indices: bool = False):
    """
    Computes the F-score@k (F@k) for items.
    In short, this is the harmonic mean of precision@k and recall@k.

    :param logits: prediction matrix about item relevance
    :param targets: 0/1 matrix encoding true item relevance, same shape as logits
    :param k: top k items to consider
    :param logits_are_top_indices: whether logits are already top-k sorted indices
    """

    p = precision(logits, targets, k, logits_are_top_indices)
    r = recall(logits, targets, k, logits_are_top_indices)

    pr = p + r
    mask = pr != 0
    f_score = torch.zeros_like(r, dtype=torch.float, device=logits.device)
    f_score[mask] = 2 * ((p * r)[mask] / pr[mask])
    return f_score


def hitrate(logits: torch.Tensor, targets: torch.Tensor, k: int = 10, logits_are_top_indices: bool = False):
    """
    Computes the Hitrate@k (HR@k) for items.
    In short, this is the proportion of relevant that could be recommended.

    :param logits: prediction matrix about item relevance
    :param targets: 0/1 matrix encoding true item relevance, same shape as logits
    :param k: top k items to consider
    :param logits_are_top_indices: whether logits are already top-k sorted indices
    """
    top_indices = _get_top_k(logits, k, logits_are_top_indices, sorted=False)
    n_relevant_items = torch.gather(targets, dim=-1, index=top_indices).sum(dim=-1)
    n_total_relevant = targets.sum(dim=-1)

    # basically a pairwise min(count_relevant_items, k)
    denominator = torch.where(n_total_relevant > k, k, n_total_relevant)

    # may happen that there are no relevant true items, therefore we need to cover this possible DivisionByZero case.
    mask = denominator != 0
    recall = torch.zeros_like(denominator, dtype=torch.float, device=logits.device)
    recall[mask] = n_relevant_items[mask] / denominator[mask]

    return recall


def coverage(logits: torch.Tensor, k: int = 10):
    """
    Computes the Coverage@k (Cov@k) for items.
    In short, this is the proportion of all items that are recommended to the users.

    :param logits: prediction matrix about item relevance
    :param k: top k items to consider
    """
    top_indices = _get_top_k(logits, k, logits_are_top_indices=False, sorted=False)
    n_items = logits.shape[-1]
    return coverage_from_top_k(top_indices, n_items)


def coverage_from_top_k(top_indices, n_items):
    n_unique_recommended_items = top_indices.unique(sorted=False).shape[0]
    return n_unique_recommended_items / n_items


_metric_fn_map_user = {
    MetricEnum.DCG: dcg,
    MetricEnum.NDCG: ndcg,
    MetricEnum.Recall: recall,
    MetricEnum.Precision: precision,
    MetricEnum.Hitrate: hitrate,
    MetricEnum.F_Score: f_score
}

_metric_fn_map_distribution = {
    MetricEnum.Coverage: coverage
}

# List of metrics that are currently supported
supported_metrics = tuple(MetricEnum)
supported_user_metrics = tuple(_metric_fn_map_user.keys())
supported_distribution_metrics = tuple(_metric_fn_map_distribution.keys())


def _calculate(metrics: Iterable[str | MetricEnum], logits: torch.Tensor, targets: torch.Tensor = None, k: int = 10,
               best_logit_indices: torch.Tensor = None, return_aggregated: bool = True, return_individual: bool = False,
               ):
    """
    Computes the values for a given list of metrics.

    :param metrics: The list of metrics to compute. Check out 'supported_metrics' for a list of names.
    :param logits: prediction matrix about item relevance
    :param targets: 0/1 matrix encoding true item relevance, same shape as logits
    :param k: top k items to consider
    :param best_logit_indices: Sorted indices of best logits, this can be used to speed up computations
    :param return_aggregated: Whether aggregated metric results should be returned.
    :param return_individual: Whether the results for individual users should be returned
    :return: a dictionary containing ...
        {metric_name: value} if 'return_aggregated=True', and/or
        {<metric_name>_individual: list_of_values} if return_individual=True'
    """
    if logits.shape[-1] < k:
        raise ValueError(f"'k' must not be greater than the number of items ({k} > {logits.shape[-1]})!")

    if not (return_individual or return_aggregated):
        raise ValueError(f"Specify either 'return_individual' or 'return_aggregated' to receive results.")

    raw_results = {}
    for metric in metrics:
        if metric in _metric_fn_map_distribution:
            raw_results[str(metric)] = _metric_fn_map_distribution[metric](logits, k)

        elif metric in _metric_fn_map_user:
            if targets is None:
                raise ValueError(f"'targets' is required to calculate '{metric}'!")
            # use pre-computed best logit indices to speed up computations
            if best_logit_indices is not None:
                raw_results[str(metric)] = _metric_fn_map_user[metric](best_logit_indices, targets, k,
                                                                       logits_are_top_indices=True)
            else:
                raw_results[str(metric)] = _metric_fn_map_user[metric](logits, targets, k,
                                                                       logits_are_top_indices=False)

        else:
            raise ValueError(f"Metric '{metric}' not supported.")

    results = {}
    if return_aggregated:
        results.update({k: torch.mean(v).item() if isinstance(v, torch.Tensor) else v
                        for k, v in raw_results.items()})

    if return_individual:
        results.update({k + "_individual": v for k, v in raw_results.items()})

    return results


def calculate(metrics: Iterable[str | MetricEnum], logits: torch.Tensor, targets: torch.Tensor = None,
              k: int | Iterable[int] = 10, return_aggregated: bool = True, return_individual: bool = False,
              flatten_results: bool = False, flattened_parts_separator: str = "/", flattened_results_prefix: str = ""):
    """
    Computes the values for a given list of metrics.

    :param metrics: The list of metrics to compute. Check out 'supported_metrics' for a list of names.
    :param logits: prediction matrix about item relevance
    :param targets: 0/1 matrix encoding true item relevance, same shape as logits
    :param k: top k items to consider
    :param return_aggregated: Whether aggregated metric results should be returned. 
    :param return_individual: Whether the results for individual users should be returned
    :param flatten_results: Whether to flatten the results' dictionary.
                            Key is of format "{prefix}/{metric}@{k}" for separator "/"
    :param flattened_parts_separator: How to separate the individual parts of the flattened key
    :param flattened_results_prefix: Prefix to prepend to the flattened results key.
    :return: a dictionary containing ...
        {metric_name: value} if 'return_aggregated=True', and/or
        {<metric_name>_individual: list_of_values} if 'return_individual=True'
    """
    if logits.shape != targets.shape:
        raise ValueError(f"Logits and targets must be of same shape ({logits.shape} != {targets.shape})")

    full_prefix = f"{flattened_results_prefix}{flattened_parts_separator}" if flattened_results_prefix else ""

    k = (k,) if isinstance(k, int) else k
    best_logit_indices = logits.topk(max(k), dim=-1, sorted=True).indices

    metric_results = dict() if flatten_results else defaultdict(lambda: dict())
    for ki in k:
        for metric, v in _calculate(metrics, logits, targets, ki, best_logit_indices=best_logit_indices,
                                    return_aggregated=return_aggregated, return_individual=return_individual).items():
            if flatten_results:
                metric_results[f"{full_prefix}{metric}@{ki}"] = v
            else:
                metric_results[metric][ki] = v

    return dict(metric_results)
