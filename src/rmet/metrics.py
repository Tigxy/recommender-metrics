import torch
from typing import Iterable
from enum import Enum
from collections import defaultdict


class MetricEnum(str, Enum):
    DCG = "dcg"
    NDCG = "ndcg"
    Precision = "precision"
    Recall = "recall"
    F_Score = "f_score"
    Hitrate = "hitrate"
    Coverage = "coverage"
    AP = "ap"
    RR = "rr"

    def __str__(self):
        return self.value


def _get_top_k(
    logits: torch.Tensor,
    k=10,
    logits_are_top_indices: bool = False,
    sorted: bool = True,
):
    """
    Gets the top-k indices for the logits

    :param logits: prediction matrix about item relevance
    :param k: top k items to consider
    :param logits_are_top_indices: whether logits are already top-k sorted indices
    :param sorted: whether indices should be returned in sorted order
    """
    return (
        logits[:, :k]
        if logits_are_top_indices
        else logits.topk(k, dim=-1, sorted=sorted).indices
    )


def _get_relevancy_scores(targets: torch.Tensor, indices: torch.Tensor):
    return torch.gather(targets, dim=-1, index=indices)


def _get_top_k_relevancies(
    logits: torch.Tensor,
    targets: torch.Tensor,
    k=10,
    logits_are_top_indices: bool = False,
    sorted: bool = True,
):
    top_indices = _get_top_k(logits, k, logits_are_top_indices, sorted=sorted)
    return _get_relevancy_scores(targets, top_indices)


def _get_n_top_k_relevant(
    logits: torch.Tensor,
    targets: torch.Tensor,
    k=10,
    logits_are_top_indices: bool = False,
    sorted: bool = True,
):
    relevancy_scores = _get_top_k_relevancies(
        logits, targets, k, logits_are_top_indices, sorted=sorted
    )
    return relevancy_scores.sum(-1)


def dcg(
    logits: torch.Tensor,
    targets: torch.Tensor,
    k=10,
    logits_are_top_indices: bool = False,
):
    """
    Computes the Discounted Cumulative Gain (DCG) for items.

    :param logits: prediction matrix about item relevance
    :param targets: 0/1 matrix encoding true item relevance, same shape as logits
    :param k: top k items to consider
    :param logits_are_top_indices: whether logits are already top-k sorted indices
    """
    relevancy_scores = _get_top_k_relevancies(
        logits, targets, k, logits_are_top_indices, sorted=True
    )
    discount = 1 / torch.log2(torch.arange(1, k + 1) + 1)
    discount = discount.to(device=logits.device)
    return relevancy_scores.float() @ discount


def ndcg(
    logits: torch.Tensor,
    targets: torch.Tensor,
    k: int = 10,
    logits_are_top_indices: bool = False,
):
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


def precision(
    logits: torch.Tensor,
    targets: torch.Tensor,
    k: int = 10,
    logits_are_top_indices: bool = False,
):
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

    n_relevant_items = _get_n_top_k_relevant(
        logits, targets, k, logits_are_top_indices, sorted=False
    )
    return n_relevant_items / k


def recall(
    logits: torch.Tensor,
    targets: torch.Tensor,
    k: int = 10,
    logits_are_top_indices: bool = False,
):
    """
    Computes the Recall@k (R@k) for items.
    In short, this is the proportion of relevant retrieved items of all relevant items.

    :param logits: prediction matrix about item relevance
    :param targets: 0/1 matrix encoding true item relevance, same shape as logits
    :param k: top k items to consider
    :param logits_are_top_indices: whether logits are already top-k sorted indices
    """
    n_relevant_items = _get_n_top_k_relevant(
        logits, targets, k, logits_are_top_indices, sorted=False
    )
    n_total_relevant = targets.sum(dim=-1)

    # may happen that there are no relevant true items, cover this possible DivisionByZero case.
    mask = n_total_relevant != 0
    recall = torch.zeros_like(n_relevant_items, dtype=torch.float, device=logits.device)
    recall[mask] = n_relevant_items[mask] / n_total_relevant[mask]

    return recall


def f_score(
    logits: torch.Tensor,
    targets: torch.Tensor,
    k: int = 10,
    logits_are_top_indices: bool = False,
):
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


def hitrate(
    logits: torch.Tensor,
    targets: torch.Tensor,
    k: int = 10,
    logits_are_top_indices: bool = False,
):
    """
    Computes the Hitrate@k (HR@k) for items.
    In short, this is a simple 0/1 metric that considers whether any of the recommended
    items are actually relevant.

    :param logits: prediction matrix about item relevance
    :param targets: 0/1 matrix encoding true item relevance, same shape as logits
    :param k: top k items to consider
    :param logits_are_top_indices: whether logits are already top-k sorted indices
    """
    n_relevant_items = _get_n_top_k_relevant(
        logits, targets, k, logits_are_top_indices, sorted=False
    )
    return n_relevant_items.clip(max=1).float()


def average_precision(
    logits: torch.Tensor,
    targets: torch.Tensor,
    k: int = 10,
    logits_are_top_indices: bool = False,
):
    """
    Computes the mean_average_precision@k (MAP@k) for items.
    In short, it combines precision values at all possible recall levels.

    :param logits: prediction matrix about item relevance
    :param targets: 0/1 matrix encoding true item relevance, same shape as logits
    :param k: top k items to consider
    :param logits_are_top_indices: whether logits are already top-k sorted indices

    :returns: average precision for each sample of the input
    """
    if k <= 0:
        raise ValueError("k is required to be positive!")

    top_indices = _get_top_k(
        logits, k, logits_are_top_indices, sorted=True
    )  # (n_samples, k)
    n_total_relevant = targets.sum(dim=-1)  # (n_samples,)

    total_precision = torch.zeros_like(
        n_total_relevant, dtype=torch.float, device=logits.device
    )  # (n_samples,)
    for ki in range(1, k + 1):  # {1, ..., k}
        # relevance of k'th indices (for -1 see offset in range)
        position_relevance = _get_relevancy_scores(
            targets, top_indices[:, ki - 1 : ki]
        )[
            :, 0
        ]  # (n_samples,)
        position_precision = precision(
            top_indices, targets, ki, logits_are_top_indices=True
        )  # (n_samples,)
        total_precision += position_precision * position_relevance

    # may happen that there are no relevant true items, cover this possible DivisionByZero case.
    mask = n_total_relevant != 0
    avg_precision = torch.zeros_like(
        n_total_relevant, dtype=torch.float, device=logits.device
    )
    avg_precision[mask] = total_precision[mask] / n_total_relevant[mask]

    return avg_precision


def reciprocal_rank(
    logits: torch.Tensor,
    targets: torch.Tensor,
    k: int = 10,
    logits_are_top_indices: bool = False,
):
    """
    Computes the reciprocal rank@k (RR@k) for items.
    In short, it is the inverse rank of the first item that is relevant to the user.
    High values indicate that early items in the recommendations are of interest to the user.
    If there is no relevant item in the top-k recommendations, the reciprocal rank is 0

    :param logits: prediction matrix about item relevance
    :param targets: 0/1 matrix encoding true item relevance, same shape as logits
    :param k: top k items to consider
    :param logits_are_top_indices: whether logits are already top-k sorted indices

    :returns: reciprocal rank for each sample of the input
    """
    if k <= 0:
        raise ValueError("k is required to be positive!")

    relevancy_scores = _get_top_k_relevancies(
        logits, targets, k, logits_are_top_indices, sorted=True
    )

    # earliest 'hits' in the recommendation list
    # about determinism, from https://pytorch.org/docs/stable/generated/torch.max.html#torch.max:
    # >>> If there are multiple maximal values in a reduced row
    # >>> then the indices of the first maximal value are returned.
    hits = torch.max(relevancy_scores, dim=-1)

    # mask to indicate which 'hits' are actually true
    # (if there are no hits at all for some items)
    mask = hits.values > 0

    # by default, assume reciprocal rank of 0 for all users,
    # which is the case if there is no match in the recommendations,
    # i.e., if lim k->inf, 1/k->0
    rr = torch.zeros_like(hits.values, dtype=torch.float)
    rr[mask] = 1.0 / (hits.indices[mask] + 1).type(
        rr.dtype
    )  # +1 because indices are zero-based, while k is one-based

    return rr


def coverage(logits: torch.Tensor, k: int = 10):
    """
    Computes the Coverage@k (Cov@k) for items.
    In short, this is the proportion of all items that are recommended to the users.

    :param logits: prediction matrix about item relevance
    :param k: top k items to consider
    """
    top_indices = _get_top_k(logits, k, logits_are_top_indices=False, sorted=False)
    n_items = logits.shape[-1]
    return coverage_from_top_k(top_indices, k, n_items)


def coverage_from_top_k(top_indices: torch.Tensor, k: int, n_items: int):
    n_unique_recommended_items = top_indices[:, :k].unique(sorted=False).shape[0]
    return n_unique_recommended_items / n_items


_metric_fn_map_user = {
    MetricEnum.DCG: dcg,
    MetricEnum.NDCG: ndcg,
    MetricEnum.Recall: recall,
    MetricEnum.Precision: precision,
    MetricEnum.Hitrate: hitrate,
    MetricEnum.F_Score: f_score,
    MetricEnum.AP: average_precision,
    MetricEnum.RR: reciprocal_rank,
}

_metric_fn_map_distribution = {MetricEnum.Coverage: coverage_from_top_k}

# List of metrics that are currently supported
supported_metrics = tuple(MetricEnum)
supported_user_metrics = tuple(_metric_fn_map_user.keys())
supported_distribution_metrics = tuple(_metric_fn_map_distribution.keys())


def calculate(
    metrics: Iterable[str | MetricEnum],
    logits: torch.Tensor = None,
    targets: torch.Tensor = None,
    k: int | Iterable[int] = 10,
    return_aggregated: bool = True,
    return_individual: bool = False,
    calculate_std: bool = False,
    flatten_results: bool = False,
    flattened_parts_separator: str = "/",
    flattened_results_prefix: str = "",
    n_items: int = None,
    best_logit_indices: torch.Tensor = None,
    return_best_logit_indices: bool = False,
):
    """
    Computes the values for a given list of metrics.

    :param metrics: The list of metrics to compute. Check out 'supported_metrics' for a list of names.
    :param logits: prediction matrix about item relevance
    :param targets: 0/1 matrix encoding true item relevance, same shape as logits
    :param k: top k items to consider
    :param return_aggregated: Whether aggregated metric results should be returned.
    :param return_individual: Whether the results for individual users should be returned
    :param calculate_std: Whether to calculate the standard deviation for the aggregated results
    :param flatten_results: Whether to flatten the results' dictionary.
                            Key is of format "{prefix}/{metric}@{k}" for separator "/"
    :param flattened_parts_separator: How to separate the individual parts of the flattened key
    :param flattened_results_prefix: Prefix to prepend to the flattened results key.
    :param n_items: Number of items in dataset (in case only best logit indices are supplied)
    :param best_logit_indices: Previously computed indices of the best logits in sorted order
    :param return_best_logit_indices: Whether to return the indices of the best logits
    :return: a dictionary containing ...
        {metric_name: value} if 'return_aggregated=True', and/or
        {<metric_name>_individual: list_of_values} if 'return_individual=True'
    """

    k = (k,) if isinstance(k, int) else k
    max_k = max(k)

    # ensure validity of supplied parameters
    if logits is not None and logits.shape[-1] < max_k:
        raise ValueError(
            f"'k' must not be greater than the number of logits "
            f"({max_k} > {logits.shape[-1]})!"
        )

    if best_logit_indices is not None and best_logit_indices.shape[-1] < max_k:
        raise ValueError(
            f"'k' must not be greater than the number of best indices "
            f"({max_k} > {best_logit_indices.shape[-1]})!"
        )

    if logits is None and (best_logit_indices is None or n_items is None):
        raise ValueError("Either logits or best_logit_indices+n_items must be supplied")

    if best_logit_indices is None and logits.shape != targets.shape:
        raise ValueError(
            f"Logits and targets must be of same shape ({logits.shape} != {targets.shape})"
        )

    if not (return_individual or return_aggregated):
        raise ValueError(
            f"Specify either 'return_individual' or 'return_aggregated' to receive results."
        )

    n_items = n_items or logits.shape[-1]
    # to speed up computations, only retrieve highest logit indices once (if not already supplied)
    if best_logit_indices is None:
        best_logit_indices = _get_top_k(
            logits, max_k, logits_are_top_indices=False, sorted=True
        )

    full_prefix = (
        f"{flattened_results_prefix}{flattened_parts_separator}"
        if flattened_results_prefix
        else ""
    )

    metric_results = dict() if flatten_results else defaultdict(lambda: dict())
    # iterate over all k's and compute the metrics for them
    for ki in k:
        raw_results = _compute_raw_results(
            metrics, ki, best_logit_indices, n_items, targets
        )

        results = {}
        if return_individual:
            individual_results = {
                k + "_individual": v
                for k, v in raw_results.items()
                if isinstance(v, torch.Tensor)
            }
            results.update(individual_results)

        if return_aggregated:
            aggregated_results = _aggregate_results(raw_results, calculate_std)
            results.update(aggregated_results)

        for metric, v in results.items():
            if flatten_results:
                metric_results[f"{full_prefix}{metric}@{ki}"] = v
            else:
                metric_results[metric][ki] = v

    if return_best_logit_indices:
        return dict(metric_results), best_logit_indices

    return dict(metric_results)


def _compute_raw_results(
    metrics: Iterable[str | MetricEnum],
    k: int,
    best_logit_indices: torch.Tensor,
    n_items: int,
    targets: torch.Tensor = None,
):
    raw_results = {}
    for metric in metrics:
        if metric in _metric_fn_map_distribution:
            raw_results[str(metric)] = _metric_fn_map_distribution[metric](
                best_logit_indices, k, n_items
            )

        elif metric in _metric_fn_map_user:
            if targets is None:
                raise ValueError(f"'targets' is required to calculate '{metric}'!")

            # do not compute metrics for users where we do not have any
            # underlying ground truth interactions
            mask = torch.argwhere(targets.sum(-1)).flatten(0)
            metric_result = torch.zeros(targets.shape[0], device=targets.device)
            metric_result[mask] = _metric_fn_map_user[metric](
                best_logit_indices[mask], targets[mask], k, logits_are_top_indices=True
            )
            raw_results[str(metric)] = metric_result

        else:
            raise ValueError(f"Metric '{metric}' not supported.")
    return raw_results

def _aggregate_results(raw_results, calculate_std: bool = False):
    results = {
        k: torch.mean(v).item() if isinstance(v, torch.Tensor) else v
        for k, v in raw_results.items()
    }
    if calculate_std:
        results.update(
            {
                f"{k}_std": torch.std(v).item()
                for k, v in raw_results.items()
                if isinstance(v, torch.Tensor)
            }
        )
    return results
