from collections import defaultdict
from typing import Union
import torch


def dcg(logits: torch.Tensor, targets: torch.Tensor, k=10):
    """
    Computes the Discounted Cumulative Gain (DCG) for items.

    :param logits: prediction matrix about item relevance
    :param targets: 0/1 matrix encoding true item relevance, same shape as logits
    :param k: top k items to consider
    """
    top_indices = logits.topk(k, dim=-1).indices
    discount = 1 / torch.log2(torch.arange(1, k + 1) + 1)
    discount = discount.to(device=logits.device)
    relevancy_scores = torch.gather(targets, dim=-1, index=top_indices)
    return relevancy_scores.float() @ discount


def ndcg(logits: torch.Tensor, targets: torch.Tensor, k=10):
    """
    Computes the Normalized Discounted Cumulative Gain (nDCG) for items.

    :param logits: prediction matrix about item relevance
    :param targets: 0/1 matrix encoding true item relevance, same shape as logits
    :param k: top k items to consider
    """
    if k <= 0:
        raise ValueError("k is required to be positive!")

    normalization = dcg(torch.ones(k), torch.ones(k), k)
    normalization = normalization.to(device=logits.device)
    return dcg(logits, targets, k) / normalization


def precision(logits: torch.Tensor, targets: torch.Tensor, k=10):
    """
    Computes the Precision@k (P@k) for items.
    In short, this is the proportion of relevant items in the retrieved items.

    :param logits: prediction matrix about item relevance
    :param targets: 0/1 matrix encoding true item relevance, same shape as logits
    :param k: top k items to consider
    """
    if k <= 0:
        raise ValueError("k is required to be positive!")

    top_indices = logits.topk(k, dim=-1).indices
    n_relevant_items = torch.gather(targets, dim=-1, index=top_indices).sum(axis=-1)
    return n_relevant_items / k


def recall(logits: torch.Tensor, targets: torch.Tensor, k=10):
    """
    Computes the Recall@k (R@k) for items.
    In short, this is the proportion of relevant retrieved items of all relevant items.

    :param logits: prediction matrix about item relevance
    :param targets: 0/1 matrix encoding true item relevance, same shape as logits
    :param k: top k items to consider
    """
    top_indices = logits.topk(k, dim=-1).indices
    n_relevant_items = torch.gather(targets, dim=-1, index=top_indices).sum(axis=-1)
    n_total_relevant = targets.sum(axis=-1)

    # may happen that there are no relevant true items, cover this possible DivisionByZero case.
    mask = n_total_relevant != 0
    recall = torch.zeros_like(n_relevant_items, dtype=torch.float, device=logits.device)
    recall[mask] = n_relevant_items[mask] / n_total_relevant[mask]

    return recall


def f_score(logits: torch.Tensor, targets: torch.Tensor, k=10):
    """
    Computes the F-score@k (F@k) for items.
    In short, this is the harmonic mena of precision@k and recall@k.

    :param logits: prediction matrix about item relevance
    :param targets: 0/1 matrix encoding true item relevance, same shape as logits
    :param k: top k items to consider
    """

    p = precision(logits, targets, k)
    r = recall(logits, targets, k)

    pr = p + r
    mask = pr != 0
    f_score = torch.zeros_like(r, dtype=torch.float, device=logits.device)
    f_score[mask] = 2 * ((p * r)[mask] / pr[mask])
    return f_score


def hitrate(logits: torch.Tensor, targets: torch.Tensor, k=10):
    """
    Computes the Hitrate@k (HR@k) for items.
    In short, this is the proportion of relevant that could be recommended.

    :param logits: prediction matrix about item relevance
    :param targets: 0/1 matrix encoding true item relevance, same shape as logits
    :param k: top k items to consider
    """
    top_indices = logits.topk(k, dim=-1).indices
    n_relevant_items = torch.gather(targets, dim=-1, index=top_indices).sum(axis=-1)
    n_total_relevant = targets.sum(axis=-1)

    # basically a pairwise min(count_relevant_items, k)
    denominator = torch.where(n_total_relevant > k, k, n_total_relevant)

    # may happen that there are no relevant true items, therefore we need to cover this possible DivisionByZero case.
    mask = denominator != 0
    recall = torch.zeros_like(denominator, dtype=torch.float, device=logits.device)
    recall[mask] = n_relevant_items[mask] / denominator[mask]

    return recall


def coverage(logits: torch.Tensor, k=10):
    """
    Computes the Coverage@k (Cov@k) for items.
    In short, this is the proportion of all items that are recommended to the users.

    :param logits: prediction matrix about item relevance
    :param k: top k items to consider
    """
    top_indices = logits.topk(k, dim=-1).indices
    n_unique_recommended_items = top_indices.unique(sorted=False).shape[0]
    n_items = logits.shape[-1]
    return n_unique_recommended_items / n_items


metric_fn_map_unary = {
    "coverage": coverage
}

metric_fn_map_bi = {
    "dcg": dcg,
    "ndcg": ndcg,
    "recall": recall,
    "precision": precision,
    "hitrate": hitrate,
    "f_score": f_score
}

# List of metrics that are currently supported
supported_metrics = list(metric_fn_map_unary.keys()) + list(metric_fn_map_bi.keys())


def _calculate(metrics: tuple, logits: torch.Tensor, targets=None, k=10, return_aggregated=True, return_individual=False):
    """
    Computes the values for a given list of metrics.

    :param metrics: The list of metrics to compute. Check out 'supported_metrics' for a list of names.
    :param logits: prediction matrix about item relevance
    :param targets: 0/1 matrix encoding true item relevance, same shape as logits
    :param k: top k items to consider
    :param return_aggregated: Whether aggregated metric results should be returned. 
    :param return_individual: Whether the results for individual users should be returned
    :return: a dictionary containing ...
        {metric_name: value} if 'return_aggregated=True', and/or
        {<metric_name>_individual: list_of_values} if return_individual=True'
    """
    if logits.shape[-1] < k:
        raise ValueError(f"'k' must not be greater than the number of items ({k} > {logits.shape[-1]})!")

    raw_results = {}
    for metric in metrics:
        if metric in metric_fn_map_unary:
            raw_results[metric] = metric_fn_map_unary[metric](logits, k)

        elif metric in metric_fn_map_bi:
            if targets is None:
                raise ValueError(f"'targets' is required to calculate '{metric}'!")
            raw_results[metric] = metric_fn_map_bi[metric](logits, targets, k)

        else:
            raise ValueError(f"Metric '{metric}' not supported.")

    results = {}

    if return_aggregated:
        results.update({k: torch.mean(v).item() if isinstance(v, torch.Tensor) else v 
                        for k, v in raw_results.items()})

    if return_individual:
        results.update({k + "_individual": v for k, v in raw_results.items()})

    return results


def calculate(metrics: tuple, logits: torch.Tensor, targets=None, k: Union[int, list]=10, return_aggregated=True, return_individual=False):
    """
    Computes the values for a given list of metrics.

    :param metrics: The list of metrics to compute. Check out 'supported_metrics' for a list of names.
    :param logits: prediction matrix about item relevance
    :param targets: 0/1 matrix encoding true item relevance, same shape as logits
    :param k: top k items to consider
    :param return_aggregated: Whether aggregated metric results should be returned. 
    :param return_individual: Whether the results for individual users should be returned
    :return: a dictionary containing ...
        {metric_name: value} if 'return_aggregated=True', and/or
        {<metric_name>_individual: list_of_values} if return_individual=True'
    """
    if isinstance(k, int):
        return _calculate(metrics, logits, targets, k, return_aggregated=return_aggregated, return_individual=return_individual)

    metric_results = defaultdict(lambda: dict())
    for tk in k:
        for metric, v in _calculate(metrics, logits, targets, tk, return_aggregated=return_aggregated, return_individual=return_individual).items():
            metric_results[metric][tk] = v
    return dict(metric_results)
