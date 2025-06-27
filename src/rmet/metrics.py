import torch
import numpy as np
from enum import Enum
import scipy.sparse as sp
from typing import Iterable
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


def _assert_supported_type(a: any):
    if not isinstance(a, (np.ndarray, torch.Tensor, sp.csr_array)):
        raise TypeError(f"Type {type(a)} of input not supported.")


def _zeros_like_float(a: torch.Tensor | np.ndarray):
    if isinstance(a, torch.Tensor):
        return torch.zeros_like(a, dtype=torch.float, device=a.device)

    elif isinstance(a, np.ndarray):
        return np.zeros_like(a, dtype=float)

    else:
        _assert_supported_type(a)


def _as_float(a: torch.Tensor | np.ndarray):
    if isinstance(a, torch.Tensor):
        return a.float()

    elif isinstance(a, np.ndarray):
        return a.astype(float)

    else:
        _assert_supported_type(a)


def _get_top_k_numpy(a: np.ndarray, k: int, sorted: bool = True):
    # use partition which is much faster than argsort() on big arrays
    indices_unsorted = np.argpartition(a, -k, axis=-1)[:, -k:]
    if not sorted:
        return indices_unsorted

    # sort indices by their values
    values_unsorted = np.take_along_axis(a, indices_unsorted, axis=-1)
    sorting = np.argsort(values_unsorted, axis=-1)
    indices_sorted = np.take_along_axis(indices_unsorted, sorting, axis=-1)

    # reverse order from high to low
    return indices_sorted[:, ::-1]


def _get_top_k_sparse(a: sp.csr_array, k: int):
    n_rows, n_cols = a.shape

    if k > n_cols:
        raise ValueError("k must be at most number of columns")

    # create placeholder to fill
    top_k_indices = np.zeros(shape=(n_rows, k), dtype=int)

    # go row by row over array
    for i in range(n_rows):
        row = a[i, :]
        data = row.data

        if isinstance(row, sp.csr_array):
            ind = row.indices
        elif isinstance(row, sp.coo_array):
            ind = row.coords[0]
        else:
            raise TypeError(f"Type {type(row)} for rows in array not expected.")

        # we stay aligned with behaviour of np.argsort, which orders
        # duplicate values by their indices of occurence (first come first)
        # note that we need only to consider at most k indices as padding candidates,
        candidate_padding_indices = np.arange(n_cols - k, n_cols)

        # select those candidates that don't already occur
        mask = np.isin(candidate_padding_indices, ind, invert=True)
        padding_indices = candidate_padding_indices[mask]
        full_indices = np.concatenate([padding_indices, ind])

        # we consider all padding indices to have data=0, which is the default behaviour
        # for scipy. This way, negative data values are lower ranked than the padded values
        padding_data = np.zeros_like(padding_indices, dtype=a.dtype)

        # sorting is always required as there might be negative values in original data
        full_data = np.concatenate([padding_data, data])
        sorting = np.argsort(full_data)
        full_indices = full_indices[sorting]

        # select final indices
        top_k_row = full_indices[-k:][::-1]
        top_k_indices[i] = top_k_row

    return top_k_indices


def _get_top_k(
    logits: torch.Tensor | np.ndarray | sp.csr_array,
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
    if logits_are_top_indices:
        return logits[:, :k]

    else:
        if isinstance(logits, torch.Tensor):
            return logits.topk(k, dim=-1, sorted=sorted).indices

        elif isinstance(logits, np.ndarray):
            return _get_top_k_numpy(logits, k, sorted=sorted)

        elif isinstance(logits, sp.csr_array):
            return _get_top_k_sparse(logits, k)

        else:
            _assert_supported_type(logits)


def _get_relevancy_scores(
    targets: torch.Tensor | np.ndarray | sp.csr_array,
    indices: torch.Tensor | np.ndarray,
):
    if isinstance(targets, torch.Tensor):
        return torch.gather(targets, dim=-1, index=indices)

    elif isinstance(targets, (np.ndarray, sp.csr_array)):
        scores = np.take_along_axis(targets, indices, axis=-1)
        # if there are zeros somewhere in the selected range,
        # take_along_axis returns a sparse array instead of a numpy array
        if isinstance(scores, sp.csr_array):
            return scores.todense()
        return scores
    else:
        _assert_supported_type(targets)


def _get_top_k_relevancies(
    logits: torch.Tensor | np.ndarray | sp.csr_array,
    targets: torch.Tensor | np.ndarray | sp.csr_array,
    k=10,
    logits_are_top_indices: bool = False,
    sorted: bool = True,
):
    top_indices = _get_top_k(logits, k, logits_are_top_indices, sorted=sorted)
    return _get_relevancy_scores(targets, top_indices)


def _get_n_top_k_relevant(
    logits: torch.Tensor | np.ndarray | sp.csr_array,
    targets: torch.Tensor | np.ndarray | sp.csr_array,
    k=10,
    logits_are_top_indices: bool = False,
    sorted: bool = True,
):
    relevancy_scores = _get_top_k_relevancies(
        logits, targets, k, logits_are_top_indices, sorted=sorted
    )
    return relevancy_scores.sum(-1)


def dcg(
    logits: torch.Tensor | np.ndarray | sp.csr_array,
    targets: torch.Tensor | np.ndarray | sp.csr_array,
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

    if isinstance(relevancy_scores, torch.Tensor):
        discount = 1 / torch.log2(torch.arange(1, k + 1) + 1)
        discount = discount.to(device=logits.device)

    elif isinstance(relevancy_scores, np.ndarray):
        discount = 1 / np.log2(np.arange(1, k + 1) + 1)

    else:
        _assert_supported_type(relevancy_scores)

    return _as_float(relevancy_scores) @ discount


def ndcg(
    logits: torch.Tensor | np.ndarray | sp.csr_array,
    targets: torch.Tensor | np.ndarray | sp.csr_array,
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
    ndcg = dcg(logits, targets, k, logits_are_top_indices) / normalization

    return ndcg


def precision(
    logits: torch.Tensor | np.ndarray | sp.csr_array,
    targets: torch.Tensor | np.ndarray | sp.csr_array,
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
    logits: torch.Tensor | np.ndarray | sp.csr_array,
    targets: torch.Tensor | np.ndarray | sp.csr_array,
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
    n_total_relevant = targets.sum(-1)

    # may happen that there are no relevant true items, cover this possible DivisionByZero case.
    mask = n_total_relevant != 0
    recall = _zeros_like_float(n_relevant_items)
    recall[mask] = n_relevant_items[mask] / n_total_relevant[mask]
    return recall


def f_score(
    logits: torch.Tensor | np.ndarray | sp.csr_array,
    targets: torch.Tensor | np.ndarray | sp.csr_array,
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
    f_score = _zeros_like_float(r)
    f_score[mask] = 2 * ((p * r)[mask] / pr[mask])
    return f_score


def hitrate(
    logits: torch.Tensor | np.ndarray | sp.csr_array,
    targets: torch.Tensor | np.ndarray | sp.csr_array,
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
    return _as_float(n_relevant_items.clip(max=1))


def average_precision(
    logits: torch.Tensor | np.ndarray | sp.csr_array,
    targets: torch.Tensor | np.ndarray | sp.csr_array,
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

    top_indices = _get_top_k(logits, k, logits_are_top_indices, sorted=True)
    n_total_relevant = targets.sum(-1)

    total_precision = _zeros_like_float(n_total_relevant)
    for ki in range(1, k + 1):
        # relevance of k'th indices (for -1 see offset in range)
        position_relevance = _get_relevancy_scores(
            targets, top_indices[:, ki - 1 : ki]
        )[:, 0]
        position_precision = precision(
            top_indices, targets, ki, logits_are_top_indices=True
        )
        total_precision += position_precision * position_relevance

    # may happen that there are no relevant true items, cover this possible DivisionByZero case.
    mask = n_total_relevant != 0

    avg_precision = _zeros_like_float(n_total_relevant)
    avg_precision[mask] = total_precision[mask] / n_total_relevant[mask]

    return avg_precision


def reciprocal_rank(
    logits: torch.Tensor | np.ndarray | sp.csr_array,
    targets: torch.Tensor | np.ndarray | sp.csr_array,
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
    if isinstance(relevancy_scores, torch.Tensor):
        max_result = torch.max(relevancy_scores, -1)
        max_indices = max_result.indices
        max_values = max_result.values

    elif isinstance(relevancy_scores, np.ndarray | sp.csr_array):
        max_indices = np.argmax(relevancy_scores, -1, keepdims=True)
        max_values = np.take_along_axis(
            relevancy_scores, max_indices, axis=-1
        ).flatten()
        max_indices = max_indices.flatten()
    else:
        _assert_supported_type(relevancy_scores)

    # mask to indicate which 'hits' are actually true
    # (if there are no hits at all for some items)
    mask = max_values > 0

    # by default, assume reciprocal rank of 0 for all users,
    # which is the case if there is no match in the recommendations,
    # i.e., if lim k->inf, 1/k->0
    rr = _zeros_like_float(max_values)

    denominator = max_indices[mask] + 1
    if isinstance(denominator, torch.Tensor):
        # pytorch is more strict with matching types, so we'll handle it specially
        denominator = denominator.type(rr.dtype)

    # +1 because indices are zero-based, while k is one-based
    rr[mask] = 1.0 / denominator
    return rr


def coverage(logits: torch.Tensor | np.ndarray | sp.csr_array, k: int = 10):
    """
    Computes the Coverage@k (Cov@k) for items.
    In short, this is the proportion of all items that are recommended to the users.

    :param logits: prediction matrix about item relevance
    :param k: top k items to consider
    """
    top_indices = _get_top_k(logits, k, logits_are_top_indices=False, sorted=False)
    n_items = logits.shape[-1]
    return coverage_from_top_k(top_indices, k, n_items)


def coverage_from_top_k(
    top_indices: torch.Tensor | np.ndarray | sp.csr_array,
    k: int,
    n_items: int,
):
    unique_values = _get_unique_values(top_indices, k)

    n_unique_recommended_items = unique_values.shape[0]
    return n_unique_recommended_items / n_items


def _get_unique_values(top_indices, k):
    if isinstance(top_indices, torch.Tensor):
        unique_values = top_indices[:, :k].unique(sorted=False)

    elif isinstance(top_indices, np.ndarray):
        unique_values = np.unique(top_indices)

    else:
        _assert_supported_type(top_indices)
    return unique_values


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
    logits: torch.Tensor | np.ndarray | sp.csr_array = None,
    targets: torch.Tensor | np.ndarray | sp.csr_array = None,
    k: int | Iterable[int] = 10,
    return_aggregated: bool = True,
    return_individual: bool = False,
    calculate_std: bool = False,
    flatten_results: bool = False,
    flattened_parts_separator: str = "/",
    flattened_results_prefix: str = "",
    n_items: int = None,
    best_logit_indices: torch.Tensor | np.ndarray = None,
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
            for k, v in raw_results.items():
                if not isinstance(v, float):
                    results[k + "_individual"] = v
                else:
                    # no individual values for global metrics available
                    results[k] = v

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
    best_logit_indices: torch.Tensor | np.ndarray,
    n_items: int,
    targets: torch.Tensor | np.ndarray | sp.csr_array = None,
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
            n_targets = targets.sum(-1)

            if isinstance(n_targets, torch.Tensor):
                mask = torch.argwhere(n_targets).flatten()
                metric_result = torch.zeros(targets.shape[0], device=targets.device)

            elif isinstance(n_targets, np.ndarray | sp.csr_array):
                mask = np.argwhere(n_targets).flatten()
                metric_result = np.zeros(targets.shape[0])

            else:
                _assert_supported_type(n_targets)

            metric_result[mask] = _metric_fn_map_user[metric](
                best_logit_indices[mask], targets[mask], k, logits_are_top_indices=True
            )
            raw_results[str(metric)] = metric_result

        else:
            raise ValueError(f"Metric '{metric}' not supported.")
    return raw_results


def _aggregate_results(
    raw_results: dict[str, torch.Tensor | np.ndarray], calculate_std: bool = False
):
    results = {}
    for k, v in raw_results.items():
        if isinstance(v, torch.Tensor):
            results[k] = torch.mean(v).item()
            if calculate_std:
                results[f"{k}_std"] = torch.std(v).item()

        elif isinstance(v, np.ndarray):
            results[k] = np.mean(v).item()
            if calculate_std:
                # set degrees of freedom to 1 to have same results as torch
                results[f"{k}_std"] = np.std(v, ddof=1).item()

        else:
            # nothing to do here, as global metrics are already registered as results before
            pass

    return results
