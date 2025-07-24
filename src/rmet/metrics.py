import torch
import numpy as np
from enum import Enum
import scipy.sparse as sp
from scipy.stats import rankdata
from typing import Iterable
from collections import defaultdict

from .type_helpers import (
    as_float,
    assert_supported_type,
    get_top_k,
    get_unique_values,
    generate_non_zero_mask,
    stack,
    std,
    zeros_float,
    zeros_like_float,
)


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
    AverageRank = "average_rank"

    def __str__(self):
        return self.value


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
        assert_supported_type(targets)


def _get_top_k_relevancies(
    logits: torch.Tensor | np.ndarray | sp.csr_array,
    targets: torch.Tensor | np.ndarray | sp.csr_array,
    k=10,
    logits_are_top_indices: bool = False,
    sorted: bool = True,
):
    top_indices = get_top_k(logits, k, logits_are_top_indices, sorted=sorted)
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
        assert_supported_type(relevancy_scores)

    return as_float(relevancy_scores) @ discount


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
    recall = zeros_like_float(n_relevant_items)
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
    f_score = zeros_like_float(r)
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
    return as_float(n_relevant_items.clip(max=1))


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

    top_indices = get_top_k(logits, k, logits_are_top_indices, sorted=True)
    n_total_relevant = targets.sum(-1)

    total_precision = zeros_like_float(n_total_relevant)
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

    avg_precision = zeros_like_float(n_total_relevant)
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
        assert_supported_type(relevancy_scores)

    # mask to indicate which 'hits' are actually true
    # (if there are no hits at all for some items)
    mask = max_values > 0

    # by default, assume reciprocal rank of 0 for all users,
    # which is the case if there is no match in the recommendations,
    # i.e., if lim k->inf, 1/k->0
    rr = zeros_like_float(max_values)

    denominator = max_indices[mask] + 1
    if isinstance(denominator, torch.Tensor):
        # pytorch is more strict with matching types, so we'll handle it specially
        denominator = denominator.type(rr.dtype)

    # +1 because indices are zero-based, while k is one-based
    rr[mask] = 1.0 / denominator
    return rr


def average_rank(
    logits: torch.Tensor | np.ndarray | sp.csr_array,
    targets: torch.Tensor | np.ndarray | sp.csr_array = None,
    item_ranks: torch.Tensor | np.ndarray | sp.csr_array = None,
    k=10,
    logits_are_top_indices: bool = False,
):
    """
    Computes the AverageRank@k (AR@k) for items, which, as the name says, is the
    average rank of the top k items. Ranks are either computed from the given targets,
    where increasing item ranks indicate decreasing popularity, or from the given
    item ranks.

    :param logits: prediction matrix about item relevance
    :param targets: 0/1 matrix encoding true item relevance, same shape as logits
    :param k: top k items to consider
    :param logits_are_top_indices: whether logits are already top-k sorted indices
    :param item_ranks: 1D item ranks to use for computing the average rank

    :returns: average rank for each sample of the input
    """
    if targets is None and item_ranks is None:
        raise ValueError("Either 'targets' or 'item_ranks' must be given.")

    if item_ranks is None:
        score_per_item = targets.sum(0)
        is_tensor_targets = isinstance(targets, torch.Tensor)

        if is_tensor_targets:
            # need to cast to and from torch tensors, as the rank computation
            # is not supported for them
            # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rankdata.html
            score_per_item = score_per_item.detach().cpu().numpy()

        # get ranks from 1 to n_items, where rank 1 has highest score
        # use "dense" to achieve the same rank for items with same values, but prevents
        # 'jumps' in ranks for the following items.
        # E.g., counts [4,4,3,5,1] should lead to [3,3,2,4,1]
        item_ranks = rankdata(score_per_item, method="dense")
        # flip ranks so that rank 1 is always most popular, rather than least popular
        # as this is better for understanding values across different numbers of items
        item_ranks = max(item_ranks) + 1 - item_ranks

        if is_tensor_targets:
            item_ranks = torch.tensor(item_ranks, device=targets.device)

    top_indices = get_top_k(logits, k, logits_are_top_indices, sorted=False)

    # gather item ranks and average them for each user individually (gather requires matching shapes)
    # this does drop the gradient, but shouldn't be relevant for evaluation metrics
    # another solution would be to gather on item_ranks.repeat((batch_size, 1)), which
    # would allocate more memory
    individual_results = [
        as_float(_get_relevancy_scores(item_ranks, ti)).mean(-1) for ti in top_indices
    ]
    individual_results = stack(individual_results)
    return individual_results


def coverage(
    logits: torch.Tensor | np.ndarray | sp.csr_array,
    k: int = 10,
    logits_are_top_indices: bool = False,
    n_items: int = None,
):
    """
    Computes the Coverage@k (Cov@k) for items.
    In short, this is the proportion of all items that are recommended to the users.

    :param logits: prediction matrix about item relevance
    :param k: top k items to consider
    :param logits_are_top_indices: whether logits are already top-k sorted indices
    :param n_items: if logits are top indices, n_items are requried for coverage computation
    """
    if logits_are_top_indices and n_items is None:
        raise ValueError("'n_items' required when using top indices as logits.")

    if not logits_are_top_indices:
        n_items = logits.shape[-1]

    top_indices = get_top_k(
        logits, k, logits_are_top_indices=logits_are_top_indices, sorted=False
    )
    unique_values = get_unique_values(top_indices[:, :k])
    n_unique_recommended_items = unique_values.shape[0]
    return n_unique_recommended_items / n_items


_metric_fn_map_user_accuracy = {
    MetricEnum.DCG: dcg,
    MetricEnum.NDCG: ndcg,
    MetricEnum.Recall: recall,
    MetricEnum.Precision: precision,
    MetricEnum.Hitrate: hitrate,
    MetricEnum.F_Score: f_score,
    MetricEnum.AP: average_precision,
    MetricEnum.RR: reciprocal_rank,
}

# beyond accuracy metrics may require additional parameters
# logits, k and logits_are_top_indices are implicitly assumed to be necessary
_metric_fn_map_user_beyond_accuracy = {
    MetricEnum.AverageRank: (average_rank, ["targets", "item_ranks"]),
}

_metric_fn_map_global = {MetricEnum.Coverage: coverage}

# List of metrics that are currently supported
supported_metrics = tuple(MetricEnum)
supported_user_accuracy_metrics = tuple(_metric_fn_map_user_accuracy.keys())
supported_user_beyond_accuracy_metrics = tuple(
    _metric_fn_map_user_beyond_accuracy.keys()
)
supported_user_metrics = (
    supported_user_accuracy_metrics + supported_user_beyond_accuracy_metrics
)
supported_global_metrics = tuple(_metric_fn_map_global.keys())


def calculate(
    metrics: str | Iterable[str | MetricEnum],
    logits: torch.Tensor | np.ndarray | sp.csr_array = None,
    targets: torch.Tensor | np.ndarray | sp.csr_array = None,
    k: int | Iterable[int] = 10,
    return_aggregated: bool = True,
    return_per_user: bool = False,
    calculate_std: bool = False,
    flatten_results: bool = False,
    flatten_prefix: str = "",
    flatten_suffix: str = "",
    n_items: int = None,
    best_logit_indices: torch.Tensor | np.ndarray = None,
    return_best_logit_indices: bool = False,
    **kwargs,
):
    """
    Computes the values for a given list of metrics.

    :param metrics:             Metric name or list of metrics to compute. Check out
                                'supported_metrics' for a list of all available metrics.
    :param logits:              prediction matrix about item relevance
    :param targets:             0/1 matrix encoding true item relevance, same shape as logits
    :param k:                   levels of "top-k" items to consider in the calculation
    :param return_aggregated:   whether aggregated metric results should be returned.
    :param return_per_user:     whether the results for individual users should be returned
    :param calculate_std:       whether to calculate the standard deviation for the aggregated results
    :param flatten_results:     whether to flatten the results' dictionary.
    :param flatten_prefix:      prefix to use for flattened results, e.g., to differentiate between
                                different splits
    :param flatten_suffix:      suffix to use for flattened results
    :param n_items:             number of items in dataset (in case only best logit indices are supplied)
    :param best_logit_indices:  previously computed indices of the best logits in sorted order
    :param return_best_logit_indices:   whether to return the indices of the best logits
    :param kwargs:                      additional parameters that are passed to the beyond-accuracy metrics

    :return: a nested dictionary depending of structure {`metric`: {`k`: {`key`: `value`}}},
             where `key` depends on the metric and parameters:
             - "user":   ..., # for user-based metrics if 'return_per_user'
             - "mean":   ..., # for user-based metrics if 'return_aggregated'
             - "std":    ..., # for user-based metrics if 'return_aggregated' and 'calculate_std'
             - "global": ..., # for global metrics
             or a nested dictionary with keys in the format
             - `{flatten_prefix}{metric}@{k}{flatten_suffix}`        # for "mean" and "global" keys
                                                                     # due to their importance
             - `{flatten_prefix}{metric}@{k}_{key}{flatten_suffix}`  # for "std" and "user" keys
    """

    k = (k,) if isinstance(k, int) else k
    max_k = max(k)

    # wrap single metrics in a tuple so that the remaining pipeline works as expected.
    # resulting dictionary output should not change
    metrics = (metrics,) if isinstance(metrics, str) else metrics

    # ensure validity of supplied parameters
    not_supported_metrics = [m for m in metrics if m not in supported_metrics]
    if len(not_supported_metrics) > 0:
        raise ValueError(f"Metrics {not_supported_metrics} are not supported")

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

    if not (return_per_user or return_aggregated):
        raise ValueError(
            f"Specify either 'return_per_user' or 'return_aggregated' to receive results."
        )

    n_items = n_items or logits.shape[-1]
    # to speed up computations, only retrieve highest logit indices once (if not already supplied)
    if best_logit_indices is None:
        best_logit_indices = get_top_k(
            logits, max_k, logits_are_top_indices=False, sorted=True
        )

    # first compute on user-level metrics
    user_metrics = _compute_user_metrics(
        metrics=set(metrics).intersection(set(supported_user_metrics)),
        k=k,
        best_logit_indices=best_logit_indices,
        targets=targets,
        **kwargs,
    )

    # gather all user-based metrics into single dict
    metric_results = defaultdict(lambda: defaultdict(lambda: dict()))
    for m, per_metric_results in user_metrics.items():
        for ki, per_k_results in per_metric_results.items():
            if return_per_user:
                metric_results[m][ki]["user"] = per_k_results

            if return_aggregated:
                metric_results[m][ki]["mean"] = per_k_results.mean().item()
                if calculate_std:
                    metric_results[m][ki]["std"] = std(per_k_results)

    # compute global-based metrics
    global_results = _compute_global_metrics(
        metrics=set(metrics).intersection(set(_metric_fn_map_global)),
        best_logit_indices=best_logit_indices,
        k=k,
        n_items=n_items,
    )
    # ... and add them to results collection
    for m, per_metric_results in global_results.items():
        for ki, k_result in per_metric_results.items():
            metric_results[m][ki]["global"] = k_result

    if flatten_results:
        final_results = dict()
        # iterate over all results and flatten them in single dictionary
        for m, per_metric_results in metric_results.items():
            for ki, per_k_results in per_metric_results.items():
                for key, per_key_results in per_k_results.items():
                    # mean and global metrics are likely the most important metrics,
                    # so to make them look nicer, we drop the key "_{key}" for them
                    # note that both are exclusive, so they will not interfere with
                    # each other
                    if key in ["global", "mean"]:
                        final_key = f"{flatten_prefix}{m}@{ki}{flatten_suffix}"
                    else:
                        final_key = f"{flatten_prefix}{m}@{ki}_{key}{flatten_suffix}"
                    final_results[final_key] = per_key_results
    else:
        # convert defaultdict to normal dict to make return values look cleaner
        final_results = {k: dict(v) for k, v in metric_results.items()}

    if return_best_logit_indices:
        # allow reusing best loggits by returning them,
        # e.g., to call "calculate()" another time
        return dict(final_results), best_logit_indices

    return dict(final_results)


def _compute_user_metrics(
    metrics: Iterable[str | MetricEnum],
    k: Iterable[int],
    best_logit_indices: torch.Tensor | np.ndarray,
    targets: torch.Tensor | np.ndarray | sp.csr_array = None,
    **kwargs,
):
    """
    Computes all user-based metrics

    :param metrics:             the user-based metrics to compute
    :param k:                   top k items to consider
    :param best_logit_indices:  previously computed indices of the best logits in sorted order
    :param targets:             0/1 matrix encoding true item relevance, same shape as logits
    :param kwargs:              additional parameters that are passed to the beyond-accuracy metrics

    :return: a nested dictionary with levels for the 'metrics' and 'k', returning the results
             for individual users, e.g., {
        "ndcg": {
            1:  [0.41, 0.23, 0.19, 0.00, 1.00],
            10: [0.52, 0.37, 0.14, 0.12, 0.79],
        },
    }
    """
    results = {}

    # compute user-based accuracy metrics
    user_metrics_to_compute = set(metrics).intersection(
        set(_metric_fn_map_user_accuracy)
    )
    if len(user_metrics_to_compute):
        results.update(
            _compute_user_accuracy_metrics(
                metrics=user_metrics_to_compute,
                k=k,
                best_logit_indices=best_logit_indices,
                targets=targets,
            )
        )

    # compute user-based beyond accuracy metrics
    user_beyond_metrics_to_compute = set(metrics).intersection(
        set(_metric_fn_map_user_beyond_accuracy)
    )
    if len(user_beyond_metrics_to_compute):
        results.update(
            _compute_user_beyond_accuracy_metrics(
                metrics=user_beyond_metrics_to_compute,
                k=k,
                best_logit_indices=best_logit_indices,
                targets=targets,
                **kwargs,
            )
        )

    return results


def _compute_user_accuracy_metrics(
    metrics: Iterable[str | MetricEnum],
    k: Iterable[int],
    best_logit_indices: torch.Tensor | np.ndarray,
    targets: torch.Tensor | np.ndarray | sp.csr_array,
):
    """
    Computes the given user-based accuracy metrics

    :param metrics:             the metrics to compute
    :param k:                   top k items to consider
    :param best_logit_indices:  previously computed indices of the best logits in sorted order
    :param targets:             0/1 matrix encoding true item relevance, same shape as logits

    :return: a nested dictionary with levels for the 'metrics' and 'k', returning the results
             for individual users, e.g., {
        "ndcg": {
            1:  [0.41, 0.23, 0.19, 0.00, 1.00],
            10: [0.52, 0.37, 0.14, 0.12, 0.79],
        },
    }
    """
    results = {}

    if targets is None:
        raise ValueError(f"'targets' is required to calculate '{metric}'!")

    # do not compute metrics for users where we do not have any
    # underlying ground truth interactions
    mask = generate_non_zero_mask(targets)

    results = defaultdict(lambda: dict())
    for ki in k:
        for metric in metrics:
            # compute metrics only for users with targets
            metric_result = zeros_float(targets.shape[0], targets)
            metric_result[mask] = _metric_fn_map_user_accuracy[metric](
                logits=best_logit_indices[mask],
                targets=targets[mask],
                k=ki,
                logits_are_top_indices=True,
            )
            results[str(metric)][ki] = metric_result

    return dict(results)


def _compute_user_beyond_accuracy_metrics(
    metrics: Iterable[str | MetricEnum],
    k: Iterable[int],
    best_logit_indices: torch.Tensor | np.ndarray,
    **kwargs,
):
    """
    Computes the given user-based beyond-accuracy metrics

    :param metrics:             the metrics to compute
    :param k:                   top k items to consider
    :param best_logit_indices:  previously computed indices of the best logits in sorted order
    :param targets:             0/1 matrix encoding true item relevance, same shape as logits
    :param kwargs:              additional parameters that are passed to the beyond-accuracy metrics

    :return: a nested dictionary with levels for the 'metrics' and 'k', returning the results
             for individual users, e.g., {
        "average_rank": {
            1:  [0.41, 0.23, 0.19, 0.00, 1.00],
            10: [0.52, 0.37, 0.14, 0.12, 0.79],
        },
    }
    """
    results = defaultdict(lambda: dict())
    for ki in k:
        for metric in metrics:
            fn, required_params = _metric_fn_map_user_beyond_accuracy[metric]
            results[str(metric)][ki] = fn(
                logits=best_logit_indices,
                k=ki,
                logits_are_top_indices=True,
                **{p: kwargs.get(p) for p in required_params},
            )
    return dict(results)


def _compute_global_metrics(
    metrics: Iterable[str | MetricEnum],
    k: Iterable[int],
    best_logit_indices: torch.Tensor | np.ndarray,
    n_items: int,
):
    """
    Computes the given global metrics

    :param metrics:             the metrics to compute
    :param k:                   top k items to consider
    :param best_logit_indices:  previously computed indices of the best logits in sorted order
    :param n_items:             number of items in dataset (in case only best logit indices are supplied)

    :return: a nested dictionary with levels for the 'metrics' and 'k', returning the
             global results, e.g., {
                 "coverage": {
                    1:  0.01,
                    10: 0.21,
                 }
             }}
    """
    results = defaultdict(lambda: dict())
    for ki in k:
        for metric in metrics:
            results[str(metric)][ki] = _metric_fn_map_global[metric](
                logits=best_logit_indices,
                k=ki,
                n_items=n_items,
                logits_are_top_indices=True,
            )
    return dict(results)
