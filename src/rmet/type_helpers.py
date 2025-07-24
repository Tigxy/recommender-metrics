import torch
import numpy as np
import scipy.sparse as sp
from typing import Iterable


def assert_supported_type(a: any):
    if not isinstance(a, (np.ndarray, torch.Tensor, sp.csr_array)):
        raise TypeError(f"Type {type(a)} of input not supported.")


def zeros_like_float(a: torch.Tensor | np.ndarray):
    if isinstance(a, torch.Tensor):
        return torch.zeros_like(a, dtype=torch.float, device=a.device)

    elif isinstance(a, np.ndarray):
        return np.zeros_like(a, dtype=float)

    else:
        assert_supported_type(a)


def zeros_float(shape, reference_a):
    if isinstance(reference_a, torch.Tensor):
        return torch.zeros(shape, device=reference_a.device)

    elif isinstance(reference_a, np.ndarray | sp.csr_array):
        return np.zeros(shape)
    else:
        assert_supported_type(reference_a)


def as_float(a: torch.Tensor | np.ndarray):
    if isinstance(a, torch.Tensor):
        return a.float()

    elif isinstance(a, np.ndarray):
        return a.astype(float)

    else:
        assert_supported_type(a)


def stack(a: Iterable[np.ndarray] | Iterable[torch.Tensor]):
    if len(a) == 0:
        raise ValueError("Cannot perform stack on empty iterable.")

    if isinstance(a[0], torch.Tensor):
        return torch.stack(a)
    else:
        return np.stack(a)


def generate_non_zero_mask(a: torch.Tensor | np.ndarray, dim=-1):
    non_zero_counter = a.sum(-1)

    if isinstance(non_zero_counter, torch.Tensor):
        return torch.argwhere(non_zero_counter).flatten()

    elif isinstance(non_zero_counter, np.ndarray | sp.csr_array):
        return np.argwhere(non_zero_counter).flatten()

    else:
        assert_supported_type(non_zero_counter)


def get_unique_values(top_indices: torch.Tensor | np.ndarray, sorted=False):
    if isinstance(top_indices, torch.Tensor):
        return top_indices.unique(sorted=sorted)

    elif isinstance(top_indices, np.ndarray):
        return np.unique(top_indices)

    else:
        assert_supported_type(top_indices)


def std(a: torch.Tensor | np.ndarray):
    if isinstance(a, torch.Tensor):
        return torch.std(a).item()

    elif isinstance(a, np.ndarray):
        # set degrees of freedom to 1 to have same results as torch
        return np.std(a, ddof=1).item()

    else:
        assert_supported_type(top_indices)


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


def get_top_k(
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
            assert_supported_type(logits)
