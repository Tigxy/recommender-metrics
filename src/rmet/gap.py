import torch
import numpy as np
import itertools
from . import calculate


class UserFeature:
    def __init__(self, name: str, labels: list):
        """
        Splits users based on some arbitrary feature in different groups. This is used
        to ease calculating differences of recommendation systems for users with different demographics, e.g., gender.

        :param name: The name of the feature.
        :param labels: The labels for the individual users of the feature. The users are grouped based on them.
        """

        self.name = name
        self.labels = labels
        self.unique_labels = set(labels)

        # gather mapping for labels to indices
        self.label_indices_map = {lbl: np.array([i for i, l in enumerate(self.labels) if l == lbl]) for lbl in labels}

    def count(self):
        return {k: len(v) for k, v in self.label_indices_map.items()}

    def __repr__(self):
        return str(self)

    def __str__(self):
        return f"UserFeatureGroup(name={self.name}, counts={self.count()})"

    def __iter__(self):
        for k, v in self.label_indices_map.items():
            yield k, v


def __mean(v):
    return torch.mean(v).item() if isinstance(v, torch.Tensor) else v


def calculate_for_feature(group: UserFeature, metrics: list, logits: torch.Tensor, targets=None,
                          k=10, return_individual=False):
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
        results[f"{group.name}_{lbl}"] = calculate(metrics, logits[indices], t, k, return_individual)

    pairs = list(itertools.combinations(group.unique_labels, 2))
    for a, b in pairs:
        results[f"{group.name}_{a}-{b}"] = {m: (__mean(results[f"{group.name}_{a}"][m]) -
                                               __mean(results[f"{group.name}_{b}"][m]))
                                            for m in metrics}
    return results
