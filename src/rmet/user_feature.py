import numpy as np


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
        self.unique_labels = tuple(sorted(set(labels)))

        self.label_map = {lbl: i for i, lbl in enumerate(self.unique_labels)}
        self.label_encodings = np.array(
            [self.label_map[lbl] for lbl in labels], dtype=int
        )

        self.label_indices_map = {
            lbl: np.argwhere(self.label_encodings == self.label_map[lbl]).flatten()
            for lbl in self.unique_labels
        }

    def count(self):
        return {k: len(v) for k, v in self.label_indices_map.items()}

    def __repr__(self):
        return str(self)

    def __str__(self):
        return f"UserFeature(name={self.name}, counts={self.count()})"

    def __iter__(self):
        for k, v in self.label_indices_map.items():
            yield k, v
