"""
Preprocessing utilities: class weight computation for multi-class
imbalance. The TrashBusters/combined dataset is fairly close to balanced
(roughly 5,000 images per class out of 6), but computing weights properly
rather than assuming balance is cheap and protects against surprises if
the dataset changes or a different one is substituted later.
"""

import collections


def compute_class_weights(labels: list, num_classes: int) -> dict:
    """Inverse-frequency class weighting, keyed by integer class index
    for Keras' class_weight argument.
    """
    counts = collections.Counter(labels)
    total = len(labels)

    weights = {}
    for cls in range(num_classes):
        count = counts.get(cls, 0)
        if count == 0:
            weights[cls] = 1.0
        else:
            weights[cls] = total / (num_classes * count)
    return weights
