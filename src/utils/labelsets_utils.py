import json
import os
import numpy as np
try:
    from .constants import CIFAR20_COARSE, CIFAR20_FINE, CIFAR20_LABELS
except ImportError:
    from constants import CIFAR20_COARSE, CIFAR20_FINE, CIFAR20_LABELS

def load_label_set(dataset, size, label_sets_dir=None):
    """Load a label-set JSON for a dataset and size (e.g., 'cifar20', 10).

    Looks for file at <label_sets_dir>/<dataset>-<size>.json or ./label_sets/<dataset>-<size>.json
    Returns dict mapping superclass -> [subclass names]
    """
    if label_sets_dir is None:
        label_sets_dir = os.path.join(os.path.dirname(__file__), "../../label_sets")
    fname = os.path.join(label_sets_dir, f"{dataset}-{size}.json")
    if not os.path.isfile(fname):
        raise FileNotFoundError(f"Label set file not found: {fname}")
    with open(fname, "r", encoding="utf-8") as f:
        return json.load(f)


def get_true_cifar20_labelset():
    """Return the ground-truth CIFAR-20 mapping (superclass -> list of fine class names).

    Uses constants CIFAR20_COARSE, CIFAR20_FINE and CIFAR20_LABELS to reconstruct the
    mapping in the same order expected by CHiLS code.
    """
    sup2sub = {}
    # Ensure coarse order is preserved
    for fine_idx, fine_name in enumerate(CIFAR20_FINE):
        coarse_idx = int(CIFAR20_LABELS[fine_idx])
        coarse_name = CIFAR20_COARSE[coarse_idx]
        sup2sub.setdefault(coarse_name, []).append(fine_name)
    # sort keys for deterministic order
    sup2sub = {k: sup2sub[k] for k in CIFAR20_COARSE if k in sup2sub}
    return sup2sub


def transform_labels(label_map, labels):
    """Apply a mapping dict label_map to an iterable/array of labels.

    Example: label_map maps original fine-label indices -> new superclass indices.
    """
    labels = np.array(labels)
    return np.array([label_map[int(i)] for i in labels])

