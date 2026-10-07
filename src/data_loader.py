"""
Loads the TrashBusters/combined dataset from Hugging Face.

Unlike the folder-per-class layout used in some image projects, this
dataset ships as a Hugging Face DatasetDict with its own train,
validation, and test splits already defined, images stored as PIL
Images, and labels stored as integer class indices. This module is a
thin, honest wrapper around that, it does not re-split or re-balance
anything the dataset authors already did.
"""

from datasets import load_dataset, DatasetDict

import config


def load_waste_dataset() -> DatasetDict:
    """Download (or use the local cache of) TrashBusters/combined.

    Returns a DatasetDict with 'train', 'validation', and 'test' splits.
    Each example has an 'image' (PIL Image) and a 'label' (int) field.
    """
    ds = load_dataset(config.HF_DATASET_NAME)
    return ds


def get_class_names(ds: DatasetDict) -> list:
    """Read the real class name order from the dataset itself rather than
    trusting config.CLASS_NAMES blindly. Call this once after loading and
    compare against config.CLASS_NAMES, they should match, but the dataset
    is the source of truth.
    """
    return ds["train"].features["label"].names


if __name__ == "__main__":
    ds = load_waste_dataset()
    names = get_class_names(ds)

    print("Splits:", {split: len(ds[split]) for split in ds})
    print("Class names from dataset:", names)
    print("Class names in config.py:", config.CLASS_NAMES)

    if names != config.CLASS_NAMES:
        print(
            "\nWARNING: config.CLASS_NAMES does not match the dataset's "
            "actual label order. Update config.py to match the printed "
            "'Class names from dataset' list above before training, or "
            "labels will be silently scrambled."
        )

    # class balance check
    import collections

    train_labels = ds["train"]["label"]
    counts = collections.Counter(train_labels)
    print("\nTrain class distribution:")
    for idx, name in enumerate(names):
        count = counts.get(idx, 0)
        pct = 100 * count / len(train_labels)
        print(f"  {name:12s} {count:6,}  ({pct:5.1f}%)")
