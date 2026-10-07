"""
Evaluate a trained waste classifier on the test split.

Produces accuracy, macro and weighted precision/recall/F1, a full
per-class classification report, a confusion matrix plot, and a list of
the most commonly confused class pairs, since "which types of waste does
the model confuse" is one of the project's explicit evaluation goals, not
just overall accuracy.
"""

import argparse
import os

import numpy as np
import tensorflow as tf
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

import config
from src.data_loader import load_waste_dataset, get_class_names
from src.dataset import make_dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default=config.CNN_MODEL_PATH)
    parser.add_argument("--max_test_samples", type=int, default=None)
    return parser.parse_args()


def most_confused_pairs(cm: np.ndarray, class_names: list, top_n: int = 5):
    """Return the top_n (true_class, predicted_class, count) triples with
    the highest off-diagonal confusion matrix counts, sorted descending.
    """
    pairs = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i != j and cm[i, j] > 0:
                pairs.append((class_names[i], class_names[j], int(cm[i, j])))
    pairs.sort(key=lambda p: p[2], reverse=True)
    return pairs[:top_n]


def main():
    args = parse_args()

    ds = load_waste_dataset()
    names = get_class_names(ds)
    display_names = [config.DISPLAY_NAMES.get(n, n) for n in names]

    test_split = ds["test"]
    if args.max_test_samples and len(test_split) > args.max_test_samples:
        test_split = test_split.shuffle(seed=config.RANDOM_STATE).select(
            range(args.max_test_samples)
        )

    print(f"Evaluating on {len(test_split):,} test images")

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(
            f"No model found at {args.model_path}. Train one first with src.train."
        )

    model = tf.keras.models.load_model(args.model_path)
    test_ds = make_dataset(
        test_split, config.IMAGE_SIZE, config.BATCH_SIZE, config.NUM_CLASSES, training=False
    )

    y_true = np.array(test_split["label"])
    y_prob = model.predict(test_ds)
    y_pred = np.argmax(y_prob, axis=1)

    acc = accuracy_score(y_true, y_pred)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    print(f"\nAccuracy: {acc:.4f}")
    print(f"Macro    precision/recall/F1: {precision_macro:.4f} / {recall_macro:.4f} / {f1_macro:.4f}")
    print(f"Weighted precision/recall/F1: {precision_weighted:.4f} / {recall_weighted:.4f} / {f1_weighted:.4f}")

    print("\nPer-class report:")
    print(classification_report(y_true, y_pred, target_names=display_names, zero_division=0))

    cm = confusion_matrix(y_true, y_pred)

    confused = most_confused_pairs(cm, display_names)
    print("Most confused class pairs (true -> predicted, count):")
    for true_cls, pred_cls, count in confused:
        print(f"  {true_cls} -> {pred_cls}: {count}")

    fig, ax = plt.subplots(figsize=(8, 8))
    disp = ConfusionMatrixDisplay(cm, display_labels=display_names)
    disp.plot(cmap="Greens", ax=ax, xticks_rotation=45)
    plt.title("Confusion Matrix - SortVision")
    plt.tight_layout()
    cm_path = os.path.join(config.OUTPUTS_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {cm_path}")

    # per-class bar chart, useful alongside the confusion matrix to see
    # at a glance which classes the model is weakest on
    precision_per, recall_per, f1_per, support_per = precision_recall_fscore_support(
        y_true, y_pred, zero_division=0
    )
    x = np.arange(len(display_names))
    width = 0.25
    plt.figure(figsize=(9, 5))
    plt.bar(x - width, precision_per, width, label="Precision")
    plt.bar(x, recall_per, width, label="Recall")
    plt.bar(x + width, f1_per, width, label="F1")
    plt.xticks(x, display_names, rotation=30, ha="right")
    plt.ylim(0, 1)
    plt.title("Per-Class Performance - SortVision")
    plt.legend()
    plt.tight_layout()
    per_class_path = os.path.join(config.OUTPUTS_DIR, "per_class_performance.png")
    plt.savefig(per_class_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {per_class_path}")


if __name__ == "__main__":
    main()
