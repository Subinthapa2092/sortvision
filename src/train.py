"""
Train the waste classifier, either a from-scratch CNN or a transfer
learning model on top of a pretrained backbone.

Examples
--------
From-scratch CNN, fast subsample for CPU iteration:
    python -m src.train --model_type scratch --max_train_samples 5000

Transfer learning with MobileNetV2:
    python -m src.train --model_type transfer --backbone mobilenet_v2

Full dataset, from-scratch CNN:
    python -m src.train --model_type scratch
"""

import argparse

import tensorflow as tf

import config
from src.data_loader import load_waste_dataset, get_class_names
from src.preprocessing import compute_class_weights
from src.dataset import make_dataset
from src.model import build_cnn_from_scratch, build_transfer_model, compile_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train the waste classifier")
    parser.add_argument("--model_type", choices=["scratch", "transfer"], default="scratch")
    parser.add_argument(
        "--backbone",
        choices=["mobilenet_v2", "efficientnet_b0", "resnet50"],
        default="mobilenet_v2",
        help="Only used when --model_type transfer",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="If set, use only this many training examples (fast CPU testing).",
    )
    parser.add_argument(
        "--max_val_samples",
        type=int,
        default=None,
        help="If set, use only this many validation examples.",
    )
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    return parser.parse_args()


def main():
    args = parse_args()

    print("Loading dataset from Hugging Face...")
    ds = load_waste_dataset()

    names = get_class_names(ds)
    if names != config.CLASS_NAMES:
        print(
            "WARNING: dataset class order does not match config.CLASS_NAMES. "
            f"Dataset says: {names}. Update config.py to match before trusting "
            "any results."
        )

    train_split = ds["train"]
    val_split = ds["validation"]

    if args.max_train_samples and len(train_split) > args.max_train_samples:
        train_split = train_split.shuffle(seed=config.RANDOM_STATE).select(
            range(args.max_train_samples)
        )
    if args.max_val_samples and len(val_split) > args.max_val_samples:
        val_split = val_split.shuffle(seed=config.RANDOM_STATE).select(
            range(args.max_val_samples)
        )

    print(f"Using {len(train_split):,} train / {len(val_split):,} val examples for this run.")

    train_ds = make_dataset(
        train_split, config.IMAGE_SIZE, args.batch_size, config.NUM_CLASSES, training=True
    )
    val_ds = make_dataset(
        val_split, config.IMAGE_SIZE, args.batch_size, config.NUM_CLASSES, training=False
    )

    if args.model_type == "scratch":
        model = build_cnn_from_scratch(config.NUM_CLASSES, config.IMAGE_SIZE, config.CHANNELS)
        save_path = config.CNN_MODEL_PATH
    else:
        model = build_transfer_model(
            args.backbone, config.NUM_CLASSES, config.IMAGE_SIZE, config.CHANNELS
        )
        save_path = config.CNN_MODEL_PATH.replace(".keras", f"_{args.backbone}.keras")

    model = compile_model(model, config.LEARNING_RATE)
    model.summary()

    class_weights = compute_class_weights(train_split["label"], config.NUM_CLASSES)
    print(f"Class weights: {class_weights}")

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy",
            mode="max",
            factor=config.REDUCE_LR_FACTOR,
            patience=config.REDUCE_LR_PATIENCE,
        ),
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        class_weight=class_weights,
        callbacks=callbacks,
    )

    model.save(save_path)
    print(f"Saved trained model to {save_path}")


if __name__ == "__main__":
    main()
