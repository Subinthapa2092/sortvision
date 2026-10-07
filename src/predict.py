"""
Run the trained classifier on a single image.

Usage:
    python -m src.predict --image_path path/to/image.jpg
"""

import argparse

import numpy as np
import tensorflow as tf
from PIL import Image

import config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", required=True)
    parser.add_argument("--model_path", default=config.CNN_MODEL_PATH)
    return parser.parse_args()


def main():
    args = parse_args()

    model = tf.keras.models.load_model(args.model_path)

    img = Image.open(args.image_path).convert("RGB")
    img = img.resize((config.IMAGE_SIZE, config.IMAGE_SIZE))
    arr = np.expand_dims(np.asarray(img, dtype=np.float32), axis=0)

    probs = model.predict(arr, verbose=0)[0]
    ranked = sorted(
        zip(config.CLASS_NAMES, probs), key=lambda pair: pair[1], reverse=True
    )

    print(f"Image: {args.image_path}\n")
    for class_name, prob in ranked:
        display_name = config.DISPLAY_NAMES.get(class_name, class_name)
        print(f"  {display_name:16s} {prob * 100:5.1f}%")

    top_class, top_prob = ranked[0]
    top_display = config.DISPLAY_NAMES.get(top_class, top_class)
    print(f"\nPredicted: {top_display} ({top_prob * 100:.1f}% confidence)")


if __name__ == "__main__":
    main()
