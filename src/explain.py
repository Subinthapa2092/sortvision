"""
Generate a Grad-CAM visualization for a single image: shows the
predicted class, confidence, and a heatmap overlay of where the model
was looking.

Usage:
    python -m src.explain --image_path path/to/image.jpg
"""

import argparse
import os

import numpy as np
import tensorflow as tf
from PIL import Image

import config
from src.gradcam import compute_gradcam_heatmap, overlay_heatmap


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", required=True)
    parser.add_argument("--model_path", default=config.CNN_MODEL_PATH)
    parser.add_argument(
        "--output_path",
        default=os.path.join(config.OUTPUTS_DIR, "gradcam_overlay.png"),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    model = tf.keras.models.load_model(args.model_path)

    original = Image.open(args.image_path).convert("RGB")
    resized = original.resize((config.IMAGE_SIZE, config.IMAGE_SIZE))
    arr = np.expand_dims(np.asarray(resized, dtype=np.float32), axis=0)

    heatmap, class_index = compute_gradcam_heatmap(model, arr)
    class_name = config.CLASS_NAMES[class_index]
    display_name = config.DISPLAY_NAMES.get(class_name, class_name)

    probs = model.predict(arr, verbose=0)[0]
    confidence = float(probs[class_index])

    overlay = overlay_heatmap(resized, heatmap)
    overlay.save(args.output_path)

    print(f"Image: {args.image_path}")
    print(f"Predicted: {display_name} ({confidence * 100:.1f}% confidence)")
    print(f"Grad-CAM overlay saved to: {args.output_path}")


if __name__ == "__main__":
    main()
