"""
Grad-CAM (Gradient-weighted Class Activation Mapping) for the waste
classifier.

Shows which pixels in an image most influenced the model's prediction
for a given class, answering "which part of the image did the model look
at when predicting plastic?"

Works for both the from-scratch CNN (last_conv layer on the main model)
and transfer learning models (last_conv layer nested inside the backbone
sub-model), since these have different internal structures.
"""

import numpy as np
import tensorflow as tf
from PIL import Image


def _find_conv_model_and_layer(model: tf.keras.Model):
    """Locate the model and layer name to target for Grad-CAM.

    For the from-scratch CNN, the target layer ('last_conv') lives
    directly on the passed-in model. For transfer learning models, the
    real convolutional layers live inside a nested backbone sub-model
    (e.g. MobileNetV2), so Grad-CAM must build its gradient model against
    that nested sub-model's last conv layer instead.
    """
    layer_names = [layer.name for layer in model.layers]
    if "last_conv" in layer_names:
        return model, "last_conv"

    # look for a nested backbone sub-model and use its last Conv2D layer
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            conv_layers = [l.name for l in layer.layers if isinstance(l, tf.keras.layers.Conv2D)]
            if conv_layers:
                return layer, conv_layers[-1]

    raise ValueError(
        "Could not find a convolutional layer to target for Grad-CAM. "
        "Pass the layer name explicitly if using a custom architecture."
    )


def compute_gradcam_heatmap(
    model: tf.keras.Model,
    img_array: np.ndarray,
    class_index: int = None,
) -> np.ndarray:
    """Compute a Grad-CAM heatmap for one preprocessed image.

    img_array should be shape (1, H, W, 3), the same batch-of-one input
    you would pass to model.predict.

    Returns a 2D heatmap, values in [0, 1], at the spatial resolution of
    the target conv layer (typically much smaller than the input image,
    the caller resizes it back up for display).
    """
    conv_model, conv_layer_name = _find_conv_model_and_layer(model)
    conv_layer = conv_model.get_layer(conv_layer_name)

    grad_model = tf.keras.models.Model(
        inputs=model.inputs, outputs=[conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(img_array)
        if class_index is None:
            class_index = int(tf.argmax(predictions[0]))
        class_channel = predictions[:, class_index]

    grads = tape.gradient(class_channel, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy(), class_index


def overlay_heatmap(
    original_img: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.45,
) -> Image.Image:
    """Resize the heatmap to the original image size and blend it on top
    using a simple red-channel overlay, no external colormap dependency
    needed beyond what PIL already provides.
    """
    heatmap_img = Image.fromarray(np.uint8(255 * heatmap)).resize(original_img.size)
    heatmap_arr = np.asarray(heatmap_img, dtype=np.float32) / 255.0

    # simple heat colormap: low = transparent/blue-ish, high = red
    overlay = np.zeros((*heatmap_arr.shape, 3), dtype=np.float32)
    overlay[..., 0] = heatmap_arr               # red channel scales with intensity
    overlay[..., 2] = 1.0 - heatmap_arr          # blue fades out as intensity rises

    base = np.asarray(original_img.convert("RGB"), dtype=np.float32) / 255.0
    blended = base * (1 - alpha) + overlay * alpha
    blended = np.clip(blended * 255, 0, 255).astype(np.uint8)

    return Image.fromarray(blended)
