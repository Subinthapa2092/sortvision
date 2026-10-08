"""
Grad-CAM (Gradient-weighted Class Activation Mapping) for the waste
classifier.

Shows which pixels in an image most influenced the model's prediction
for a given class, answering "which part of the image did the model look
at when predicting plastic?"

Works for both model families in this project:

1. The from scratch CNN, where the target is the layer named 'last_conv'.
2. Transfer learning models, where the real convolutional layers live
   inside a nested backbone sub-model (MobileNetV2, EfficientNetB0,
   ResNet50). The target is the backbone's final feature map.

The two need different handling. A layer taken from inside a nested
sub-model is not connected to the outer model's input, so Keras refuses
to build a gradient model around it. Instead, transfer models are split
at the backbone and run in three connected pieces: the preprocessing in
front of it, the backbone itself, and the classifier head after it.
"""

import numpy as np
import tensorflow as tf
from PIL import Image


def _find_backbone_layer(model: tf.keras.Model):
    """Return the nested backbone sub-model of a transfer model, or None."""
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            return layer
    return None


def _scratch_heatmap(model, img_array, class_index):
    conv_layer = model.get_layer("last_conv")
    grad_model = tf.keras.models.Model(
        inputs=model.inputs, outputs=[conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model([img_array])
        if class_index is None:
            class_index = int(tf.argmax(predictions[0]))
        class_channel = predictions[:, class_index]

    grads = tape.gradient(class_channel, conv_output)
    return conv_output, grads, class_index


def _split_transfer_model(model, backbone):
    """Split a transfer model into (pre_model, head_layers) around its
    backbone.

    pre_model maps raw input pixels to exactly what the backbone receives,
    including the preprocessing steps in front of it. head_layers are the
    layers after the backbone, replayed in order.

    Uses model.operations, not model.layers. The preprocessing steps (for
    example the divide and subtract of MobileNetV2 preprocessing, or the
    channel flip and mean subtraction of ResNet50) are operations, not
    layers, so model.layers silently skips them and would feed the
    backbone raw 0 to 255 pixels instead of the scaled values it was
    trained on.
    """
    operations = model.operations
    backbone_index = operations.index(backbone)

    pre_model = tf.keras.models.Model(
        inputs=model.inputs, outputs=operations[backbone_index - 1].output
    )

    head = operations[backbone_index + 1:]
    for op in head:
        if not isinstance(op, tf.keras.layers.Layer):
            raise ValueError(
                f"Unexpected non-layer operation '{op.name}' after the "
                "backbone, Grad-CAM cannot replay this model's head."
            )
    return pre_model, head


def _transfer_heatmap(model, backbone, img_array, class_index):
    pre_model, head = _split_transfer_model(model, backbone)

    preprocessed = pre_model(tf.convert_to_tensor(img_array, dtype=tf.float32))

    with tf.GradientTape() as tape:
        feature_map = backbone(preprocessed, training=False)
        tape.watch(feature_map)
        x = feature_map
        for layer in head:
            x = layer(x)
        predictions = x
        if class_index is None:
            class_index = int(tf.argmax(predictions[0]))
        class_channel = predictions[:, class_index]

    grads = tape.gradient(class_channel, feature_map)
    return feature_map, grads, class_index


def compute_gradcam_heatmap(
    model: tf.keras.Model,
    img_array: np.ndarray,
    class_index: int = None,
):
    """Compute a Grad-CAM heatmap for one image.

    img_array should be shape (1, H, W, 3) with raw 0 to 255 pixel
    values, the same batch of one you would pass to model.predict.

    Returns (heatmap, class_index). The heatmap is a 2D array with values
    in [0, 1] at the spatial resolution of the target feature map, which
    is much smaller than the input image, the caller resizes it up for
    display.
    """
    backbone = _find_backbone_layer(model)

    if backbone is None:
        if "last_conv" not in [layer.name for layer in model.layers]:
            raise ValueError(
                "Could not find a convolutional layer to target for "
                "Grad-CAM. Expected a layer named 'last_conv' or a nested "
                "backbone sub-model."
            )
        feature_map, grads, class_index = _scratch_heatmap(model, img_array, class_index)
    else:
        feature_map, grads, class_index = _transfer_heatmap(
            model, backbone, img_array, class_index
        )

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    feature_map = feature_map[0]
    heatmap = feature_map @ pooled_grads[..., tf.newaxis]
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

    # simple heat colormap: low = blue-ish, high = red
    overlay = np.zeros((*heatmap_arr.shape, 3), dtype=np.float32)
    overlay[..., 0] = heatmap_arr               # red channel scales with intensity
    overlay[..., 2] = 1.0 - heatmap_arr          # blue fades out as intensity rises

    base = np.asarray(original_img.convert("RGB"), dtype=np.float32) / 255.0
    blended = base * (1 - alpha) + overlay * alpha
    blended = np.clip(blended * 255, 0, 255).astype(np.uint8)

    return Image.fromarray(blended)