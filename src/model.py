"""
Model architectures for multi-class waste classification.

Unlike the binary sigmoid output used in a two-class project, this uses a
softmax output over num_classes, since an image belongs to exactly one of
six waste categories, not a yes/no decision.
"""

import tensorflow as tf
from tensorflow.keras import layers, models


def build_cnn_from_scratch(
    num_classes: int, image_size: int = 224, channels: int = 3
) -> tf.keras.Model:
    inputs = layers.Input(shape=(image_size, image_size, channels))
    x = layers.Rescaling(1.0 / 255)(inputs)

    # Block 1
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    # Block 2
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    # Block 3
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    # Block 4, real photos at 224x224 have more to learn from than a
    # 50x50 microscope patch, so this gets one more block than a
    # small-patch project would use.
    x = layers.Conv2D(256, 3, padding="same", activation="relu", name="last_conv")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return models.Model(inputs, outputs, name="sortvision_cnn_from_scratch")


_BACKBONES = {
    "mobilenet_v2": (tf.keras.applications.MobileNetV2, tf.keras.applications.mobilenet_v2.preprocess_input),
    "efficientnet_b0": (tf.keras.applications.EfficientNetB0, tf.keras.applications.efficientnet.preprocess_input),
    "resnet50": (tf.keras.applications.ResNet50, tf.keras.applications.resnet50.preprocess_input),
}


def build_transfer_model(
    backbone_name: str,
    num_classes: int,
    image_size: int = 224,
    channels: int = 3,
    trainable: bool = False,
) -> tf.keras.Model:
    if backbone_name not in _BACKBONES:
        raise ValueError(
            f"Unknown backbone '{backbone_name}'. Choose from {list(_BACKBONES)}"
        )

    backbone_cls, preprocess = _BACKBONES[backbone_name]

    inputs = layers.Input(shape=(image_size, image_size, channels))
    x = preprocess(inputs)

    base_model = backbone_cls(
        include_top=False, weights="imagenet", input_shape=(image_size, image_size, 3)
    )
    base_model.trainable = trainable

    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D(name="last_conv_pool")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return models.Model(inputs, outputs, name=f"sortvision_transfer_{backbone_name}")


def get_last_conv_layer_name(model: tf.keras.Model, backbone_name: str = None) -> str:
    """Return the name of the last convolutional layer, for Grad-CAM.

    For the from-scratch CNN this is the explicitly named 'last_conv'
    layer. For transfer models, the last conv layer lives inside the
    nested backbone sub-model, so Grad-CAM needs to target that nested
    model's last conv layer, not a layer on the outer model directly.
    """
    if backbone_name is None:
        return "last_conv"

    # the backbone is nested as a sub-model; find its name and let the
    # caller (src/gradcam.py) look inside it specifically
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            return layer.name
    raise ValueError("Could not find a nested backbone model for Grad-CAM.")


def compile_model(model: tf.keras.Model, learning_rate: float = 1e-3) -> tf.keras.Model:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=[
            tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc", multi_label=False),
        ],
    )
    return model
