"""
tf.data pipeline built on top of a Hugging Face dataset split.

Hugging Face datasets store images as PIL Images rather than file paths,
so instead of tf.io.decode_png (used in a folder-of-files project), this
pulls images through a Python generator. This is a little slower than a
pure tf.data file pipeline, but it avoids writing 29,849 images back out
to disk just to read them again.
"""

import numpy as np
import tensorflow as tf


def _generator(hf_split, image_size):
    for example in hf_split:
        img = example["image"].convert("RGB").resize((image_size, image_size))
        arr = np.asarray(img, dtype=np.float32)
        yield arr, example["label"]


def _augment(img, label):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.random_brightness(img, max_delta=0.15)
    img = tf.image.random_contrast(img, lower=0.85, upper=1.15)
    return img, label


def make_dataset(
    hf_split,
    image_size: int,
    batch_size: int,
    num_classes: int,
    training: bool = False,
    shuffle_buffer: int = 2048,
):
    """Build a tf.data.Dataset from one split of a Hugging Face DatasetDict."""
    output_signature = (
        tf.TensorSpec(shape=(image_size, image_size, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int64),
    )

    ds = tf.data.Dataset.from_generator(
        lambda: _generator(hf_split, image_size),
        output_signature=output_signature,
    )

    if training:
        ds = ds.shuffle(buffer_size=min(shuffle_buffer, len(hf_split)), seed=42)
        ds = ds.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.map(
        lambda img, label: (img, tf.one_hot(label, num_classes)),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds
