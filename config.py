"""
Central configuration for the SortVision waste classification project.

Dataset: TrashBusters/combined on Hugging Face
https://huggingface.co/datasets/TrashBusters/combined

A harmonized merge of four public waste image datasets (TrashNet,
Household Garbage, Garbage Classification V2, TriCascade), 29,849 images
across 6 classes, with a pre-made train, validation, test split.

Note on categories: the original project brief called for six categories
(Plastic, Paper/Cardboard, Metal, Glass, Organic, Other/General). This
dataset's native classes are cardboard, glass, metal, paper, plastic,
trash, which keeps paper and cardboard as separate classes and has no
dedicated Organic class (food and biological waste falls under trash).
This project trains on the dataset's native 6 classes first, and treats
merging paper and cardboard, and adding a dedicated Organic class from a
second dataset, as a deliberate second pass rather than silently
reinterpreting the data.
"""

import os

# --- Paths -----------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

CNN_MODEL_PATH = os.path.join(MODEL_DIR, "sortvision_cnn.keras")

# The model the web app serves. Kept separate from CNN_MODEL_PATH on purpose:
# train.py builds its save paths from CNN_MODEL_PATH, so pointing
# CNN_MODEL_PATH at a transfer model would make a later from scratch
# training run overwrite it.
APP_MODEL_PATH = os.path.join(MODEL_DIR, "sortvision_cnn_efficientnet_b0.keras")

# --- Dataset -----------------------------------------------------------------
HF_DATASET_NAME = "TrashBusters/combined"

# Native class order as published by the dataset. Verify this against
# dataset["train"].features["label"].names the first time you load it,
# Hugging Face datasets usually sort ClassLabel names alphabetically but
# this should be confirmed rather than assumed blindly.
CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
NUM_CLASSES = len(CLASS_NAMES)

# Friendlier names for display in the UI and reports. "trash" is the
# dataset's catch-all bucket, shown as "Other / General" since that is
# what it actually contains (not literally just litter).
DISPLAY_NAMES = {
    "cardboard": "Cardboard",
    "glass": "Glass",
    "metal": "Metal",
    "paper": "Paper",
    "plastic": "Plastic",
    "trash": "Other / General",
}

# --- Image / data ------------------------------------------------------------
IMAGE_SIZE = 224          # standard size for transfer learning backbones
CHANNELS = 3

RANDOM_STATE = 42

# --- Training ------------------------------------------------------------------
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 1e-3

EARLY_STOPPING_PATIENCE = 5
REDUCE_LR_PATIENCE = 3
REDUCE_LR_FACTOR = 0.5