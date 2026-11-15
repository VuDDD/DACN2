import os
import torch

# Base directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
IDRID_DIR = os.path.join(DATA_DIR, "IDRiD")
DRIVE_DIR = os.path.join(DATA_DIR, "DRIVE")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "ai", "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Device Configuration
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

DEVICE = get_device()

# Image dimensions for training & inference
IMAGE_SIZE = (512, 512) # H, W for high resolution lesion detail preservation

# IDRiD Multi-Lesion Segmentation Setup
IDRID_CLASSES = ["MA", "HE", "EX", "SE", "OD"]
IDRID_CLASS_NAMES = {
    "MA": "Microaneurysms",
    "HE": "Haemorrhages",
    "EX": "Hard Exudates",
    "SE": "Soft Exudates",
    "OD": "Optic Disc"
}
IDRID_NUM_CLASSES = len(IDRID_CLASSES) # 5 channels

# Visual color palette for masks (RGB format)
CLASS_COLORS = {
    "MA": (239, 68, 68),     # Red
    "HE": (249, 115, 22),    # Orange
    "EX": (234, 179, 8),     # Yellow
    "SE": (34, 197, 94),     # Green
    "OD": (59, 130, 246),    # Blue
    "VESSEL": (168, 85, 247) # Purple
}

# DRIVE Blood Vessel Segmentation Setup
DRIVE_CLASSES = ["VESSEL"]
DRIVE_NUM_CLASSES = 1 # Binary segmentation

# Hyperparameters
BATCH_SIZE = 4 if torch.cuda.is_available() else 2
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
NUM_EPOCHS = 20
EARLY_STOPPING_PATIENCE = 7
NUM_WORKERS = 0 # 0 for Windows multiprocessing safety
