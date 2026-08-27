"""
===================================================================
  SENTINEL MEDICAL AI — GOOGLE COLAB TRAINING SCRIPT
===================================================================
  RUN THIS ON COLAB (100% FREE, T4 GPU)

  HOW TO USE:
  1. Go to https://colab.research.google.com
  2. Click "File" > "New notebook"
  3. Click "Runtime" > "Change runtime type" > GPU (T4)
  4. Upload your data to Google Drive first:
     drive.google.com > New > Folder "sentinel_data"
     Upload nih-chestxray14 and rsna-pneumonia to that folder
  5. Paste this whole script in a cell
  6. Run it
  7. Download best_model.pt when done

  ALTERNATIVELY: Use Kaggle (datasets built-in, faster).
  See kaggle_train.py

  COLAB LIMITATIONS:
  - 12-hour max session (will reconnect)
  - Free T4 GPU (15GB VRAM)
  - Disconnects if idle 90+ min
  - Save checkpoints frequently!
===================================================================
"""

# First cell - Install + mount Drive
SETUP_CELL = """
# Cell 1 — Run this first in Colab:
from google.colab import drive
drive.mount('/content/drive')

!pip install -q pydicom loguru

# Verify GPU
import torch
print('CUDA:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')
"""

# Main training script (same as kaggle but with Colab paths)
import os
import sys
import json
import time
import math
import copy
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.cuda.amp import GradScaler, autocast
import torchvision.models as models
import torchvision.transforms as T
import cv2
from PIL import Image
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# Colab-specific paths — update these to your Google Drive folder
CONFIG = {
    # Colab/Drive paths
    'nih_dir': '/content/drive/MyDrive/sentinel_data/nih-chestxray14',
    'rsna_dir': '/content/drive/MyDrive/sentinel_data/rsna-pneumonia',
    'output_dir': '/content/drive/MyDrive/sentinel_data/output',

    'class_names': [
        'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
        'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',
        'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
        'Pleural_Thickening', 'Hernia',
    ],
    'image_size': 320,
    'batch_size': 32,  # T4 has 15GB
    'epochs': 15,
    'learning_rate': 3e-4,
    'min_lr': 1e-6,
    'weight_decay': 1e-4,
    'warmup_epochs': 2,
    'label_smoothing': 0.05,
    'dropout': 0.3,
    'focal_gamma': 2.0,
    'focal_alpha': 0.75,
    'ema_decay': 0.999,
    'target_recall': 0.85,
    'mixed_precision': True,
    'num_workers': 2,  # Colab has fewer CPUs
    'seed': 42,
    # Checkpoint every N epochs (Colab disconnects often)
    'checkpoint_every': 1,
}


# ============ Copy all the classes from kaggle_train.py ============
# (Same SentinelNet, GeMPool, FocalLoss, EMA, ChestDataset etc.)
# See kaggle_train.py for full implementation.

print("""
============================================================
  For the full Colab script, combine:
  1. The SETUP_CELL above (paste in FIRST cell of Colab)
  2. The entire kaggle_train.py script (paste in SECOND cell)
  3. Just change the paths at the top of CONFIG:
     'nih_dir': '/content/drive/MyDrive/sentinel_data/...'
     'rsna_dir': '/content/drive/MyDrive/sentinel_data/...'

  Kaggle is EASIER because datasets are already there.
  Use Colab only if you need Google Drive integration.
============================================================
""")
