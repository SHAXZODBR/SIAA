"""
================================================================================
  SENTINEL MEDICAL AI — PRE-TRAINED MODEL LOADER
================================================================================
  Uses pre-trained TorchXRayVision models — NO TRAINING NEEDED.

  This gives you a WORKING medical AI immediately, using weights from models
  trained on 500,000+ chest X-rays (NIH + CheXpert + MIMIC + PadChest combined).

  ADVANTAGES over custom training:
  ✅ Works TODAY — no 8-hour Kaggle training
  ✅ Better initial accuracy — trained on 5x more data than NIH alone
  ✅ No data needed — works out of the box
  ✅ Already validated in research papers
  ✅ Can fine-tune later on your clinic data

  MODEL USED: densenet121-res224-all
  - DenseNet121 architecture
  - 224×224 input
  - Trained on ALL datasets (NIH + CheXpert + MIMIC + PadChest + OpenI + Kaggle)
  - Detects 18 pathologies
  - ~30 MB download

  USAGE:
    python -m src.inference.pretrained_model          # Download model
    python run_server.py                              # Server uses it automatically

  REPLACEMENT FLOW:
    Before: 8 hours Kaggle training → best_model.pt → copy to Mac
    After:  5 minutes download → ready to use
================================================================================
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Optional


PATHOLOGY_CLASSES_XRV = [
    'Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax',
    'Edema', 'Emphysema', 'Fibrosis', 'Effusion', 'Pneumonia',
    'Pleural_Thickening', 'Cardiomegaly', 'Nodule', 'Mass', 'Hernia',
    'Lung_Lesion', 'Fracture', 'Lung_Opacity', 'Enlarged_Cardiomediastinum',
]


def download_pretrained_model(
    model_name: str = 'densenet121-res224-all',
    output_dir: str = 'models/densenet',
) -> Path:
    """Download a pre-trained TorchXRayVision model.

    Available models:
    - 'densenet121-res224-all'           → All 18 pathologies (RECOMMENDED)
    - 'densenet121-res224-nih'           → NIH-only (14 pathologies)
    - 'densenet121-res224-chex'          → CheXpert (13 pathologies)
    - 'densenet121-res224-mimic_nb'      → MIMIC-CXR
    - 'densenet121-res224-pc'            → PadChest
    - 'resnet50-res512-all'              → ResNet50, bigger, better accuracy

    Args:
        model_name: Pre-trained model identifier
        output_dir: Where to save the converted model

    Returns:
        Path to saved model file
    """
    try:
        import torchxrayvision as xrv
    except ImportError:
        print("Installing torchxrayvision...")
        os.system(f"{sys.executable} -m pip install torchxrayvision")
        import torchxrayvision as xrv

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / 'best_model.pt'

    print(f"\n[DOWNLOAD] Loading pre-trained: {model_name}")
    print(f"  This downloads once (~30 MB), then cached forever.")

    # Download + load the pre-trained model
    model = xrv.models.DenseNet(weights=model_name)
    model.eval()

    print(f"\n[CONVERT] Converting to Sentinel format...")

    # Save in our format so inference server can load it
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_names': model.pathologies,
        'optimal_thresholds': [0.5] * len(model.pathologies),
        'image_size': 224,
        'model_name': model_name,
        'xrv_weights': model_name,
        'is_pretrained': True,
        'source': 'torchxrayvision',
    }, model_path)

    print(f"\n[SUCCESS] Model saved to {model_path}")
    print(f"  Classes detected: {len(model.pathologies)}")
    for i, cls in enumerate(model.pathologies):
        print(f"    {i+1}. {cls}")

    print(f"\n[NEXT] Start the server:")
    print(f"  python run_server.py")
    print(f"\n[NEXT] Upload a chest X-ray DICOM to test!")

    return model_path


class PretrainedXrayClassifier(nn.Module):
    """Wrapper around TorchXRayVision models for our inference server.

    Automatically uses pre-trained weights if loaded from an XRV checkpoint,
    otherwise falls back to standard DenseNet121 architecture.
    """

    def __init__(self, num_classes: int = 18, xrv_weights: Optional[str] = None):
        super().__init__()

        if xrv_weights:
            import torchxrayvision as xrv
            self.model = xrv.models.DenseNet(weights=xrv_weights)
            self.class_names = self.model.pathologies
            self.is_xrv = True
        else:
            # Fallback to our custom DenseNet
            from torchvision import models
            backbone = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
            num_features = backbone.classifier.in_features
            backbone.classifier = nn.Linear(num_features, num_classes)
            self.model = backbone
            self.class_names = PATHOLOGY_CLASSES_XRV[:num_classes]
            self.is_xrv = False

    def forward(self, x):
        """Forward pass — handles both XRV and custom models."""
        if self.is_xrv:
            # XRV expects grayscale images normalized to [-1024, 1024]
            # Convert RGB to grayscale if needed
            if x.shape[1] == 3:
                x = x.mean(dim=1, keepdim=True)
            # Scale from [0, 1] or [-1, 1] to XRV's expected range
            x = x * 2048 - 1024
            return self.model(x)
        return self.model(x)


def test_inference(dicom_path: Optional[str] = None):
    """Test the pre-trained model on a sample DICOM."""
    import torchxrayvision as xrv
    import numpy as np

    model = xrv.models.DenseNet(weights='densenet121-res224-all')
    model.eval()

    if dicom_path:
        import pydicom
        import cv2

        ds = pydicom.dcmread(dicom_path, force=True)
        img = ds.pixel_array.astype(np.float32)
        # Normalize to [-1024, 1024] (XRV convention)
        img = (img - img.mean()) / (img.std() + 1e-8) * 500
        img = cv2.resize(img, (224, 224))

        # Add batch + channel dims: (1, 1, 224, 224)
        tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float()

        with torch.no_grad():
            output = model(tensor)
            probs = torch.sigmoid(output)[0].numpy()

        print("\n" + "="*60)
        print(f"PREDICTIONS for {Path(dicom_path).name}")
        print("="*60)
        for pathology, prob in zip(model.pathologies, probs):
            bar = '█' * int(prob * 30)
            print(f"  {pathology:25s} {prob:.3f} {bar}")
    else:
        print("No DICOM provided. Model loaded successfully.")
        print(f"Detects: {model.pathologies}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='densenet121-res224-all',
                        help='Pre-trained model name')
    parser.add_argument('--output', default='models/densenet',
                        help='Output directory')
    parser.add_argument('--test', type=str, help='Test DICOM file path')
    args = parser.parse_args()

    if args.test:
        test_inference(args.test)
    else:
        download_pretrained_model(args.model, args.output)
