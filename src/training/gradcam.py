"""Grad-CAM heatmap generation for DenseNet121.

Generates visual explanations showing which regions of the medical image
influenced the model's classification decision. Critical for radiologist trust.
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Optional
from loguru import logger


class GradCAM:
    """Gradient-weighted Class Activation Mapping for DenseNet121.

    Hooks into the last convolutional layer to capture gradients
    and activations, producing a heatmap overlay.
    """

    def __init__(self, model: torch.nn.Module, target_layer_name: str = "features.denseblock4"):
        """
        Args:
            model: DenseNet121 model.
            target_layer_name: Name of the layer to compute Grad-CAM on.
        """
        self.model = model
        self.model.eval()

        self.gradients = None
        self.activations = None

        # Register hooks on target layer
        target_layer = self._find_layer(model, target_layer_name)
        if target_layer is None:
            raise ValueError(f"Layer '{target_layer_name}' not found in model")

        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

        logger.debug(f"Grad-CAM initialized on layer: {target_layer_name}")

    def _find_layer(self, model, layer_name: str):
        """Find a layer by its name in the model."""
        parts = layer_name.split(".")
        current = model
        for part in parts:
            if hasattr(current, part):
                current = getattr(current, part)
            else:
                return None
        return current

    def _save_activation(self, module, input, output):
        """Forward hook to save activations."""
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        """Backward hook to save gradients."""
        self.gradients = grad_output[0].detach()

    @torch.enable_grad()
    def generate(
        self,
        input_image: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """Generate Grad-CAM heatmap for an input image.

        Args:
            input_image: Input tensor of shape (1, 1, H, W).
            target_class: Class index to generate heatmap for.
                          If None, uses the highest-scoring class.

        Returns:
            Heatmap as numpy array of shape (H, W), values in [0, 1].
        """
        self.model.eval()

        # Ensure requires_grad for backward pass
        input_image = input_image.clone().requires_grad_(True)

        # Forward pass
        output = self.model(input_image)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Zero gradients
        self.model.zero_grad()

        # Backward pass for target class
        target_score = output[0, target_class]
        target_score.backward()

        if self.gradients is None or self.activations is None:
            logger.error("Grad-CAM: No gradients/activations captured")
            return np.zeros((input_image.shape[2], input_image.shape[3]))

        # Global average pooling of gradients → weights
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)  # (1, C, 1, 1)

        # Weighted combination of activation maps
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, H', W')

        # ReLU — only keep positive contributions
        cam = F.relu(cam)

        # Resize to input image size
        cam = F.interpolate(
            cam,
            size=(input_image.shape[2], input_image.shape[3]),
            mode="bilinear",
            align_corners=False,
        )

        # Normalize to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam

    def generate_overlay(
        self,
        input_image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.4,
        colormap: int = cv2.COLORMAP_JET,
    ) -> np.ndarray:
        """Overlay Grad-CAM heatmap on original image.

        Args:
            input_image: Original grayscale image (H, W), float32 [0, 1].
            heatmap: Grad-CAM heatmap (H, W), float32 [0, 1].
            alpha: Transparency of heatmap overlay.
            colormap: OpenCV colormap for heatmap visualization.

        Returns:
            RGB overlay image (H, W, 3), uint8.
        """
        # Convert grayscale to RGB
        img_uint8 = (input_image * 255).clip(0, 255).astype(np.uint8)
        img_rgb = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2RGB)

        # Apply colormap to heatmap
        heatmap_uint8 = (heatmap * 255).clip(0, 255).astype(np.uint8)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)

        # Blend
        overlay = cv2.addWeighted(img_rgb, 1 - alpha, heatmap_colored, alpha, 0)
        return overlay
