"""Grad-CAM explainability for SonoCNN.

Implements Grad-CAM (Selvaraju et al., 2017) on the last convolutional
block so the app can show WHERE the model looked on the ultrasound image.
"""
import cv2
import numpy as np
import torch


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model.eval()
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    @torch.no_grad()
    def _predict(self, x):
        return self.model(x)

    def heatmap(self, x, class_idx=None):
        """x: (1,1,28,28) tensor. Returns 28x28 float heatmap in [0,1]."""
        self.model.zero_grad()
        with torch.enable_grad():
            out = self.model(x)
            if class_idx is None:
                class_idx = out.argmax(dim=1).item()
            score = out[0, class_idx]
            score.backward()
        acts = self.activations[0]          # (C,H,W)
        grads = self.gradients[0]          # (C,H,W)
        weights = grads.mean(dim=(1, 2))   # (C,)
        cam = (weights[:, None, None] * acts).sum(dim=0)
        cam = torch.relu(cam)
        cam = cam.cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()
        cam = cv2.resize(cam, (28, 28), interpolation=cv2.INTER_LINEAR)
        return cam.astype(np.float32), int(class_idx)

    def overlay(self, gray_u8, heatmap, alpha=0.45, colormap=cv2.COLORMAP_JET):
        """gray_u8: HxW uint8 grayscale. Returns HxWx3 BGR overlay."""
        h, w = gray_u8.shape
        hm = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)
        color = cv2.applyColorMap((hm * 255).astype(np.uint8), colormap)
        base = cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR)
        return cv2.addWeighted(color, alpha, base, 1 - alpha, 0)
