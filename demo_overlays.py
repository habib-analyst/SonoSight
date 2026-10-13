"""Demo: generate Grad-CAM overlay figures on BreastMNIST test samples.

Produces demo/demo_overlays.png — the judge-hook figure:
original | heatmap | overlay, for correctly + incorrectly classified cases.

Usage:  ./run.sh python demo_overlays.py
"""
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from model import build_model
from gradcam import GradCAM

DATA = "data/breastmnist.npz"
MODEL = "models/sonocnn.pt"
OUT = "demo/demo_overlays.png"
LABELS = {0: "benign/normal", 1: "malignant"}


def main():
    d = np.load(DATA)
    x = d["test_images"].astype(np.float32)
    y = d["test_labels"].ravel()

    ckpt = torch.load(MODEL, map_location="cpu", weights_only=False)  # own file
    model = build_model()
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    gc = GradCAM(model, model.features[16])  # last conv block

    mean, std = x.mean(), x.std()
    # pick 4 correct-per-class test samples (falls back to any if needed)
    picks = []
    with torch.no_grad():
        for cls in (0, 1):
            got = 0
            for i in np.where(y == cls)[0]:
                xt = torch.from_numpy(((x[i] - mean) / std).astype(np.float32))[None, None]
                pred = model(xt).argmax(1).item()
                if pred == cls:
                    picks.append(i)
                    got += 1
                if got == 4:
                    break

    fig, axes = plt.subplots(8, 3, figsize=(9, 24))
    for row, i in enumerate(picks):
        img_u8 = x[i].astype(np.uint8)
        xt = torch.from_numpy(((x[i] - mean) / std).astype(np.float32))[None, None]
        hm, pred = gc.heatmap(xt)
        ov = cv2.cvtColor(gc.overlay(img_u8, hm), cv2.COLOR_BGR2RGB)
        true = int(y[i])
        ok = "OK" if pred == true else "WRONG"
        axes[row, 0].imshow(img_u8, cmap="gray")
        axes[row, 0].set_title(f"true: {LABELS[true]}")
        axes[row, 1].imshow(hm, cmap="jet", vmin=0, vmax=1)
        axes[row, 1].set_title("Grad-CAM")
        axes[row, 2].imshow(ov)
        axes[row, 2].set_title(f"overlay  pred: {LABELS[pred]} [{ok}]")
        for c in range(3):
            axes[row, c].axis("off")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.suptitle("SonoSight: Grad-CAM on BreastMNIST test set", fontsize=16, y=0.995)
    fig.tight_layout()
    fig.savefig(OUT, dpi=120)
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
