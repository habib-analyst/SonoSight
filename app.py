"""SonoSight web UI — breast-ultrasound lesion assistant with Grad-CAM.

Upload an ultrasound image (or pick a sample): the app returns
  (a) a classification (benign/normal vs malignant) with confidence,
  (b) a Grad-CAM heatmap overlaid on YOUR uploaded image showing where
      the model looked,
  (c) a plain-language summary.

Research demo — NOT medical advice. See README "Limitations".

Usage:  ./run.sh python app.py   (then open the printed local URL)
"""
import os

import cv2
import gradio as gr
import numpy as np
import torch

from model import build_model
from gradcam import GradCAM

DATA = os.path.join(os.path.dirname(__file__), "data", "breastmnist.npz")
MODEL = os.path.join(os.path.dirname(__file__), "models", "sonocnn.pt")
LABELS = {0: "Benign / Normal", 1: "Malignant"}

_d = np.load(DATA)
TEST_X = _d["test_images"]          # (156,28,28) uint8
TEST_Y = _d["test_labels"].ravel()
MEAN, STD = TEST_X.mean(), TEST_X.std()

_ckpt = torch.load(MODEL, map_location="cpu", weights_only=False)  # own file
_model = build_model()
_model.load_state_dict(_ckpt["state_dict"])
_model.eval()
_gc = GradCAM(_model, _model.features[16])  # last conv block

# sample images: 3 benign + 3 malignant, correctly classified
SAMPLES = []
with torch.no_grad():
    for cls in (0, 1):
        got = 0
        for i in np.where(TEST_Y == cls)[0]:
            xt = torch.from_numpy((((TEST_X[i].astype(np.float32) - MEAN) / STD)).astype(np.float32))[None, None]
            if _model(xt).argmax(1).item() == cls:
                SAMPLES.append((TEST_X[i], f"sample {i} (true {LABELS[cls]})"))
                got += 1
            if got == 3:
                break


def preprocess(img_u8):
    """img_u8: any-size grayscale or RGB -> 28x28 float tensor + original gray."""
    if img_u8.ndim == 3:
        gray = cv2.cvtColor(img_u8, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_u8
    small = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
    t = torch.from_numpy((((small.astype(np.float32) - MEAN) / STD)).astype(np.float32))[None, None]
    return t, gray


def analyze(image, sample_name):
    if image is None and not sample_name:
        return None, None, "Upload an image or pick a sample first."
    if image is not None:
        t, orig_gray = preprocess(image.astype(np.uint8))
        src_note = "your upload"
    else:
        idx = next(i for i, (_, n) in enumerate(SAMPLES) if n == sample_name)
        img28 = SAMPLES[idx][0]
        t = torch.from_numpy((((img28.astype(np.float32) - MEAN) / STD)).astype(np.float32))[None, None]
        orig_gray = cv2.resize(img28, (224, 224), interpolation=cv2.INTER_NEAREST)
        src_note = SAMPLES[idx][1]

    with torch.no_grad():
        probs = torch.softmax(_model(t), 1)[0]
    pred = int(probs.argmax())
    conf = float(probs[pred])
    hm, _ = _gc.heatmap(t, class_idx=pred)
    overlay_bgr = _gc.overlay(orig_gray, hm)
    overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

    if pred == 0:
        verdict = (f"The model leans **{LABELS[0]}** (confidence {conf:.0%}). "
                   f"The heatmap shows the regions it relied on most. "
                   f"This is a research demo on {src_note} — not a medical opinion.")
    else:
        verdict = (f"The model flags this as potentially **{LABELS[1]}** "
                   f"(confidence {conf:.0%}). The highlighted region is what drove "
                   f"that call. This is a research demo on {src_note} — a real "
                   f"screening decision needs a radiologist.")
    return overlay_rgb, f"{LABELS[pred]} — {conf:.1%}", verdict


with gr.Blocks(title="SonoSight") as demo:
    gr.Markdown(
        "# SonoSight — breast-ultrasound lesion assistant\n"
        "Upload an ultrasound image: a small CNN classifies it as **benign/normal** "
        "vs **malignant** and Grad-CAM shows *where it looked*. "
        "Trained on BreastMNIST (breast ultrasound). **Research demo, not medical advice.**"
    )
    with gr.Row():
        with gr.Column():
            img = gr.Image(label="Ultrasound image (upload)", type="numpy")
            sample = gr.Dropdown([n for _, n in SAMPLES], label="…or pick a sample test image")
            btn = gr.Button("Analyze", variant="primary")
        with gr.Column():
            overlay = gr.Image(label="Grad-CAM overlay")
            label = gr.Label(label="Assessment")
            summary = gr.Markdown()
    btn.click(analyze, inputs=[img, sample], outputs=[overlay, label, summary])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
