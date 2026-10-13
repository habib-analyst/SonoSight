# SonoSight — explainable breast-ultrasound lesion assistant

**OpenCV AI Competition 2026 entry (healthcare diagnostics theme).**
Computer vision is core: a trained CNN classifies breast-ultrasound images and
Grad-CAM shows *where* the model looked, overlaid on the image itself.

Upload an ultrasound image (or pick a bundled sample) → get
(a) a benign/normal vs malignant assessment with confidence,
(b) a Grad-CAM heatmap overlay on **your** image,
(c) a plain-language summary.

![Grad-CAM overlays on test samples](demo/demo_overlays.png)

## Model approach (honest)

- **Data:** BreastMNIST — real breast-ultrasound images (Al-Dhabyani et al., via
  MedMNIST v2). 546 train / 78 val / 156 test, 28×28 grayscale, labels
  0 = normal/benign, 1 = malignant. Bundled as `data/breastmnist.npz`.
- **Model:** `SonoCNN` — a small 3-block CNN (32→64→128 channels, batchnorm,
  dropout), trained from scratch on CPU. No pretrained backbone, no tricks.
- **Training:** 40 epochs, AdamW, class-weighted cross-entropy (dataset is
  ~73% malignant), light augmentation (horizontal flip, ±3px translation,
  tiny noise). Trains in ~70 s on CPU.
- **Explainability:** Grad-CAM on the last conv block (7×7 feature map,
  upsampled and blended with OpenCV).

## Measured test-set score

On the held-out BreastMNIST test split (156 images, never seen in training):

| metric | value |
|---|---|
| test accuracy | **87.2%** |
| test AUC | **0.887** |
| majority-class baseline | 73.1% |
| confusion matrix | TN 34, FP 8, FN 12, TP 102 |

Reproduce with `./run.sh python train.py --epochs 40` (prints the metrics JSON).

## <5-minute demo

```bash
cd ~/workspace/earning/hackathons/builds/opencv-healthcv
./run.sh python app.py     # ~10 s startup; open the printed http://127.0.0.1:7860
```

Then upload any ultrasound image or pick a sample from the dropdown and hit
**Analyze**. Or skip the server:

```bash
./run.sh python demo_overlays.py   # regenerates demo/demo_overlays.png
```

Full pipeline from scratch (train + demo figure): ~2 min CPU.

## Limitations (stated plainly)

- 28×28 input is tiny; the model sees a coarse thumbnail, not diagnostic-grade detail.
- BreastMNIST is a curated, small research dataset — results do **not** transfer
  to clinical scans, and the model is not validated for real patients.
- 87% accuracy on 156 test images has wide confidence intervals; false negatives
  exist (12/114 malignant missed in the test set).
- Grad-CAM shows correlation (where activations were high), not causation —
  it is an explanation aid, not proof the model "sees" a lesion.
- Uploaded images are resized to 28×28 for classification; the overlay is
  rendered back onto the original resolution for display.

## Files

- `model.py` — SonoCNN architecture
- `train.py` — training + honest test evaluation (saves `models/sonocnn.pt`)
- `gradcam.py` — Grad-CAM implementation
- `app.py` — Gradio web UI
- `demo_overlays.py` — judge-hook figure generator → `demo/demo_overlays.png`
- `run.sh` — venv + proxy-env wrapper (always use this to run things)
- `requirements.txt`, `BUILD_LOG.md`

By Habib Ur Rehman (GitHub: habib-analyst). Research demo — not medical advice.
