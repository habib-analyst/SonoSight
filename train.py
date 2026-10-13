"""Train SonoCNN on BreastMNIST (breast ultrasound).

Real training, real metrics. Reports test accuracy + AUC on the held-out
BreastMNIST test split (156 images). Honest baseline: ~73% majority class.

Usage:  ./run.sh python train.py [--epochs 30] [--out models/sonocnn.pt]
"""
import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset

from model import build_model

MEAN, STD = 0.3804, 0.2688  # computed on train split; recomputed below anyway


def load_split(path, split):
    d = np.load(path)
    x = d[f"{split}_images"].astype(np.float32) / 255.0
    y = d[f"{split}_labels"].ravel().astype(np.int64)
    return x, y


def augment(x, rng):
    """Light augmentation suitable for ultrasound: horizontal flip, tiny
    translation (breaks positional bias — BreastMNIST has a dark border
    the model otherwise latches onto), tiny noise."""
    out = x.copy()
    flip = rng.random(len(out)) < 0.5
    out[flip] = out[flip][:, :, ::-1]
    # random integer shift ±3 px, edge-replicated
    padded = np.pad(out, ((0, 0), (3, 3), (3, 3)), mode="edge")
    dy = rng.integers(0, 7, len(out))
    dx = rng.integers(0, 7, len(out))
    shifted = np.stack([padded[i, dy[i]:dy[i] + 28, dx[i]:dx[i] + 28]
                        for i in range(len(out))])
    return np.clip(shifted + rng.normal(0, 0.01, shifted.shape), 0, 1).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--out", default="models/sonocnn.pt")
    ap.add_argument("--data", default="data/breastmnist.npz")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    x_train, y_train = load_split(args.data, "train")
    x_val, y_val = load_split(args.data, "val")
    x_test, y_test = load_split(args.data, "test")
    print(f"train {x_train.shape}  val {x_val.shape}  test {x_test.shape}", flush=True)

    # class-weighted loss for the 73/27 imbalance
    counts = np.bincount(y_train, minlength=2).astype(np.float32)
    weights = torch.tensor(len(y_train) / (2 * counts))
    print(f"class counts {counts.tolist()} -> loss weights {weights.tolist()}", flush=True)

    def norm(x):
        return ((x - x_train.mean()) / x_train.std()).astype(np.float32)

    train_ds = TensorDataset(
        torch.from_numpy(norm(augment(x_train, rng))[:, None]),
        torch.from_numpy(y_train),
    )
    val_ds = TensorDataset(torch.from_numpy(norm(x_val)[:, None]), torch.from_numpy(y_val))
    test_ds = TensorDataset(torch.from_numpy(norm(x_test)[:, None]), torch.from_numpy(y_test))
    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=64)
    test_dl = DataLoader(test_ds, batch_size=64)

    device = torch.device("cpu")
    model = build_model().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = nn.CrossEntropyLoss(weight=weights)

    best_val, best_state, t0 = 0.0, None, time.time()
    for epoch in range(args.epochs):
        model.train()
        tot, n = 0.0, 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(xb)
            n += len(xb)
        sched.step()
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for xb, yb in val_dl:
                preds += model(xb.to(device)).argmax(1).cpu().tolist()
                trues += yb.tolist()
        acc = accuracy_score(trues, preds)
        if acc > best_val:
            best_val, best_state = acc, {k: v.cpu() for k, v in model.state_dict().items()}
        print(f"epoch {epoch+1:2d}/{args.epochs}  train_loss {tot/n:.4f}  val_acc {acc:.4f}", flush=True)

    model.load_state_dict(best_state)
    model.eval()
    probs, preds, trues = [], [], []
    with torch.no_grad():
        for xb, yb in test_dl:
            out = torch.softmax(model(xb.to(device)), 1)
            probs += out[:, 1].cpu().tolist()
            preds += out.argmax(1).cpu().tolist()
            trues += yb.tolist()
    acc = accuracy_score(trues, preds)
    auc = roc_auc_score(trues, probs)
    cm = confusion_matrix(trues, preds).tolist()
    metrics = {
        "test_accuracy": round(acc, 4),
        "test_auc": round(auc, 4),
        "confusion_matrix": cm,  # [[TN, FP], [FN, TP]]
        "majority_baseline": round(max(np.bincount(y_test)) / len(y_test), 4),
        "epochs": args.epochs,
        "best_val_acc": round(best_val, 4),
        "train_seconds": round(time.time() - t0, 1),
    }
    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({"state_dict": best_state, "metrics": metrics}, args.out)
    with open(args.out.replace(".pt", "_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2), flush=True)
    print(f"saved -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
