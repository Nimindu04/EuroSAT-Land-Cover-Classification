"""Shared helpers for the EN3150 Assignment 03 notebooks.

Keep this file in the SAME folder as the notebooks.
Run the Model A notebook first: it creates outputs/eurosat_64.npz and outputs/splits.npz,
which every other notebook reuses so that all models see exactly the same 70/15/15 split.
"""
import os, time, json, random, copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix, classification_report,
                             precision_recall_fscore_support, accuracy_score)

SEED = 42
IMG_SIZE = 64
OUT_DIR = Path("./outputs")
OUT_DIR.mkdir(exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(os.cpu_count())

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def set_seed(s=SEED):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


# --------------------------------------------------------------------------- data
class Data:
    """Loads the cached uint8 images + saved split indices and builds the data loaders.

    norm="dataset"  -> per-channel mean/std computed on the TRAIN split (custom models)
    norm="imagenet" -> ImageNet mean/std (pretrained models)
    Images stay uint8 in RAM and are converted/normalised per batch (fast on CPU).
    Augmentation (random horizontal/vertical flips) is applied to training batches only.
    """

    def __init__(self, norm="dataset", batch_size=64, augment=True, cache_name="eurosat_64.npz"):
        cache, splits = OUT_DIR / cache_name, OUT_DIR / "splits.npz"
        if not (cache.exists() and splits.exists()):
            raise FileNotFoundError(
                "Run the Model A notebook first: it creates outputs/%s and outputs/splits.npz" % cache_name)
        d, sp = np.load(cache), np.load(splits)
        self.classes = [str(c) for c in d["classes"]]
        self.num_classes = len(self.classes)
        self.Xt = torch.from_numpy(d["X"]).permute(0, 3, 1, 2).contiguous()   # N,3,H,W uint8
        self.yt = torch.from_numpy(d["y"]).long()
        self.idx = {k: sp[k] for k in ("train", "val", "test")}
        self.augment = augment

        if norm == "imagenet":
            mean, std = torch.tensor(IMAGENET_MEAN), torch.tensor(IMAGENET_STD)
        else:
            s, s2, n = torch.zeros(3), torch.zeros(3), 0
            tr = self.idx["train"]
            for i in range(0, len(tr), 2000):
                b = self.Xt[tr[i:i + 2000]].float().div(255)
                s += b.sum(dim=(0, 2, 3))
                s2 += (b ** 2).sum(dim=(0, 2, 3))
                n += b.shape[0] * b.shape[2] * b.shape[3]
            mean = s / n
            std = (s2 / n - mean ** 2).sqrt()
        self.mean, self.std = mean.view(1, 3, 1, 1), std.view(1, 3, 1, 1)

        def make(k):
            return TensorDataset(self.Xt[self.idx[k]], self.yt[self.idx[k]])

        self.train_loader = DataLoader(make("train"), batch_size=batch_size, shuffle=True, drop_last=True)
        self.val_loader = DataLoader(make("val"), batch_size=256)
        self.test_loader = DataLoader(make("test"), batch_size=256)
        self.probe = self.prep(self.Xt[self.idx["test"][:1]])      # one image for latency tests

    def prep(self, xb, train=False):
        xb = xb.float().div(255)
        if train and self.augment:
            m = torch.rand(xb.size(0)) < 0.5
            xb = torch.where(m[:, None, None, None], xb.flip(3), xb)    # horizontal flip
            m = torch.rand(xb.size(0)) < 0.5
            xb = torch.where(m[:, None, None, None], xb.flip(2), xb)    # vertical flip
        return (xb - self.mean) / self.std


# --------------------------------------------------------------------------- models
class ModelA(nn.Module):
    """Standard CNN: 4 x [Conv3x3 -> ReLU -> MaxPool2] + FC(128) + FC(classes).  914,122 params for 10 classes."""

    def __init__(self, num_classes=10, in_ch=3, img_size=IMG_SIZE):
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2))

        self.features = nn.Sequential(block(in_ch, 32), block(32, 64), block(64, 128), block(128, 256))
        flat = 256 * (img_size // 16) ** 2
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(flat, 128), nn.ReLU(inplace=True),
                                        nn.Dropout(0.3), nn.Linear(128, num_classes))

    def forward(self, x):
        return self.classifier(self.features(x))


class DSConv(nn.Sequential):
    """Depthwise separable convolution: DW 3x3 (groups=cin) -> BN -> ReLU6 -> PW 1x1 -> BN -> ReLU6.
    Index map: [0]=depthwise conv, [1]=BN, [2]=ReLU6, [3]=pointwise conv, [4]=BN, [5]=ReLU6."""

    def __init__(self, cin, cout):
        super().__init__(
            nn.Conv2d(cin, cin, 3, padding=1, groups=cin, bias=False), nn.BatchNorm2d(cin), nn.ReLU6(inplace=True),
            nn.Conv2d(cin, cout, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU6(inplace=True))


class ModelB(nn.Module):
    """Lightweight CNN built only from depthwise separable convolutions. 67,179 params for 10 classes."""

    def __init__(self, num_classes=10, in_ch=3):
        super().__init__()
        self.features = nn.Sequential(
            DSConv(in_ch, 32), nn.MaxPool2d(2),                      # 64 -> 32
            DSConv(32, 64), nn.MaxPool2d(2),                         # 32 -> 16
            DSConv(64, 128), DSConv(128, 128), nn.MaxPool2d(2),      # 16 -> 8
            DSConv(128, 256), nn.MaxPool2d(2))                       # 8  -> 4
        self.classifier = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.2),
                                        nn.Linear(256, num_classes))

    def forward(self, x):
        return self.classifier(self.features(x))


# --------------------------------------------------------------------------- model statistics
def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def summarize(model, input_size=(1, 3, IMG_SIZE, IMG_SIZE), verbose=True):
    """Per-layer output shape, parameters and MACs (conv + linear; BN is foldable so counted as 0 MACs)."""
    rows, hooks = [], []

    def make_hook(name):
        def fn(m, inp, out):
            p = sum(q.numel() for q in m.parameters(recurse=False))
            if isinstance(m, nn.Conv2d):
                macs = (out.numel() // out.shape[0]) * (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1]
            elif isinstance(m, nn.Linear):
                macs = m.in_features * m.out_features
            else:
                macs = 0
            rows.append((name, m.__class__.__name__, tuple(out.shape[1:]), p, macs))
        return fn

    for name, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear, nn.MaxPool2d, nn.BatchNorm2d)):
            hooks.append(m.register_forward_hook(make_hook(name)))
    dev = next(model.parameters()).device
    was_training = model.training
    model.eval()
    with torch.no_grad():
        model(torch.zeros(input_size, device=dev))
    model.train(was_training)
    for h in hooks:
        h.remove()

    total_p = sum(r[3] for r in rows)
    total_m = sum(r[4] for r in rows)
    if verbose:
        print(f"{'Layer':<22}{'Type':<12}{'Output shape':<18}{'Params':>10}{'MACs':>14}")
        print("-" * 76)
        for name, typ, shp, p, macs in rows:
            print(f"{name:<22}{typ:<12}{str(shp):<18}{p:>10,}{macs:>14,}")
        print("-" * 76)
        print(f"{'Total (conv/linear/BN)':<52}{total_p:>10,}{total_m:>14,}")
    return total_p, total_m


def model_size_kb(model, name):
    """Saves the float32 state_dict to outputs/<name>.pt and returns its size on disk in KB."""
    path = OUT_DIR / f"{name}.pt"
    torch.save(model.state_dict(), path)
    return os.path.getsize(path) / 1024


def cpu_latency_ms(model, data, n=200):
    """Average CPU latency for ONE image (batch size 1). Use for relative comparison only."""
    m = copy.deepcopy(model).to("cpu").eval()
    with torch.no_grad():
        for _ in range(20):
            m(data.probe)
        t0 = time.perf_counter()
        for _ in range(n):
            m(data.probe)
    return (time.perf_counter() - t0) / n * 1000


# --------------------------------------------------------------------------- training
class History(dict):
    """dict of per-epoch curves; also remembers the best-validation weights."""
    best_acc = -1.0
    best_state = None


def run_epoch(model, loader, criterion, data, optimizer=None):
    train = optimizer is not None
    model.train(train)
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for xb, yb in loader:
            xb, yb = data.prep(xb, train).to(DEVICE), yb.to(DEVICE)
            out = model(xb)
            loss = criterion(out, yb)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * yb.size(0)
            correct += (out.argmax(1) == yb).sum().item()
            n += yb.size(0)
    return total_loss / n, correct / n


def train_model(model, optimizer, data, epochs=25, name="model", hist=None, verbose=True):
    """Trains for `epochs` epochs. Pass the returned `hist` back in to continue (e.g. 2-phase fine-tuning).
    Keeps the weights of the best VALIDATION accuracy and loads them at the end."""
    criterion = nn.CrossEntropyLoss()
    if hist is None:
        hist = History(train_loss=[], val_loss=[], train_acc=[], val_acc=[], epoch_time=[])
    start = len(hist["train_loss"])
    for ep in range(start + 1, start + epochs + 1):
        t0 = time.time()
        tl, ta = run_epoch(model, data.train_loader, criterion, data, optimizer)
        epoch_time = time.time() - t0                       # training time only
        vl, va = run_epoch(model, data.val_loader, criterion, data)
        for k, v in zip(("train_loss", "val_loss", "train_acc", "val_acc", "epoch_time"),
                        (tl, vl, ta, va, epoch_time)):
            hist[k].append(float(v))
        if va > hist.best_acc:
            hist.best_acc, hist.best_state = va, copy.deepcopy(model.state_dict())
        if verbose:
            print(f"[{name}] epoch {ep:02d} | train loss {tl:.4f} acc {ta:.4f} | "
                  f"val loss {vl:.4f} acc {va:.4f} | {epoch_time:.1f}s")
    model.load_state_dict(hist.best_state)
    print(f"[{name}] best validation accuracy {hist.best_acc:.4f} | "
          f"mean time/epoch (this call) {np.mean(hist['epoch_time'][start:]):.1f}s")
    return hist


# --------------------------------------------------------------------------- evaluation
def predict(model, loader, data):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for xb, yb in loader:
            ps.append(model(data.prep(xb).to(DEVICE)).argmax(1).cpu())
            ys.append(yb)
    return torch.cat(ys).numpy(), torch.cat(ps).numpy()


def evaluate(model, data, verbose=True):
    """Test-set accuracy, macro precision/recall/F1 (+ per-class report when verbose)."""
    y_true, y_pred = predict(model, data.test_loader, data)
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    if verbose:
        print(f"Test accuracy: {acc:.4f}")
        print(f"Macro precision: {p:.4f} | Macro recall: {r:.4f} | Macro F1: {f1:.4f}\n")
        print(classification_report(y_true, y_pred, target_names=data.classes, digits=4, zero_division=0))
    return {"acc": float(acc), "precision": float(p), "recall": float(r), "f1": float(f1),
            "y_true": y_true, "y_pred": y_pred}


# --------------------------------------------------------------------------- plots
def plot_history(hist, title, fname, marks=None):
    """Loss and accuracy curves. marks = epoch numbers after which a dashed line is drawn (phase boundaries)."""
    ep = range(1, len(hist["train_loss"]) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for a, (k1, k2, yl) in zip(ax, [("train_loss", "val_loss", "Cross-entropy loss"),
                                    ("train_acc", "val_acc", "Accuracy")]):
        a.plot(ep, hist[k1], label="train")
        a.plot(ep, hist[k2], label="validation")
        for m in (marks or []):
            a.axvline(m + 0.5, color="gray", ls="--", lw=1)
        a.set_xlabel("Epoch")
        a.set_ylabel(yl)
        a.set_title(f"{title} - {yl.split()[-1].lower()}")
        a.legend()
        a.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / fname, dpi=150)
    plt.show()


def plot_confusion(y_true, y_pred, classes, title, fname):
    k = len(classes)
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(k))
    ax.set_yticks(range(k))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(k):
        for j in range(k):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(OUT_DIR / fname, dpi=150)
    plt.show()
    off = cm.copy()
    np.fill_diagonal(off, 0)
    print("Most confused pairs (true -> predicted : count)")
    for idx in np.argsort(off.ravel())[::-1][:5]:
        i, j = divmod(idx, k)
        if off[i, j] > 0:
            print(f"  {classes[i]} -> {classes[j]} : {off[i, j]}")
    return cm


def plot_compare(hists, key, ylabel, title, fname, logy=False):
    """Overlay one curve (e.g. val_loss) from several runs."""
    plt.figure(figsize=(7.5, 4.5))
    for name, h in hists.items():
        plt.plot(range(1, len(h[key]) + 1), h[key], label=name)
    if logy:
        plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT_DIR / fname, dpi=150)
    plt.show()


# --------------------------------------------------------------------------- results for the final table
def save_results(tag, label, model, hist, ev, optimizer_desc, data, time_per_epoch_s=None, extra=None):
    """Computes params / MACs / size / latency, saves weights and outputs/<tag>_results.json."""
    total, trainable = count_params(model)
    _, macs = summarize(model, verbose=False)
    size_kb = model_size_kb(model, tag)
    lat = cpu_latency_ms(model, data)
    if time_per_epoch_s is None:
        time_per_epoch_s = float(np.mean(hist["epoch_time"]))
    res = {
        "model": label, "optimizer": optimizer_desc, "epochs": len(hist["train_loss"]),
        "params": int(total), "trainable_params": int(trainable), "macs": int(macs),
        "size_kb": round(size_kb, 1), "time_per_epoch_s": round(float(time_per_epoch_s), 2),
        "latency_ms_cpu": round(lat, 2), "best_val_acc": float(max(hist["val_acc"])),
        "test_acc": ev["acc"], "macro_precision": ev["precision"], "macro_recall": ev["recall"],
        "macro_f1": ev["f1"], "history": dict(hist),
    }
    if extra:
        res.update(extra)
    with open(OUT_DIR / f"{tag}_results.json", "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps({k: v for k, v in res.items() if k != "history"}, indent=2))
    return res
