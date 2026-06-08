import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--features_path", required=True)
    p.add_argument("--target", required=True, choices=["effect", "stage", "target"])
    p.add_argument("--representations", default="video_pre,video_out,action_pre,action_out,context,proprio_embed")
    p.add_argument("--probe_type", default="linear", choices=["linear", "mlp"])
    p.add_argument("--episode_split", default="0.7,0.15,0.15")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--output_dir", required=True)
    return p.parse_args()


def make_probe(d: int, c: int, probe_type: str):
    if probe_type == "linear":
        return nn.Linear(d, c)
    hidden = max(32, d // 2)
    return nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, c))


def split_by_episode(groups: np.ndarray, seed: int, split: str):
    ratios = np.array([float(x) for x in split.split(",")], dtype=np.float64)
    ratios = ratios / ratios.sum()
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n = len(uniq)
    n_train = max(1, int(round(n * ratios[0])))
    n_val = max(1, int(round(n * ratios[1])))
    if n_train + n_val >= n:
        n_val = max(1, min(n - n_train - 1, n_val))
    train_groups = set(uniq[:n_train])
    val_groups = set(uniq[n_train:n_train + n_val])
    test_groups = set(uniq[n_train + n_val:])
    tr = np.array([g in train_groups for g in groups])
    va = np.array([g in val_groups for g in groups])
    te = np.array([g in test_groups for g in groups])
    return tr, va, te


def train_probe(X, y, groups, probe_type, seed, epochs, batch_size, lr, split, shuffle_labels=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    tr, va, te = split_by_episode(groups, seed, split)
    y_train = y[tr].copy()
    if shuffle_labels:
        rng = np.random.default_rng(seed)
        rng.shuffle(y_train)
    num_classes = int(y.max()) + 1
    model = make_probe(X.shape[1], num_classes, probe_type)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    train_loader = DataLoader(TensorDataset(torch.tensor(X[tr]).float(), torch.tensor(y_train).long()), batch_size=batch_size, shuffle=True)
    val_x = torch.tensor(X[va]).float()
    val_y = torch.tensor(y[va]).long()
    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
        if len(val_x) > 0:
            model.eval()
            with torch.no_grad():
                _ = model(val_x)
    model.eval()
    with torch.no_grad():
        test_x = torch.tensor(X[te]).float()
        logits = model(test_x)
        pred = logits.argmax(dim=1).cpu().numpy()
    return pred, te, tr, va


def time_baseline(features, y, groups, seed, split):
    progress = np.array([m.get("timestep", 0) / max(1, m.get("episode_len", 1) - 1) for m in features["metadata"]], dtype=np.float32)
    X = progress[:, None]
    pred, te, _, _ = train_probe(X, y, groups, "linear", seed, 20, 64, 1e-2, split, shuffle_labels=False)
    return pred, te


def main():
    args = parse_args()
    payload = torch.load(args.features_path, map_location="cpu")
    rep_names = [r.strip() for r in args.representations.split(",") if r.strip()]
    y = payload["labels"][args.target].cpu().numpy()
    groups = np.array([m["episode_id"] for m in payload["metadata"]])
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for rep in rep_names:
        X = payload["features"][rep].cpu().numpy()
        pred, te, tr, _ = train_probe(X, y, groups, args.probe_type, args.seed, args.epochs, args.batch_size, args.lr, args.episode_split)
        y_test = y[te]

        rand_pred, _, _, _ = train_probe(X, y, groups, args.probe_type, args.seed, args.epochs, args.batch_size, args.lr, args.episode_split, shuffle_labels=True)
        time_pred, time_te = time_baseline(payload, y, groups, args.seed, args.episode_split)

        rows.append({
            "representation": rep,
            "target": args.target,
            "probe_type": args.probe_type,
            "acc": float(accuracy_score(y_test, pred)) if len(y_test) else float("nan"),
            "balanced_acc": float(balanced_accuracy_score(y_test, pred)) if len(y_test) else float("nan"),
            "macro_f1": float(f1_score(y_test, pred, average="macro")) if len(y_test) else float("nan"),
            "random_label_acc": float(accuracy_score(y_test, rand_pred)) if len(y_test) else float("nan"),
            "time_baseline_acc": float(accuracy_score(y[y_t := time_te], time_pred)) if len(time_te) else float("nan"),
            "num_train": int(tr.sum()),
            "num_test": int(te.sum()),
        })
        cm = confusion_matrix(y_test, pred) if len(y_test) else np.zeros((1, 1), dtype=int)
        np.save(out_dir / f"{rep}_{args.target}_confusion.npy", cm)

    csv_path = out_dir / f"probe_{args.target}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with open(out_dir / f"probe_{args.target}.json", "w") as f:
        json.dump(rows, f, indent=2)
    print(csv_path)


if __name__ == "__main__":
    main()
