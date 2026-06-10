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
    p.add_argument("--target", required=True)
    p.add_argument("--representations", default="video_pre,video_out,action_pre,action_out,context_pure,context_with_proprio,proprio_embed")
    p.add_argument("--control_modes", default="rep,time,rep_time,residual")
    p.add_argument("--time_degree", type=int, default=3)
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
        n_val = max(0, min(n - n_train - 1, n_val))
    train_groups = set(uniq[:n_train])
    val_groups = set(uniq[n_train:n_train + n_val])
    test_groups = set(uniq[n_train + n_val:])
    tr = np.array([g in train_groups for g in groups])
    va = np.array([g in val_groups for g in groups])
    te = np.array([g in test_groups for g in groups])
    if not tr.any() or not te.any():
        raise ValueError(
            f"Empty train/test split: episodes={n}, train_samples={int(tr.sum())}, test_samples={int(te.sum())}."
        )
    return tr, va, te


def standardize_from_train(X: np.ndarray, tr: np.ndarray):
    mean = X[tr].mean(axis=0, keepdims=True)
    std = X[tr].std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (X - mean) / std


def make_time_features(metadata: list[dict], degree: int):
    progress = np.array(
        [m.get("timestep", 0) / max(1, m.get("episode_len", 1) - 1) for m in metadata],
        dtype=np.float32,
    )
    degree = max(1, int(degree))
    return np.stack([progress ** d for d in range(1, degree + 1)], axis=1)


def residualize_representation(X_rep: np.ndarray, X_time: np.ndarray, tr: np.ndarray):
    design = np.concatenate([np.ones((X_time.shape[0], 1), dtype=X_time.dtype), X_time], axis=1)
    coef, *_ = np.linalg.lstsq(design[tr], X_rep[tr], rcond=None)
    return X_rep - design @ coef


def prepare_features(X_rep: np.ndarray | None, X_time: np.ndarray, tr: np.ndarray, mode: str):
    X_time_std = standardize_from_train(X_time.astype(np.float32), tr)
    if mode == "time":
        return X_time_std
    if X_rep is None:
        raise ValueError(f"control_mode={mode} requires representation features")
    X_rep = X_rep.astype(np.float32)
    if mode == "rep":
        return standardize_from_train(X_rep, tr)
    if mode == "rep_time":
        return np.concatenate([standardize_from_train(X_rep, tr), X_time_std], axis=1)
    if mode == "residual":
        residual = residualize_representation(X_rep, X_time_std, tr)
        return standardize_from_train(residual.astype(np.float32), tr)
    raise ValueError(f"Unknown control mode: {mode}")


def train_probe(X, y, tr, va, te, probe_type, seed, epochs, batch_size, lr, shuffle_labels=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    y_train = y[tr].copy()
    if shuffle_labels:
        rng = np.random.default_rng(seed)
        rng.shuffle(y_train)
    num_classes = int(y.max()) + 1
    model = make_probe(X.shape[1], num_classes, probe_type)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    train_loader = DataLoader(
        TensorDataset(torch.tensor(X[tr]).float(), torch.tensor(y_train).long()),
        batch_size=batch_size,
        shuffle=True,
    )
    val_x = torch.tensor(X[va]).float()
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
        logits = model(torch.tensor(X[te]).float())
        pred = logits.argmax(dim=1).cpu().numpy()
    return pred


def compute_metrics(y_true: np.ndarray, pred: np.ndarray):
    if len(y_true) == 0:
        return {"acc": float("nan"), "balanced_acc": float("nan"), "macro_f1": float("nan")}
    return {
        "acc": float(accuracy_score(y_true, pred)),
        "balanced_acc": float(balanced_accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
    }


def main():
    args = parse_args()
    payload = torch.load(args.features_path, map_location="cpu")
    rep_names = [r.strip() for r in args.representations.split(",") if r.strip()]
    control_modes = [m.strip() for m in args.control_modes.split(",") if m.strip()]
    y = payload["labels"][args.target].cpu().numpy()
    groups = np.array([m["episode_id"] for m in payload["metadata"]])
    tr, va, te = split_by_episode(groups, args.seed, args.episode_split)
    y_test = y[te]
    X_time = make_time_features(payload["metadata"], args.time_degree)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    time_done = False
    for rep in rep_names:
        if rep not in payload["features"]:
            raise KeyError(f"Representation {rep!r} not found in {args.features_path}")
        X_rep = payload["features"][rep].cpu().numpy()
        for mode in control_modes:
            if mode == "time":
                if time_done:
                    continue
                row_rep = "time"
                X = prepare_features(None, X_time, tr, mode)
                time_done = True
            else:
                row_rep = rep
                X = prepare_features(X_rep, X_time, tr, mode)

            pred = train_probe(X, y, tr, va, te, args.probe_type, args.seed, args.epochs, args.batch_size, args.lr)
            rand_pred = train_probe(
                X,
                y,
                tr,
                va,
                te,
                args.probe_type,
                args.seed,
                args.epochs,
                args.batch_size,
                args.lr,
                shuffle_labels=True,
            )
            metric = compute_metrics(y_test, pred)
            rand_metric = compute_metrics(y_test, rand_pred)
            rows.append({
                "representation": row_rep,
                "target": args.target,
                "control_mode": mode,
                "probe_type": args.probe_type,
                "seed": args.seed,
                "acc": metric["acc"],
                "balanced_acc": metric["balanced_acc"],
                "macro_f1": metric["macro_f1"],
                "random_label_acc": rand_metric["acc"],
                "random_label_balanced_acc": rand_metric["balanced_acc"],
                "random_label_macro_f1": rand_metric["macro_f1"],
                "num_train": int(tr.sum()),
                "num_test": int(te.sum()),
            })
            labels = np.arange(int(y.max()) + 1)
            cm = confusion_matrix(y_test, pred, labels=labels)
            np.save(out_dir / f"{row_rep}_{args.target}_{mode}_confusion.npy", cm)

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
