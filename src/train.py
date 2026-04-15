"""
train.py

Phase 1 training: MLP and LightGBM baselines on CICIDS2017 flow features.
Prints F1/accuracy benchmarks that PacketFormer (Phase 2) needs to beat.

Usage:
    python src/train.py --model mlp
    python src/train.py --model lgbm
    python src/train.py --model both
"""

import os
import argparse
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.metrics import classification_report, f1_score, accuracy_score
import lightgbm as lgb

from src.dataset import get_dataloaders, load_processed
from src.model import MLPClassifier

CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# MLP Training
# ─────────────────────────────────────────────

def train_mlp(
    num_epochs: int = 20,
    batch_size: int = 512,
    lr: float = 1e-3,
    device: str = None,
) -> dict:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training MLP on {device}")

    train_loader, val_loader, test_loader, num_features, num_classes = get_dataloaders(
        batch_size=batch_size
    )

    model = MLPClassifier(num_features=num_features, num_classes=num_classes).to(device)
    print(f"MLP parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=lr,
        steps_per_epoch=len(train_loader),
        epochs=num_epochs,
    )

    best_val_f1 = 0.0

    for epoch in range(1, num_epochs + 1):
        # Train
        model.train()
        total_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        # Validate
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                preds = model(X_batch).argmax(dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_labels.extend(y_batch.numpy())

        val_f1 = f1_score(val_labels, val_preds, average="macro")
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch:02d}/{num_epochs} | loss: {avg_loss:.4f} | val F1 (macro): {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "mlp_best.pt"))

    # Test evaluation
    model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "mlp_best.pt")))
    model.eval()
    test_preds, test_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            preds = model(X_batch).argmax(dim=1).cpu().numpy()
            test_preds.extend(preds)
            test_labels.extend(y_batch.numpy())

    acc = accuracy_score(test_labels, test_preds)
    f1 = f1_score(test_labels, test_preds, average="macro")
    print("\n── MLP Test Results ──")
    print(f"Accuracy: {acc:.4f} | F1 (macro): {f1:.4f}")
    print(classification_report(test_labels, test_preds))

    return {"model": "MLP", "accuracy": acc, "f1_macro": f1}


# ─────────────────────────────────────────────
# LightGBM Training
# ─────────────────────────────────────────────

def train_lgbm() -> dict:
    print("Training LightGBM baseline...")
    X, y = load_processed()

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    clf = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=63,
        n_jobs=-1,
        verbose=-1,
    )
    clf.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="macro")

    print("\n── LightGBM Test Results ──")
    print(f"Accuracy: {acc:.4f} | F1 (macro): {f1:.4f}")
    print(classification_report(y_test, preds))

    with open(os.path.join(CHECKPOINT_DIR, "lgbm_model.pkl"), "wb") as f:
        pickle.dump(clf, f)

    return {"model": "LightGBM", "accuracy": acc, "f1_macro": f1}


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["mlp", "lgbm", "both"],
        default="both",
        help="Which model to train (default: both)",
    )
    args = parser.parse_args()

    results = []
    if args.model in ("mlp", "both"):
        results.append(train_mlp())
    if args.model in ("lgbm", "both"):
        results.append(train_lgbm())

    print("\n── Phase 1 Benchmark Summary ──")
    for r in results:
        print(f"{r['model']:12s} | Accuracy: {r['accuracy']:.4f} | F1 (macro): {r['f1_macro']:.4f}")
    print("\nThese are the baselines PacketFormer (Phase 2) targets to beat.")


if __name__ == "__main__":
    main()
