from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


def _epoch_pass(
    model: nn.Module,
    dataloader: DataLoader,
    criterion,
    device: str,
    optimizer: Optional[optim.Optimizer] = None,
    positive_index: int = 1,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    running_loss = 0.0
    total = 0
    correct = 0
    all_labels = []
    all_scores = []

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for images, labels in tqdm(dataloader, desc="train" if is_train else "eval"):
            images = images.to(device)
            labels = labels.to(device)
            if is_train:
                optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            if is_train:
                loss.backward()
                optimizer.step()

            running_loss += loss.item()
            probabilities = F.softmax(outputs, dim=1)
            predicted = probabilities.argmax(dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.detach().cpu().tolist())
            all_scores.extend(probabilities[:, positive_index].detach().cpu().tolist())

    auc = float("nan")
    try:
        auc = float(roc_auc_score(all_labels, all_scores))
    except ValueError:
        pass
    return {
        "loss": running_loss / max(len(dataloader), 1),
        "accuracy": correct / total if total else float("nan"),
        "auc": auc,
    }


def _is_better(value: float, best: Optional[float]) -> bool:
    if best is None:
        return True
    if pd.isna(value):
        return False
    return value > best


def save_checkpoint(model: nn.Module, path: Path, epoch: int, metrics: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "metrics": metrics}, path)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    output_dir: Path,
    epochs: int = 30,
    learning_rate: float = 1e-3,
    device: Optional[str] = None,
    positive_index: int = 1,
    save_every_epoch: bool = True,
    checkpoint_metric: str = "val_auc",
) -> pd.DataFrame:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(output_dir)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    history = []
    best_value: Optional[float] = None

    for epoch in range(1, epochs + 1):
        train_metrics = _epoch_pass(model, train_loader, criterion, device, optimizer, positive_index)
        val_metrics = _epoch_pass(model, val_loader, criterion, device, None, positive_index)
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_auc": train_metrics["auc"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_auc": val_metrics["auc"],
        }
        history.append(row)

        if save_every_epoch:
            save_checkpoint(model, output_dir / "models" / f"epoch_{epoch:03d}.pt", epoch, row)

        metric_value = row.get(checkpoint_metric)
        if metric_value is not None and _is_better(metric_value, best_value):
            best_value = float(metric_value)
            save_checkpoint(model, output_dir / "models" / "best.pt", epoch, row)

        history_df = pd.DataFrame(history)
        (output_dir / "logs").mkdir(parents=True, exist_ok=True)
        history_df.to_csv(output_dir / "logs" / "training_history.csv", index=False)
        print(
            f"Epoch {epoch}/{epochs} "
            f"train_loss={row['train_loss']:.4f} train_auc={row['train_auc']:.4f} "
            f"val_loss={row['val_loss']:.4f} val_auc={row['val_auc']:.4f}"
        )

    return pd.DataFrame(history)
