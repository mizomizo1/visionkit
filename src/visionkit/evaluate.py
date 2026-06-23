from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from visionkit.data import build_dataset, build_transforms
from visionkit.metrics import classification_metrics


def _record_at(records, index: int) -> dict:
    if not isinstance(records, dict):
        return dict(records[index])
    row = {}
    for key, value in records.items():
        item = value[index]
        if hasattr(item, "item"):
            try:
                item = item.item()
            except ValueError:
                pass
        row[key] = item
    return row


def predict(
    model: torch.nn.Module,
    dataloader: DataLoader,
    class_names: list[str],
    device: Optional[str] = None,
    positive_index: Optional[int] = None,
) -> pd.DataFrame:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    positive_index = positive_index if positive_index is not None else min(1, len(class_names) - 1)
    model.to(device)
    model.eval()
    rows = []

    with torch.no_grad():
        for images, labels, records in tqdm(dataloader, desc="predict"):
            images = images.to(device)
            outputs = model(images)
            probabilities = F.softmax(outputs, dim=1)
            predicted = probabilities.argmax(dim=1).cpu().tolist()
            scores = probabilities[:, positive_index].cpu().tolist()

            for idx in range(images.size(0)):
                record = _record_at(records, idx)
                true_label_idx = labels[idx].item() if labels[idx] is not None else None
                row = dict(record)
                row.update(
                    {
                        "true_label_num": true_label_idx,
                        "true_label": class_names[true_label_idx] if true_label_idx is not None else None,
                        "predicted_class_num": predicted[idx],
                        "predicted_class": class_names[predicted[idx]],
                        "positive_class_probability": scores[idx],
                    }
                )
                for class_idx, class_name in enumerate(class_names):
                    row[f"probability_{class_name}"] = probabilities[idx, class_idx].item()
                rows.append(row)
    return pd.DataFrame(rows)


def evaluate_dataframe(
    predictions: pd.DataFrame,
    positive_index: int = 1,
    n_bootstraps: int = 1000,
    seed: int = 123,
) -> dict[str, float]:
    if predictions.empty or predictions["true_label_num"].nunique() < 2:
        return {}
    return classification_metrics(
        predictions["true_label_num"].astype(int),
        predictions["predicted_class_num"].astype(int),
        predictions["positive_class_probability"],
        positive_index=positive_index,
        n_bootstraps=n_bootstraps,
        seed=seed,
    )


def evaluate_model(
    model: torch.nn.Module,
    config,
    split_name: Optional[str],
    class_names: list[str],
    output_csv: Path,
    positive_index: int = 1,
) -> tuple[pd.DataFrame, dict[str, float]]:
    transform = build_transforms(
        image_size=config.image_size,
        train=False,
        mean=config.normalize_mean,
        std=config.normalize_std,
    )
    dataset, _ = build_dataset(config, split_name, transform, return_metadata=True)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
    predictions = predict(model, loader, class_names, device=config.device, positive_index=positive_index)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_csv, index=False)
    metrics = evaluate_dataframe(predictions, positive_index=positive_index, seed=config.seed)
    if metrics:
        pd.DataFrame([metrics]).to_csv(output_csv.with_name(output_csv.stem + "_metrics.csv"), index=False)
    return predictions, metrics
