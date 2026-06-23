from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from visionkit.config import ExperimentConfig
from visionkit.reproducibility import seed_worker


class CSVImageDataset(Dataset):
    def __init__(
        self,
        rows: pd.DataFrame,
        image_dir: Path,
        filename_column: str = "filename",
        label_column: Optional[str] = "label",
        class_to_idx: Optional[dict[str, int]] = None,
        transform=None,
        return_metadata: bool = False,
    ) -> None:
        self.rows = rows.reset_index(drop=True).copy()
        self.image_dir = Path(image_dir)
        self.filename_column = filename_column
        self.label_column = label_column
        self.class_to_idx = class_to_idx or {}
        self.transform = transform
        self.return_metadata = return_metadata
        self.samples: list[tuple[Path, Optional[int], dict]] = []

        missing_files = []
        unknown_labels = []
        for record in self.rows.to_dict(orient="records"):
            filename = str(record[filename_column])
            image_path = self.image_dir / filename
            if not image_path.exists():
                missing_files.append(filename)
                continue

            target = None
            if label_column is not None and label_column in record:
                label = str(record[label_column])
                if label not in self.class_to_idx:
                    unknown_labels.append(label)
                    continue
                target = self.class_to_idx[label]
            self.samples.append((image_path, target, record))

        if missing_files:
            preview = ", ".join(missing_files[:5])
            raise FileNotFoundError(f"Images listed in CSV were not found: {preview}")
        if unknown_labels:
            labels = ", ".join(sorted(set(unknown_labels)))
            raise ValueError(f"Labels are missing from class mapping: {labels}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, target, record = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        if self.return_metadata:
            return image, target, record
        return image, target


def read_metadata(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def infer_class_names(df: pd.DataFrame, label_column: str, class_names: Optional[Sequence[str]]) -> list[str]:
    if class_names:
        return [str(name) for name in class_names]
    return sorted(str(label) for label in df[label_column].dropna().unique())


def class_mapping(class_names: Sequence[str]) -> tuple[dict[str, int], list[str]]:
    names = [str(name) for name in class_names]
    return {name: idx for idx, name in enumerate(names)}, names


def filter_split(df: pd.DataFrame, split_column: Optional[str], split_name: Optional[str]) -> pd.DataFrame:
    if not split_column or split_name is None:
        return df.copy()
    if split_column not in df.columns:
        raise KeyError(f"Split column '{split_column}' was not found in CSV.")
    return df[df[split_column].astype(str) == str(split_name)].copy()


def build_transforms(
    image_size: tuple[int, int] = (300, 300),
    train: bool = False,
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: tuple[float, float, float] = (0.229, 0.224, 0.225),
):
    ops = [transforms.Resize(image_size)]
    if train:
        ops.extend(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(10),
            ]
        )
    ops.extend([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    return transforms.Compose(ops)


def build_dataset(
    config: ExperimentConfig,
    split_name: Optional[str],
    transform,
    return_metadata: bool = False,
) -> tuple[CSVImageDataset, list[str]]:
    df = read_metadata(config.csv_path)
    class_names = infer_class_names(df, config.label_column, config.class_names)
    class_to_idx, class_names = class_mapping(class_names)
    rows = filter_split(df, config.split_column, split_name)
    dataset = CSVImageDataset(
        rows=rows,
        image_dir=config.image_dir,
        filename_column=config.filename_column,
        label_column=config.label_column,
        class_to_idx=class_to_idx,
        transform=transform,
        return_metadata=return_metadata,
    )
    return dataset, class_names


def build_dataloaders(config: ExperimentConfig) -> tuple[DataLoader, DataLoader, list[str]]:
    train_transform = build_transforms(
        image_size=config.image_size,
        train=True,
        mean=config.normalize_mean,
        std=config.normalize_std,
    )
    eval_transform = build_transforms(
        image_size=config.image_size,
        train=False,
        mean=config.normalize_mean,
        std=config.normalize_std,
    )
    train_dataset, class_names = build_dataset(config, config.train_split, train_transform)
    val_dataset, _ = build_dataset(config, config.val_split, eval_transform)

    generator = torch.Generator().manual_seed(config.seed)
    num_workers = config.num_workers if config.num_workers is not None else 0
    worker_init = partial(seed_worker, base_seed=config.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=worker_init,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=worker_init,
    )
    return train_loader, val_loader, class_names
