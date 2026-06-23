from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence


@dataclass
class ExperimentConfig:
    csv_path: Path
    image_dir: Path
    output_dir: Path
    filename_column: str = "filename"
    label_column: str = "label"
    split_column: Optional[str] = "split"
    train_split: str = "train"
    val_split: str = "validation"
    class_names: Optional[Sequence[str]] = None
    architecture: str = "efficientnet_b3"
    weights: Optional[str] = "DEFAULT"
    image_size: tuple[int, int] = (300, 300)
    batch_size: int = 8
    epochs: int = 30
    learning_rate: float = 1e-3
    num_workers: Optional[int] = None
    seed: int = 123
    positive_class: Optional[str] = None
    target_layer: Optional[str] = None
    device: Optional[str] = None
    checkpoint_metric: str = "val_auc"
    save_every_epoch: bool = True
    normalize_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    normalize_std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    extra_metadata_columns: Sequence[str] = field(default_factory=tuple)

    def model_dir(self) -> Path:
        return self.output_dir / "models"

    def log_dir(self) -> Path:
        return self.output_dir / "logs"

    def prediction_dir(self) -> Path:
        return self.output_dir / "predictions"

    def gradcam_dir(self) -> Path:
        return self.output_dir / "gradcam"

    def ensure_output_dirs(self) -> None:
        for path in [
            self.output_dir,
            self.model_dir(),
            self.log_dir(),
            self.prediction_dir(),
            self.gradcam_dir(),
        ]:
            path.mkdir(parents=True, exist_ok=True)
