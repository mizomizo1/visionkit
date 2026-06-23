from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from visionkit.config import ExperimentConfig
from visionkit.data import build_dataloaders, infer_class_names, read_metadata
from visionkit.evaluate import evaluate_model
from visionkit.gradcam import generate_gradcam_report
from visionkit.models import build_model, load_checkpoint
from visionkit.reproducibility import set_seed
from visionkit.train import train_model


def _class_names(config: ExperimentConfig) -> list[str]:
    return infer_class_names(read_metadata(config.csv_path), config.label_column, config.class_names)


def _positive_index(config: ExperimentConfig, class_names: list[str]) -> int:
    if config.positive_class is not None:
        return class_names.index(config.positive_class)
    return min(1, len(class_names) - 1)


def run_training(config: ExperimentConfig):
    config.ensure_output_dirs()
    set_seed(config.seed)
    train_loader, val_loader, class_names = build_dataloaders(config)
    model = build_model(config.architecture, len(class_names), config.weights, config.device)
    history = train_model(
        model,
        train_loader,
        val_loader,
        config.output_dir,
        epochs=config.epochs,
        learning_rate=config.learning_rate,
        device=config.device,
        positive_index=_positive_index(config, class_names),
        save_every_epoch=config.save_every_epoch,
        checkpoint_metric=config.checkpoint_metric,
    )
    return model, history


def run_evaluation(
    config: ExperimentConfig,
    checkpoint_path: Path,
    split_name: Optional[str] = None,
    output_csv: Optional[Path] = None,
):
    config.ensure_output_dirs()
    class_names = _class_names(config)
    model = build_model(config.architecture, len(class_names), config.weights, config.device)
    load_checkpoint(model, checkpoint_path, config.device)
    split_name = split_name or config.val_split
    output_csv = output_csv or config.prediction_dir() / f"{split_name}_predictions.csv"
    return evaluate_model(
        model,
        config,
        split_name=split_name,
        class_names=class_names,
        output_csv=output_csv,
        positive_index=_positive_index(config, class_names),
    )


def run_gradcam(
    config: ExperimentConfig,
    checkpoint_path: Path,
    split_name: Optional[str] = None,
    output_dir: Optional[Path] = None,
    output_csv: Optional[Path] = None,
    target_class: Optional[Union[str, int]] = None,
):
    config.ensure_output_dirs()
    class_names = _class_names(config)
    model = build_model(config.architecture, len(class_names), config.weights, config.device)
    load_checkpoint(model, checkpoint_path, config.device)
    split_name = split_name or config.val_split
    output_dir = output_dir or config.gradcam_dir() / str(split_name)
    output_csv = output_csv or config.prediction_dir() / f"gradcam_{split_name}.csv"
    return generate_gradcam_report(
        model=model,
        csv_path=config.csv_path,
        image_dir=config.image_dir,
        output_dir=output_dir,
        output_csv=output_csv,
        filename_column=config.filename_column,
        label_column=config.label_column,
        split_column=config.split_column,
        split_name=split_name,
        class_names=class_names,
        target_class=target_class or config.positive_class,
        target_layer=config.target_layer,
        image_size=config.image_size,
        mean=config.normalize_mean,
        std=config.normalize_std,
        device=config.device,
    )
