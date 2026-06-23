from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from visionkit.config import ExperimentConfig
from visionkit.pipeline import run_evaluation, run_gradcam, run_training


def _image_size(value: str) -> tuple[int, int]:
    parts = value.lower().replace("x", ",").split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("image size must look like 300x300")
    return int(parts[0]), int(parts[1])


def _base_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--csv", dest="csv_path", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--filename-column", default="filename")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--split-column", default="split")
    parser.add_argument("--classes", nargs="*")
    parser.add_argument("--architecture", default="efficientnet_b3")
    parser.add_argument("--weights", default="DEFAULT")
    parser.add_argument("--image-size", type=_image_size, default=(300, 300))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--positive-class")
    parser.add_argument("--target-layer")
    parser.add_argument("--device")


def _config(args) -> ExperimentConfig:
    return ExperimentConfig(
        csv_path=args.csv_path,
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        filename_column=args.filename_column,
        label_column=args.label_column,
        split_column=args.split_column,
        train_split=getattr(args, "train_split", "train"),
        val_split=getattr(args, "val_split", "validation"),
        class_names=args.classes,
        architecture=args.architecture,
        weights=args.weights,
        image_size=args.image_size,
        batch_size=args.batch_size,
        epochs=getattr(args, "epochs", 30),
        learning_rate=getattr(args, "learning_rate", 1e-3),
        seed=args.seed,
        positive_class=args.positive_class,
        target_layer=args.target_layer,
        device=args.device,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="visionkit")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train")
    _base_parser(train)
    train.add_argument("--train-split", default="train")
    train.add_argument("--val-split", default="validation")
    train.add_argument("--epochs", type=int, default=30)
    train.add_argument("--learning-rate", type=float, default=1e-3)

    evaluate = subparsers.add_parser("evaluate")
    _base_parser(evaluate)
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--split", default="validation")
    evaluate.add_argument("--output-csv", type=Path)

    gradcam = subparsers.add_parser("gradcam")
    _base_parser(gradcam)
    gradcam.add_argument("--checkpoint", type=Path, required=True)
    gradcam.add_argument("--split", default="validation")
    gradcam.add_argument("--gradcam-output-dir", type=Path)
    gradcam.add_argument("--output-csv", type=Path)
    gradcam.add_argument("--target-class")
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = _config(args)

    if args.command == "train":
        _, history = run_training(config)
        print(history.tail(1).to_string(index=False))
    elif args.command == "evaluate":
        _, metrics = run_evaluation(config, args.checkpoint, split_name=args.split, output_csv=args.output_csv)
        print(metrics)
    elif args.command == "gradcam":
        _, metrics = run_gradcam(
            config,
            args.checkpoint,
            split_name=args.split,
            output_dir=args.gradcam_output_dir,
            output_csv=args.output_csv,
            target_class=args.target_class,
        )
        print(metrics)


if __name__ == "__main__":
    main()
