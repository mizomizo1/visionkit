# visionkit

`visionkit` is a small PyTorch package for image-classification projects where
metadata lives in a CSV and images live in a directory. It supports:

- CSV-backed datasets with configurable filename, label, and split columns
- train/validation loops with checkpoints and history CSVs
- prediction CSVs and classification metrics
- Grad-CAM heatmaps and overlay images
- a Python API and a `visionkit` command line interface

## CSV format

At minimum, the CSV needs image filenames and labels:

```csv
filename,label,split
case001.png,F,train
case002.png,N,validation
```

Column names are configurable. If `--classes` is omitted, class names are
inferred from the CSV labels in sorted order.

## CLI examples

Train EfficientNet-B3:

```bash
visionkit train \
  --csv data/meta.csv \
  --image-dir data/images \
  --output-dir outputs/experiment_001 \
  --filename-column filename \
  --label-column label \
  --split-column split \
  --train-split train \
  --val-split validation \
  --classes F N \
  --architecture efficientnet_b3 \
  --epochs 30
```

Evaluate a checkpoint:

```bash
visionkit evaluate \
  --csv data/meta.csv \
  --image-dir data/images \
  --output-dir outputs/experiment_001 \
  --checkpoint outputs/experiment_001/models/best.pt \
  --output-csv outputs/experiment_001/predictions/validation.csv \
  --split-column split \
  --split validation \
  --classes F N
```

Generate Grad-CAM overlays:

```bash
visionkit gradcam \
  --csv data/meta.csv \
  --image-dir data/images \
  --output-dir outputs/experiment_001 \
  --checkpoint outputs/experiment_001/models/best.pt \
  --gradcam-output-dir outputs/experiment_001/gradcam/validation \
  --output-csv outputs/experiment_001/predictions/gradcam_validation.csv \
  --split-column split \
  --split validation \
  --classes F N \
  --target-class N
```

## Python API

```python
from pathlib import Path

from visionkit.config import ExperimentConfig
from visionkit.pipeline import run_training, run_evaluation, run_gradcam

config = ExperimentConfig(
    csv_path=Path("data/meta.csv"),
    image_dir=Path("data/images"),
    output_dir=Path("outputs/experiment_001"),
    class_names=["F", "N"],
    split_column="split",
)

model, history = run_training(config)
metrics = run_evaluation(config, checkpoint_path=config.output_dir / "models" / "best.pt")
run_gradcam(
    config,
    checkpoint_path=config.output_dir / "models" / "best.pt",
    split_name="validation",
    target_class="N",
)
```
# visionkit
