from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm

from visionkit.data import build_transforms, class_mapping, filter_split, infer_class_names, read_metadata
from visionkit.evaluate import evaluate_dataframe
from visionkit.models import get_module_by_path


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.handles = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        def forward_hook(module, inputs, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.handles.append(self.target_layer.register_full_backward_hook(backward_hook))

    def generate(self, input_tensor: torch.Tensor, target_class: Optional[int] = None) -> np.ndarray:
        output = self.model(input_tensor)
        if target_class is None:
            target_class = int(output.argmax(dim=1).item())
        self.model.zero_grad()
        output[0, target_class].backward()

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Grad-CAM hooks did not capture gradients or activations.")

        gradients = self.gradients[0]
        activations = self.activations[0]
        weights = gradients.mean(dim=(1, 2), keepdim=True)
        cam = torch.sum(weights * activations, dim=0)
        cam = F.relu(cam)
        cam = F.interpolate(
            cam.unsqueeze(0).unsqueeze(0),
            size=input_tensor.shape[2:],
            mode="bilinear",
            align_corners=False,
        )
        cam_np = cam.squeeze().detach().cpu().numpy()
        denominator = cam_np.max() - cam_np.min()
        if denominator <= 1e-12:
            return np.zeros_like(cam_np)
        return (cam_np - cam_np.min()) / denominator

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def _denormalize(image_tensor: torch.Tensor, mean: Sequence[float], std: Sequence[float]) -> np.ndarray:
    image = image_tensor.detach().cpu().clone()
    for channel, (mean_value, std_value) in enumerate(zip(mean, std)):
        image[channel] = image[channel] * std_value + mean_value
    image = image.clamp(0, 1).permute(1, 2, 0).numpy()
    return image


def save_gradcam_figure(
    original_image: np.ndarray,
    cam: np.ndarray,
    save_path: Path,
    title: str,
    alpha: float = 0.4,
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(original_image)
    axes[0].set_title(title)
    axes[1].imshow(cam, cmap="jet")
    axes[1].set_title("Grad-CAM")
    axes[2].imshow(original_image)
    axes[2].imshow(cam, cmap="jet", alpha=alpha)
    axes[2].set_title("Overlay")
    for axis in axes:
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def generate_gradcam_report(
    model: torch.nn.Module,
    csv_path: Path,
    image_dir: Path,
    output_dir: Path,
    output_csv: Path,
    filename_column: str = "filename",
    label_column: str = "label",
    split_column: Optional[str] = "split",
    split_name: Optional[str] = None,
    class_names: Optional[Sequence[str]] = None,
    target_class: Optional[Union[str, int]] = None,
    target_layer: Optional[str] = None,
    image_size: tuple[int, int] = (300, 300),
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    device: Optional[str] = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    df = filter_split(read_metadata(csv_path), split_column, split_name)
    class_names = infer_class_names(df, label_column, class_names)
    class_to_idx, class_names = class_mapping(class_names)
    positive_index = min(1, len(class_names) - 1)
    if target_class is None:
        target_index = positive_index
    elif isinstance(target_class, int):
        target_index = target_class
    else:
        target_index = class_to_idx[str(target_class)]

    transform = build_transforms(image_size=image_size, train=False, mean=mean, std=std)
    target_module = get_module_by_path(model, target_layer)
    gradcam = GradCAM(model, target_module)
    output_dir = Path(output_dir)
    output_csv = Path(output_csv)
    results = []

    try:
        for record in tqdm(df.to_dict(orient="records"), desc="gradcam"):
            filename = str(record[filename_column])
            image_path = Path(image_dir) / filename
            if not image_path.exists():
                continue
            image = Image.open(image_path).convert("RGB")
            image_tensor = transform(image).unsqueeze(0).to(device)
            cam = gradcam.generate(image_tensor, target_class=target_index)

            with torch.no_grad():
                outputs = model(image_tensor)
                probabilities = F.softmax(outputs, dim=1)[0].detach().cpu()
                predicted_index = int(probabilities.argmax().item())

            true_label = str(record[label_column]) if label_column in record else None
            true_index = class_to_idx[true_label] if true_label in class_to_idx else None
            original = _denormalize(image_tensor[0], mean, std)
            pred_label = class_names[predicted_index]
            save_name = f"{Path(filename).stem}__pred-{pred_label}.png"
            save_gradcam_figure(
                original,
                cam,
                output_dir / save_name,
                title=f"{filename}\ntrue={true_label} pred={pred_label}",
            )

            result = dict(record)
            result.update(
                {
                    "gradcam_file": save_name,
                    "target_class_num": target_index,
                    "target_class": class_names[target_index],
                    "true_label_num": true_index,
                    "true_label": true_label,
                    "predicted_class_num": predicted_index,
                    "predicted_class": pred_label,
                    "positive_class_probability": float(probabilities[positive_index].item()),
                }
            )
            for class_idx, class_name in enumerate(class_names):
                result[f"probability_{class_name}"] = float(probabilities[class_idx].item())
            results.append(result)
    finally:
        gradcam.close()

    results_df = pd.DataFrame(results)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_csv, index=False)
    metrics = evaluate_dataframe(results_df, positive_index=positive_index)
    if metrics:
        pd.DataFrame([metrics]).to_csv(output_csv.with_name(output_csv.stem + "_metrics.csv"), index=False)
    return results_df, metrics
