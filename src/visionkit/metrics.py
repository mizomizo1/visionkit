from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc as pr_auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    roc_auc_score,
)


def safe_auc(y_true, y_score) -> float:
    try:
        return float(roc_auc_score(y_true, y_score))
    except ValueError:
        return float("nan")


def bootstrap_auc_ci(y_true, y_score, n_bootstraps: int = 1000, seed: int = 123) -> tuple[float, float]:
    y_true = pd.Series(y_true).reset_index(drop=True)
    y_score = pd.Series(y_score).reset_index(drop=True)
    if y_true.nunique() < 2:
        return float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(n_bootstraps):
        indices = rng.choice(len(y_true), len(y_true), replace=True)
        sample_true = y_true.iloc[indices]
        sample_score = y_score.iloc[indices]
        if sample_true.nunique() > 1:
            scores.append(roc_auc_score(sample_true, sample_score))
    if not scores:
        return float("nan"), float("nan")
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def classification_metrics(
    y_true,
    y_pred,
    y_score: Optional[object] = None,
    positive_index: int = 1,
    n_bootstraps: int = 1000,
    seed: int = 123,
) -> dict[str, float]:
    y_true_series = pd.Series(y_true).astype(int)
    y_pred_series = pd.Series(y_pred).astype(int)
    unique_labels = sorted(set(y_true_series.dropna().tolist()))
    binary = len(unique_labels) <= 2

    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true_series, y_pred_series)),
    }

    if binary:
        metrics["precision"] = float(
            precision_score(y_true_series, y_pred_series, pos_label=positive_index, zero_division=0)
        )
        metrics["f1"] = float(f1_score(y_true_series, y_pred_series, pos_label=positive_index, zero_division=0))
        labels = sorted(set(unique_labels) | {0, positive_index})
        cm = confusion_matrix(y_true_series, y_pred_series, labels=labels)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics["sensitivity"] = float(tp / (tp + fn)) if (tp + fn) else float("nan")
            metrics["specificity"] = float(tn / (tn + fp)) if (tn + fp) else float("nan")
    else:
        metrics["macro_precision"] = float(precision_score(y_true_series, y_pred_series, average="macro", zero_division=0))
        metrics["macro_f1"] = float(f1_score(y_true_series, y_pred_series, average="macro", zero_division=0))

    if y_score is not None and binary:
        metrics["auc"] = safe_auc(y_true_series, y_score)
        precision_curve, recall_curve, _ = precision_recall_curve(y_true_series, y_score, pos_label=positive_index)
        metrics["pr_auc"] = float(pr_auc(recall_curve, precision_curve))
        lower, upper = bootstrap_auc_ci(y_true_series, y_score, n_bootstraps=n_bootstraps, seed=seed)
        metrics["auc_ci_lower"] = lower
        metrics["auc_ci_upper"] = upper
    return metrics
