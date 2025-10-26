#!/usr/bin/env python3
"""
plot_training_multi.py

Usage examples:

# Plot two training logs together (spatial + temporal)
python plot_training_multi.py --csv_files checkpoints/spatial/training_log.csv checkpoints/temporal/training_log.csv --save_fig figures/combined.png

# Plot and save confusion matrix from CSV selecting the model predictions to use:
python plot_training_multi.py \
  --csv_files checkpoints/spatial/training_log.csv checkpoints/temporal/training_log.csv \
  --confusion_csv results/test_predictions.csv \
  --confusion_model fusion \
  --confusion_out figures/confusion_fusion.png
"""

import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Optional


def smooth_series(series, window=5, mode="rolling"):
    if mode == "savgol":
        try:
            from scipy.signal import savgol_filter

            w = int(window) if int(window) % 2 == 1 else int(window) + 1
            if w < 3:
                w = 3
            return savgol_filter(series, window_length=w, polyorder=2, mode="interp")
        except Exception:
            mode = "rolling"

    if mode == "rolling":
        return (
            pd.Series(series)
            .rolling(window=window, center=True, min_periods=1)
            .mean()
            .to_numpy()
        )
    return (
        pd.Series(series)
        .rolling(window=window, center=True, min_periods=1)
        .mean()
        .to_numpy()
    )


def instability_from_residuals(y, smoothed, resid_window=5, metric="std"):
    resid = y - smoothed
    s = pd.Series(resid)
    if metric == "std":
        inst = (
            s.rolling(window=resid_window, center=True, min_periods=1)
            .std()
            .fillna(0)
            .to_numpy()
        )
    else:
        inst = (
            s.abs()
            .rolling(window=resid_window, center=True, min_periods=1)
            .mean()
            .fillna(0)
            .to_numpy()
        )
    return inst


def detect_prefix_from_df(df: pd.DataFrame) -> str:
    """
    Detects a prefix such that prefix + 'train_loss' in columns.
    If no prefix, returns empty string.
    """
    for col in df.columns:
        if col.endswith("train_loss"):
            return col[: col.rfind("train_loss")]
    if "train_loss" in df.columns:
        return ""
    for pref in ["spatial_", "temporal_", "spatial", "temporal"]:
        if f"{pref}train_loss" in df.columns:
            return pref
    return ""


def extract_series_from_df(df: pd.DataFrame, prefix: str) -> Dict[str, np.ndarray]:
    """
    Returns dict with epoch, train_loss, val_loss, train_acc, val_acc arrays.
    """
    col_train_loss = f"{prefix}train_loss" if prefix != "" else "train_loss"
    col_val_loss = f"{prefix}val_loss" if prefix != "" else "val_loss"
    col_train_acc = f"{prefix}train_acc" if prefix != "" else "train_acc"
    col_val_acc = f"{prefix}val_acc" if prefix != "" else "val_acc"

    cols = df.columns.tolist()

    def pick(colname_candidates):
        for c in colname_candidates:
            if c in cols:
                return c
        return None

    c_epoch = pick(["epoch", "Epoch", "epo", "e"])
    c_train_loss = pick([col_train_loss, "train_loss", "loss_train"])
    c_val_loss = pick([col_val_loss, "val_loss", "loss_val"])
    c_train_acc = pick([col_train_acc, "train_acc", "acc_train"])
    c_val_acc = pick([col_val_acc, "val_acc", "acc_val"])

    missing = [
        name
        for name, c in [
            ("epoch", c_epoch),
            ("train_loss", c_train_loss),
            ("val_loss", c_val_loss),
            ("train_acc", c_train_acc),
            ("val_acc", c_val_acc),
        ]
        if c is None
    ]

    if missing:
        raise ValueError(
            f"Missing columns in CSV ({', '.join(missing)}). Found columns: {cols}"
        )

    return {
        "epoch": df[c_epoch].to_numpy(),
        "train_loss": df[c_train_loss].to_numpy(),
        "val_loss": df[c_val_loss].to_numpy(),
        "train_acc": df[c_train_acc].to_numpy(),
        "val_acc": df[c_val_acc].to_numpy(),
    }


def plot_multi_csv(
    csv_files,
    labels=None,
    smooth_window=5,
    resid_window=7,
    smooth_mode="rolling",
    metric="std",
    figsize=(14, 10),
    alpha_fill=0.2,
    linewidth=2,
    save_path=None,
):
    """
    csv_files: list of paths to CSV training logs
    labels: optional list of strings for legend (defaults to basenames)
    """
    if labels is None:
        labels = [os.path.splitext(os.path.basename(p))[0] for p in csv_files]

    fig, axs = plt.subplots(2, 2, figsize=figsize)
    (ax_tl, ax_tr), (ax_bl, ax_br) = axs

    for i, csv in enumerate(csv_files):
        df = pd.read_csv(csv)
        try:
            prefix = detect_prefix_from_df(df)
            series = extract_series_from_df(df, prefix)
        except Exception as e:
            raise RuntimeError(f"Error reading {csv}: {e}")

        epochs = series["epoch"]
        x = epochs

        train_loss = series["train_loss"]
        val_loss = series["val_loss"]
        train_acc = series["train_acc"]
        val_acc = series["val_acc"]

        train_loss_s = smooth_series(train_loss, window=smooth_window, mode=smooth_mode)
        val_loss_s = smooth_series(val_loss, window=smooth_window, mode=smooth_mode)
        train_acc_s = smooth_series(train_acc, window=smooth_window, mode=smooth_mode)
        val_acc_s = smooth_series(val_acc, window=smooth_window, mode=smooth_mode)

        train_loss_inst = instability_from_residuals(
            train_loss, train_loss_s, resid_window=resid_window, metric=metric
        )
        val_loss_inst = instability_from_residuals(
            val_loss, val_loss_s, resid_window=resid_window, metric=metric
        )
        train_acc_inst = instability_from_residuals(
            train_acc, train_acc_s, resid_window=resid_window, metric=metric
        )
        val_acc_inst = instability_from_residuals(
            val_acc, val_acc_s, resid_window=resid_window, metric=metric
        )

        label = labels[i]

        ax_tl.plot(x, train_loss_s, label=label, linewidth=linewidth)
        ax_tl.fill_between(
            x,
            train_loss_s - train_loss_inst,
            train_loss_s + train_loss_inst,
            alpha=alpha_fill,
        )

        ax_tr.plot(x, train_acc_s, label=label, linewidth=linewidth)
        ax_tr.fill_between(
            x,
            train_acc_s - train_acc_inst,
            train_acc_s + train_acc_inst,
            alpha=alpha_fill,
        )

        ax_bl.plot(x, val_loss_s, label=label, linewidth=linewidth)
        ax_bl.fill_between(
            x, val_loss_s - val_loss_inst, val_loss_s + val_loss_inst, alpha=alpha_fill
        )

        ax_br.plot(x, val_acc_s, label=label, linewidth=linewidth)
        ax_br.fill_between(
            x, val_acc_s - val_acc_inst, val_acc_s + val_acc_inst, alpha=alpha_fill
        )

    ax_tl.set_title("Training Loss (smoothed ± instability)")
    ax_tl.set_xlabel("Epoch")
    ax_tl.set_ylabel("Loss")
    ax_tl.grid(True)

    ax_tr.set_title("Training Accuracy (smoothed ± instability)")
    ax_tr.set_xlabel("Epoch")
    ax_tr.set_ylabel("Accuracy")
    ax_tr.grid(True)

    ax_bl.set_title("Validation Loss (smoothed ± instability)")
    ax_bl.set_xlabel("Epoch")
    ax_bl.set_ylabel("Loss")
    ax_bl.grid(True)

    ax_br.set_title("Validation Accuracy (smoothed ± instability)")
    ax_br.set_xlabel("Epoch")
    ax_br.set_ylabel("Accuracy")
    ax_br.grid(True)

    ax_tr.legend()
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path is not None:
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        fig.savefig(save_path, dpi=300)
        print(f"Saved combined figure to {save_path}")
        plt.close(fig)
    else:
        plt.show()


# ------------------------------
# Confusion matrix plotting
# ------------------------------
def plot_confusion_from_csv(
    confusion_csv: str,
    model_choice: str = "fusion",
    labels: Optional[list] = None,
    save_path: Optional[str] = None,
    normalize: Optional[str] = None,
):
    """
    Expects CSV with columns containing:
      - a ground truth 'y_true' (or variants) column
      - a prediction column, depending on model_choice:
        - 'spatial' -> looks for 'y_pred_spatial' or 'y_pred'
        - 'temporal' -> looks for 'y_pred_temporal'
        - 'fusion'/'dual' -> looks for 'y_pred_fusion' or 'y_pred'
    normalize: None | 'true' | 'pred' | 'all' (passed to sklearn.metrics.confusion_matrix via normalize param)
    """
    df = pd.read_csv(confusion_csv)

    # detect true column
    possible_true = ["y_true", "true", "true_label", "label_true"]
    ctrue = next((c for c in possible_true if c in df.columns), None)
    if ctrue is None:
        raise ValueError(
            f"confusion_csv must contain a ground truth column (e.g. 'y_true'). Found columns: {df.columns.tolist()}"
        )

    # pick prediction column based on model_choice
    model_choice = model_choice.lower()
    if model_choice == "spatial":
        candidates = ["y_pred_spatial", "y_pred_sp", "y_pred"]
    elif model_choice == "temporal":
        candidates = ["y_pred_temporal", "y_pred_tm", "y_pred"]
    elif model_choice in ("fusion", "dual"):
        candidates = ["y_pred_fusion", "y_pred_f", "y_pred"]
    else:
        raise ValueError(
            "confusion_model must be one of: spatial, temporal, fusion (or dual)"
        )

    cpred = next((c for c in candidates if c in df.columns), None)
    if cpred is None:
        raise ValueError(
            f"Prediction column for model '{model_choice}' not found in {confusion_csv}. "
            f"Expected one of {candidates}. Available columns: {df.columns.tolist()}"
        )

    y_true = df[ctrue].to_numpy()
    y_pred = df[cpred].to_numpy()

    # compute confusion matrix using sklearn
    try:
        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    except Exception as e:
        raise RuntimeError(
            "scikit-learn is required for confusion matrix plotting. Install scikit-learn."
        ) from e

    cm = confusion_matrix(y_true, y_pred, normalize=normalize)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax, cmap="Blues", colorbar=True)
    plt.xticks(rotation=45)
    title = f"Confusion Matrix ({model_choice})"
    if normalize:
        title += f" - normalized={normalize}"
    plt.title(title)
    plt.tight_layout()

    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        fig.savefig(save_path, dpi=300)
        print(f"Saved confusion matrix to {save_path}")
        plt.close(fig)
    else:
        plt.show()


# ------------------------------
# CLI
# ------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot multiple training_log CSVs together and optionally confusion matrix"
    )
    parser.add_argument(
        "--csv_files",
        nargs="+",
        help="One or more training_log CSV files to plot (space separated)",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Optional legend labels for csv files (same order)",
    )
    parser.add_argument("--smooth_window", type=int, default=5, help="Smoothing window")
    parser.add_argument(
        "--resid_window", type=int, default=7, help="Residual window for instability"
    )
    parser.add_argument(
        "--smooth_mode", type=str, default="rolling", choices=["rolling", "savgol"]
    )
    parser.add_argument("--metric", type=str, default="std", choices=["std", "mad"])
    parser.add_argument(
        "--figsize", nargs=2, type=float, default=(14, 10), help="Figure size (w h)"
    )
    parser.add_argument(
        "--save_fig",
        type=str,
        default=None,
        help="Path to save combined figure (PNG). If omitted shows figure.",
    )
    parser.add_argument(
        "--confusion_csv",
        type=str,
        default=None,
        help="Optional CSV with predictions to plot confusion matrix",
    )
    parser.add_argument(
        "--confusion_model",
        type=str,
        default="fusion",
        choices=["spatial", "temporal", "fusion", "dual"],
        help="Which model predictions to use for confusion matrix",
    )
    parser.add_argument(
        "--confusion_labels",
        type=str,
        default=None,
        help="Optional comma-separated label names for confusion matrix",
    )
    parser.add_argument(
        "--confusion_out",
        type=str,
        default=None,
        help="Path to save confusion matrix image (PNG). If omitted shows it.",
    )
    parser.add_argument(
        "--confusion_normalize",
        type=str,
        default=None,
        choices=[None, "true", "pred", "all"],
        help="Normalize confusion matrix ('true','pred','all') or omit",
    )
    args = parser.parse_args()

    plot_multi_csv(
        args.csv_files,
        labels=args.labels,
        smooth_window=args.smooth_window,
        resid_window=args.resid_window,
        smooth_mode=args.smooth_mode,
        metric=args.metric,
        figsize=tuple(args.figsize),
        alpha_fill=0.25,
        linewidth=2,
        save_path=args.save_fig,
    )

    if args.confusion_csv:
        lbls = None
        if args.confusion_labels:
            lbls = [s.strip() for s in args.confusion_labels.split(",")]
        plot_confusion_from_csv(
            args.confusion_csv,
            model_choice=args.confusion_model,
            labels=lbls,
            save_path=args.confusion_out,
            normalize=args.confusion_normalize,
        )
