import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def smooth_series(series, window=5, mode="rolling"):
    """
    Smooths a 1D series.
    - mode="rolling": centred moving average (pandas rolling centre=True)
    - mode="savgol": Savitzky-Golay (if SciPy available) -> else fallback to rolling
    """
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
    """
    Calculates a measure of local instability of residuals (y - smoothed).
    metric: 'std' -> local standard deviation of residuals
            'mad' -> local mean absolute deviation
    """
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


def plot_training_logs(
    csv_file,
    smooth_window=5,
    resid_window=7,
    smooth_mode="rolling",
    metric="std",
    figsize=(14, 10),
    alpha_fill=0.25,
    linewidth=2,
):
    """
    Plot smoothed loss and accuracy curves with local instability bands.
    Works with a single training log file containing:
        epoch, train_loss, train_acc, val_loss, val_acc
    """

    df = pd.read_csv(csv_file)
    required = {"epoch", "train_loss", "val_loss", "train_acc", "val_acc"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"The CSV file must contain the following columns: {required}")

    epochs = df["epoch"].to_numpy()
    train_loss = df["train_loss"].to_numpy()
    val_loss = df["val_loss"].to_numpy()
    train_acc = df["train_acc"].to_numpy()
    val_acc = df["val_acc"].to_numpy()

    # Smooth the series
    train_loss_s = smooth_series(train_loss, window=smooth_window, mode=smooth_mode)
    val_loss_s = smooth_series(val_loss, window=smooth_window, mode=smooth_mode)
    train_acc_s = smooth_series(train_acc, window=smooth_window, mode=smooth_mode)
    val_acc_s = smooth_series(val_acc, window=smooth_window, mode=smooth_mode)

    # Compute instability (variance of residuals)
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

    # Plot 2x2 grid
    fig, axs = plt.subplots(2, 2, figsize=figsize)
    (ax_tl, ax_tr), (ax_bl, ax_br) = axs

    # Training loss
    ax_tl.plot(
        epochs, train_loss_s, label="Train Loss", color="tab:blue", linewidth=linewidth
    )
    ax_tl.fill_between(
        epochs,
        train_loss_s - train_loss_inst,
        train_loss_s + train_loss_inst,
        color="tab:blue",
        alpha=alpha_fill,
    )
    ax_tl.set_title("Training Loss (smoothed ± instability)")
    ax_tl.set_xlabel("Epoch")
    ax_tl.set_ylabel("Loss")
    ax_tl.grid(True)

    # Training accuracy
    ax_tr.plot(
        epochs,
        train_acc_s,
        label="Train Accuracy",
        color="tab:green",
        linewidth=linewidth,
    )
    ax_tr.fill_between(
        epochs,
        train_acc_s - train_acc_inst,
        train_acc_s + train_acc_inst,
        color="tab:green",
        alpha=alpha_fill,
    )
    ax_tr.set_title("Training Accuracy (smoothed ± instability)")
    ax_tr.set_xlabel("Epoch")
    ax_tr.set_ylabel("Accuracy")
    ax_tr.grid(True)

    # Validation loss
    ax_bl.plot(
        epochs,
        val_loss_s,
        label="Validation Loss",
        color="tab:orange",
        linewidth=linewidth,
    )
    ax_bl.fill_between(
        epochs,
        val_loss_s - val_loss_inst,
        val_loss_s + val_loss_inst,
        color="tab:orange",
        alpha=alpha_fill,
    )
    ax_bl.set_title("Validation Loss (smoothed ± instability)")
    ax_bl.set_xlabel("Epoch")
    ax_bl.set_ylabel("Loss")
    ax_bl.grid(True)

    # Validation accuracy
    ax_br.plot(
        epochs,
        val_acc_s,
        label="Validation Accuracy",
        color="tab:red",
        linewidth=linewidth,
    )
    ax_br.fill_between(
        epochs,
        val_acc_s - val_acc_inst,
        val_acc_s + val_acc_inst,
        color="tab:red",
        alpha=alpha_fill,
    )
    ax_br.set_title("Validation Accuracy (smoothed ± instability)")
    ax_br.set_xlabel("Epoch")
    ax_br.set_ylabel("Accuracy")
    ax_br.grid(True)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot training and validation curves from log file"
    )
    parser.add_argument(
        "--csv_file", type=str, required=True, help="Path to training_log.csv"
    )
    parser.add_argument(
        "--smooth_window", type=int, default=5, help="Smoothing window size"
    )
    args = parser.parse_args()

    plot_training_logs(args.csv_file, smooth_window=args.smooth_window)
