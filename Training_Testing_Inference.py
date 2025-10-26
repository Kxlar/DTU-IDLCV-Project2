# Training_Testing_Inference.py
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms as T
from tqdm import tqdm
import os
import csv
from datetime import datetime
import numpy as np
import pandas as pd

from Dataloader import CombinedVideoDataset
from DualStreamNetwork import DualStreamConvNet

print("Script started", flush=True)


# ------------------------------------------------
# Defaults
# ------------------------------------------------
DEFAULT_ROOT_DIR = "/dtu/datasets1/02516/ucf101_noleakage"
DEFAULT_FLOWS_DIR = os.path.join(DEFAULT_ROOT_DIR, "flows")


# ------------------------------------------------
# Utilities
# ------------------------------------------------
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def log_metrics(log_path, epoch, train_loss, train_acc, val_loss, val_acc, prefix=""):
    header = [
        "epoch",
        f"{prefix}train_loss",
        f"{prefix}train_acc",
        f"{prefix}val_loss",
        f"{prefix}val_acc",
    ]
    new_file = not os.path.exists(log_path)
    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(header)
        writer.writerow([epoch, train_loss, train_acc, val_loss, val_acc])


# ------------------------------------------------
# Train / Eval helpers (per-stream)
# ------------------------------------------------
def train_epoch_stream(
    model_stream,
    dataloader,
    criterion,
    optimizer,
    device,
    stream="spatial",
    nb_frames_rgb=10,
):
    """
    model_stream: callable that for spatial expects [B*T, C, H, W] or for temporal expects [B, 2L, H, W]
    For spatial we collapse frames and average logits across frames for loss/metrics.
    """
    model_stream.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for rgb_stack, flow_stack, labels in tqdm(
        dataloader, desc=f"Training {stream}", leave=False
    ):
        labels = labels.to(device)
        if stream == "spatial":
            # rgb_stack: [B, T, 3, H, W]
            B, T, C, H, W = rgb_stack.shape
            x = rgb_stack.view(B * T, C, H, W).to(device)
            optimizer.zero_grad()
            logits_frames = model_stream(x)  # [B*T, nb_class]
            logits = logits_frames.view(B, T, -1).mean(dim=1)  # [B, nb_class]
        else:
            # temporal stream expects flow_stack [B, 2L, H, W]
            x = flow_stack.to(device)
            optimizer.zero_grad()
            logits = model_stream(x)

        # check logits finite
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print(
                "Warning: logits contain NaN/Inf — dumping tensors and skipping batch."
            )
            torch.save(
                {
                    "logits": logits.detach().cpu(),
                    "flow": flow_stack.detach().cpu(),
                    "labels": labels.detach().cpu(),
                },
                "bad_logits_batch.pth",
            )
            optimizer.zero_grad()
            continue

        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


def eval_epoch_stream(
    model_stream, dataloader, criterion, device, stream="spatial", nb_frames_rgb=10
):
    model_stream.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for rgb_stack, flow_stack, labels in tqdm(
            dataloader, desc=f"Eval {stream}", leave=False
        ):
            labels = labels.to(device)
            if stream == "spatial":
                B, T, C, H, W = rgb_stack.shape
                x = rgb_stack.view(B * T, C, H, W).to(device)
                logits_frames = model_stream(x)
                logits = logits_frames.view(B, T, -1).mean(dim=1)
            else:
                x = flow_stack.to(device)
                logits = model_stream(x)

            loss = criterion(logits, labels)
            running_loss += loss.item() * labels.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return running_loss / total, correct / total


# ------------------------------------------------
# Fusion evaluation (softmax average) — used by test_only
# ------------------------------------------------
def evaluate_fusion(spatial_model, temporal_model, dataloader, device):
    spatial_model.eval()
    temporal_model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for rgb_stack, flow_stack, labels in tqdm(
            dataloader, desc="Fusion Eval", leave=False
        ):
            rgb_stack = rgb_stack.to(device)
            flow_stack = flow_stack.to(device)
            labels = labels.to(device)

            B, T, C, H, W = rgb_stack.shape
            # Spatial
            logits_frames = spatial_model(rgb_stack.view(B * T, C, H, W))
            logits_rgb = logits_frames.view(B, T, -1).mean(dim=1)
            probs_rgb = torch.softmax(logits_rgb, dim=1)

            # Temporal
            logits_flow = temporal_model(flow_stack)
            probs_flow = torch.softmax(logits_flow, dim=1)

            probs_fused = 0.5 * (probs_rgb + probs_flow)
            preds = torch.argmax(probs_fused, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total


def load_model_state(model, ckpt_path, device):
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    elif isinstance(state, dict) and all(k.startswith("module.") for k in state.keys()):
        # sometimes saved state_dict directly with module.* keys
        model.load_state_dict(state)
    else:
        # if the file is a dict but not 'model_state_dict', try direct load
        try:
            model.load_state_dict(state)
        except Exception:
            raise RuntimeError(f"Unable to load model state from {ckpt_path}")
    return model


def make_scheduler(optimizer, scheduler_name, args):
    """
    Create a learning rate scheduler based on user input.
    """
    if scheduler_name == "cosine":
        # Smooth cosine decay over epochs
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1e-5
        )

    elif scheduler_name == "step":
        # Classical step decay every 10 epochs
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    elif scheduler_name == "plateau":
        # Reduces LR on plateau (monitor val loss)
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3, verbose=True
        )

    elif scheduler_name == "onecycle":
        # OneCycleLR — popular for fast convergence
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.lr,
            steps_per_epoch=1,  # Will be updated inside training loop if needed
            epochs=args.epochs,
            anneal_strategy="cos",
        )

    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_name}")


# ------------------------------------------------
# Training entrypoints
# ------------------------------------------------
def train_spatial(args, device):
    print("Starting spatial training...", flush=True)

    # transforms
    train_rgb_transform = T.Compose(
        [
            T.Resize(256),
            T.RandomCrop(224),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    val_rgb_transform = T.Compose(
        [
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    flow_transform = T.Compose(
        [T.Resize(224), T.ToTensor(), T.Lambda(lambda x: (x - 0.5) * 2)]
    )

    train_dataset = CombinedVideoDataset(
        root_dir=args.root_dir,
        split="train",
        rgb_transform=train_rgb_transform,
        flow_transform=flow_transform,
        n_sampled_frames=args.n_frames,
        flows_dir=args.flows_dir,
        nb_flow_frames=9,
    )
    val_dataset = CombinedVideoDataset(
        root_dir=args.root_dir,
        split="val",
        rgb_transform=val_rgb_transform,
        flow_transform=flow_transform,
        n_sampled_frames=args.n_frames,
        flows_dir=args.flows_dir,
        nb_flow_frames=9,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Model: spatial stream with pretrained VGG backbone
    spatial_model = DualStreamConvNet(
        nb_class=args.nb_class,
        nb_rgb_frames=args.n_frames,
        nb_flow_frames=9,
        spatial_pretrained=not args.no_pretrained,
        temporal_dropout=0.5,
    ).spatial_stream.to(device)

    # We freeze the backbone features in DualStreamNetwork.SpatialStreamConvNet when use_pretrained=True
    # Only train the fc layers initially
    params_to_train = [p for p in spatial_model.parameters() if p.requires_grad]
    print("Params to train (spatial):", sum(p.numel() for p in params_to_train))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        params_to_train, lr=args.lr, momentum=0.9, weight_decay=1e-4
    )
    scheduler = make_scheduler(optimizer, args.scheduler, args)

    best_val_acc = 0.0
    out_dir = os.path.join(args.output_dir, "spatial")
    ensure_dir(out_dir)
    log_path = os.path.join(out_dir, "training_log.csv")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch_stream(
            spatial_model,
            train_loader,
            criterion,
            optimizer,
            device,
            stream="spatial",
            nb_frames_rgb=args.n_frames,
        )
        val_loss, val_acc = eval_epoch_stream(
            spatial_model,
            val_loader,
            criterion,
            device,
            stream="spatial",
            nb_frames_rgb=args.n_frames,
        )
        print(
            f"[Spatial][Epoch {epoch:02d}] Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}"
        )
        # Log per-epoch metrics for the spatial stream
        log_metrics(
            log_path, epoch, train_loss, train_acc, val_loss, val_acc, prefix="spatial_"
        )
        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = os.path.join(out_dir, f"best_spatial_epoch_{epoch}.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": spatial_model.state_dict(),
                    "val_acc": val_acc,
                },
                ckpt_path,
            )
            print("Saved spatial checkpoint:", ckpt_path)

    print("Spatial training complete. Best val acc:", best_val_acc)
    return os.path.join(out_dir, f"best_spatial_epoch_{epoch}.pth")


def train_temporal(args, device, temporal_dropout=0.9):
    print("Starting temporal training...", flush=True)

    # Use deterministic RGB transforms for the dataset (we only need flows for temporal training)
    rgb_dummy_transform = T.Compose(
        [
            T.Resize(224),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    flow_transform = T.Compose(
        [T.Resize(224), T.ToTensor(), T.Lambda(lambda x: (x - 0.5) * 2)]
    )

    train_dataset = CombinedVideoDataset(
        root_dir=args.root_dir,
        split="train",
        rgb_transform=rgb_dummy_transform,
        flow_transform=flow_transform,
        n_sampled_frames=args.n_frames,
        flows_dir=args.flows_dir,
        nb_flow_frames=9,
    )
    val_dataset = CombinedVideoDataset(
        root_dir=args.root_dir,
        split="val",
        rgb_transform=rgb_dummy_transform,
        flow_transform=flow_transform,
        n_sampled_frames=args.n_frames,
        flows_dir=args.flows_dir,
        nb_flow_frames=9,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    temporal_model = DualStreamConvNet(
        nb_class=args.nb_class,
        nb_rgb_frames=args.n_frames,
        nb_flow_frames=9,
        spatial_pretrained=False,
        temporal_dropout=temporal_dropout,
    ).temporal_stream.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        temporal_model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4
    )
    scheduler = make_scheduler(optimizer, args.scheduler, args)

    best_val_acc = 0.0
    out_dir = os.path.join(args.output_dir, "temporal")
    ensure_dir(out_dir)
    log_path = os.path.join(out_dir, "training_log.csv")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch_stream(
            temporal_model,
            train_loader,
            criterion,
            optimizer,
            device,
            stream="temporal",
        )
        val_loss, val_acc = eval_epoch_stream(
            temporal_model, val_loader, criterion, device, stream="temporal"
        )
        print(
            f"[Temporal][Epoch {epoch:02d}] Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}"
        )
        # Log per-epoch metrics for the temporal stream
        log_metrics(
            log_path,
            epoch,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
            prefix="temporal_",
        )
        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_name = f"best_temporal_epoch_{epoch}.pth"
            ckpt_path = os.path.join(out_dir, ckpt_name)

            # Supprimer tous les anciens checkpoints "best_temporal_epoch_*"
            old_checkpoints = [
                f for f in os.listdir(out_dir) if f.startswith("best_temporal_epoch_")
            ]
            for old_ckpt in old_checkpoints:
                old_path = os.path.join(out_dir, old_ckpt)
                try:
                    os.remove(old_path)
                    print(f"Removed old checkpoint: {old_ckpt}")
                except Exception as e:
                    print(f"Could not remove {old_ckpt}: {e}")

            # Sauvegarde du nouveau meilleur modèle
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": temporal_model.state_dict(),
                    "val_acc": val_acc,
                },
                ckpt_path,
            )
            print(f"Saved new best temporal checkpoint: {ckpt_path}")

    print("Temporal training complete. Best val acc:", best_val_acc)
    return os.path.join(out_dir, f"best_temporal_epoch_{epoch}.pth")


def test_only_evaluation(args, device):
    """
    Loads spatial and temporal checkpoints, evaluates on the TEST split,
    writes per-sample predictions to args.output_dir/test_predictions.csv
    and prints test accuracies for spatial, temporal and fusion.
    """
    if not (args.spatial_ckpt and args.temporal_ckpt):
        raise ValueError(
            "For test_only mode, provide --spatial_ckpt and --temporal_ckpt"
        )

    print("Loading checkpoints:", args.spatial_ckpt, args.temporal_ckpt, flush=True)
    from torchvision import transforms as T

    # Build models and load weights
    wrapper = DualStreamConvNet(
        nb_class=args.nb_class,
        nb_rgb_frames=args.n_frames,
        nb_flow_frames=9,
        spatial_pretrained=True,
    )
    spatial_model = wrapper.spatial_stream.to(device)
    temporal_model = wrapper.temporal_stream.to(device)

    spatial_model = load_model_state(spatial_model, args.spatial_ckpt, device)
    temporal_model = load_model_state(temporal_model, args.temporal_ckpt, device)

    # Deterministic test transforms
    test_rgb_transform = T.Compose(
        [
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    flow_transform = T.Compose(
        [T.Resize(224), T.ToTensor(), T.Lambda(lambda x: (x - 0.5) * 2)]
    )

    test_dataset = CombinedVideoDataset(
        root_dir=args.root_dir,
        split="test",
        rgb_transform=test_rgb_transform,
        flow_transform=flow_transform,
        n_sampled_frames=args.n_frames,
        flows_dir=args.flows_dir,
        nb_flow_frames=9,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print(f"Test set size: {len(test_dataset)} samples", flush=True)

    spatial_model.eval()
    temporal_model.eval()

    # Containers for predictions
    y_true_list = []
    y_pred_spatial_list = []
    y_pred_temporal_list = []
    y_pred_fusion_list = []

    softmax = nn.Softmax(dim=1)

    with torch.no_grad():
        for rgb_stack, flow_stack, labels in tqdm(
            test_loader, desc="Testing", leave=False
        ):
            # Move to device
            rgb_stack = rgb_stack.to(device)  # [B, T, 3, H, W]
            flow_stack = flow_stack.to(device)  # [B, 2*L, H, W]
            labels = labels.to(device)  # [B]

            B, T, C, H, W = rgb_stack.shape

            # Spatial forward: collapse frames -> [B*T, C, H, W], then average frame logits
            logits_frames = spatial_model(
                rgb_stack.view(B * T, C, H, W)
            )  # [B*T, nb_class]
            logits_rgb = logits_frames.view(B, T, -1).mean(dim=1)  # [B, nb_class]
            probs_rgb = softmax(logits_rgb)  # [B, nb_class]
            preds_rgb = torch.argmax(probs_rgb, dim=1)  # [B]

            # Temporal forward
            logits_flow = temporal_model(flow_stack)  # [B, nb_class]
            probs_flow = softmax(logits_flow)  # [B, nb_class]
            preds_flow = torch.argmax(probs_flow, dim=1)  # [B]

            # Fusion (average softmax probabilities)
            probs_fused = 0.5 * (probs_rgb + probs_flow)
            preds_fused = torch.argmax(probs_fused, dim=1)

            # Append per-sample
            y_true_list.extend(labels.cpu().tolist())
            y_pred_spatial_list.extend(preds_rgb.cpu().tolist())
            y_pred_temporal_list.extend(preds_flow.cpu().tolist())
            y_pred_fusion_list.extend(preds_fused.cpu().tolist())

    # compute accuracies
    y_true_arr = np.array(y_true_list)
    sp_arr = np.array(y_pred_spatial_list)
    tm_arr = np.array(y_pred_temporal_list)
    fu_arr = np.array(y_pred_fusion_list)

    spatial_acc = (y_true_arr == sp_arr).mean() if len(y_true_arr) > 0 else 0.0
    temporal_acc = (y_true_arr == tm_arr).mean() if len(y_true_arr) > 0 else 0.0
    fusion_acc = (y_true_arr == fu_arr).mean() if len(y_true_arr) > 0 else 0.0

    print(f"[TEST] Spatial-only accuracy: {spatial_acc:.4f}")
    print(f"[TEST] Temporal-only accuracy: {temporal_acc:.4f}")
    print(f"[TEST] Fusion (softmax average) accuracy: {fusion_acc:.4f}")

    # Save per-sample predictions to CSV so you can draw confusion matrices later
    ensure_dir(args.output_dir)
    preds_csv = os.path.join(args.output_dir, "test_predictions.csv")
    df_out = pd.DataFrame(
        {
            "y_true": y_true_list,
            "y_pred_spatial": y_pred_spatial_list,
            "y_pred_temporal": y_pred_temporal_list,
            "y_pred_fusion": y_pred_fusion_list,
        }
    )
    df_out.to_csv(preds_csv, index=False)
    print(f"Saved test predictions CSV to {preds_csv}", flush=True)

    # Also save a small summary CSV with overall accuracies and metadata
    summary_csv = os.path.join(args.output_dir, "test_summary.csv")
    summary_row = {
        "timestamp": datetime.now().isoformat(),
        "spatial_ckpt": args.spatial_ckpt,
        "temporal_ckpt": args.temporal_ckpt,
        "test_set_size": len(test_dataset),
        "spatial_acc": f"{spatial_acc:.6f}",
        "temporal_acc": f"{temporal_acc:.6f}",
        "fusion_acc": f"{fusion_acc:.6f}",
    }
    write_header = not os.path.exists(summary_csv)
    with open(summary_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(summary_row)

    print(f"Saved test summary CSV to {summary_csv}", flush=True)
    return


# ------------------------------------------------
# Main
# ------------------------------------------------
def main(args):
    print("Main started", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    ensure_dir(args.output_dir)
    ensure_dir(os.path.join(args.output_dir, "spatial"))
    ensure_dir(os.path.join(args.output_dir, "temporal"))

    if args.mode == "train_spatial":
        train_spatial(args, device)
        return
    if args.mode == "train_temporal":
        train_temporal(args, device, temporal_dropout=0.7)
        return
    if args.mode == "train_both":
        # Train spatial first (with pretrained backbone), then temporal
        print("Train spatial then temporal sequentially", flush=True)
        train_spatial(args, device)
        train_temporal(args, device, temporal_dropout=0.7)
        return

    if args.mode == "test_only":
        test_only_evaluation(args, device)
        return

    print(
        "Unknown mode. Use --mode train_spatial|train_temporal|train_both|test_only",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train / Validate / Test Dual Stream ConvNet (separate streams)"
    )
    parser.add_argument("--root_dir", type=str, default=DEFAULT_ROOT_DIR)
    parser.add_argument("--flows_dir", type=str, default=DEFAULT_FLOWS_DIR)
    parser.add_argument("--nb_class", type=int, default=10)
    parser.add_argument("--n_frames", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument(
        "--scheduler",
        type=str,
        default="cosine",
        choices=["cosine", "step", "plateau", "onecycle"],
        help="Type of learning rate scheduler to use (default: cosine)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="train_both",
        help="train_spatial|train_temporal|train_both|test_only",
    )
    parser.add_argument("--spatial_ckpt", type=str, default=None)
    parser.add_argument("--temporal_ckpt", type=str, default=None)
    parser.add_argument(
        "--no_pretrained", action="store_true", help="Train spatial stream from scratch"
    )

    args = parser.parse_args()
    main(args)
