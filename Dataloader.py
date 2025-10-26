# Dataloader.py
from glob import glob
import os
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms as T
import numpy as np
import random


class FrameVideoDataset(torch.utils.data.Dataset):
    """
    Loads frames for a video. Returns a tensor of shape [C, T, H, W]
    (where C=3 for RGB after transform), and the label (int).
    transform should be a torchvision transform that outputs a tensor [C,H,W].
    """

    def __init__(
        self,
        root_dir="/dtu/datasets1/02516/ucf101_noleakage",
        split="train",
        transform=None,
        stack_frames=True,
        n_sampled_frames=10,
    ):
        self.video_paths = sorted(glob(f"{root_dir}/videos/{split}/*/*.avi"))
        self.df = pd.read_csv(f"{root_dir}/metadata/{split}.csv")
        self.split = split
        self.transform = transform  # RGB transform (tensor & normalization)
        self.stack_frames = stack_frames
        self.n_sampled_frames = n_sampled_frames

    def __len__(self):
        return len(self.video_paths)

    def _get_meta(self, attr, value):
        return self.df.loc[self.df[attr] == value]

    def load_frames(self, frames_dir, start_idx=1):
        """
        Load frames start_idx ... start_idx + n_sampled_frames - 1
        If a frame is missing this will raise; we assume frames exist as provided.
        Returns list of PIL images (RGB).
        """
        frames = []
        for i in range(start_idx, start_idx + self.n_sampled_frames):
            frame_file = os.path.join(frames_dir, f"frame_{i}.jpg")
            frame = Image.open(frame_file).convert("RGB")
            frames.append(frame)
        return frames

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        video_name = os.path.basename(video_path).split(".avi")[0]
        video_meta = self._get_meta("video_name", video_name)
        label = int(video_meta["label"].item())

        video_frames_dir = video_path.replace("videos", "frames").replace(".avi", "")

        # NOTE: we do NOT implement random temporal crop because the provided flow
        # frames are fixed and you mentioned it's not possible in your dataset.
        # We keep the deterministic sampling: frames 1..n_sampled_frames.
        video_frames = self.load_frames(video_frames_dir, start_idx=1)

        # Apply RGB transform per-frame (if any)
        if self.transform:
            frames = [self.transform(frame) for frame in video_frames]
        else:
            frames = [T.ToTensor()(frame) for frame in video_frames]

        # Stack frames: resulting shape [T, C, H, W] -> then permute to [C, T, H, W]
        if self.stack_frames:
            frames = torch.stack(frames).permute(1, 0, 2, 3)

        return frames, label


class CombinedVideoDataset(torch.utils.data.Dataset):
    """
    Returns:
     - rgb_stack: tensor [T, 3, H, W]  (after FrameVideoDataset we permute)
     - flow_stack: tensor [2*nb_flow_frames, H, W]  (u1,v1,u2,v2,...)
     - label: int
    Accepts separate transforms for RGB and Flow.
    """

    def __init__(
        self,
        root_dir="/dtu/datasets1/02516/ucf101_noleakage",
        split="train",
        rgb_transform=None,
        flow_transform=None,
        n_sampled_frames=10,
        flows_dir=None,
        nb_flow_frames=9,
        flow_mean=None,
        flow_std=None,
    ):
        # FrameVideoDataset expects transform as its 'transform' arg (RGB)
        self.rgb_dataset = FrameVideoDataset(
            root_dir, split, rgb_transform, True, n_sampled_frames
        )
        self.split = split
        self.rgb_transform = rgb_transform
        self.flow_transform = flow_transform
        self.n_sampled_frames = n_sampled_frames
        self.nb_flow_frames = nb_flow_frames
        self.root_dir = root_dir
        self.flows_dir = flows_dir if flows_dir else os.path.join(root_dir, "flows")
        self.flow_mean = flow_mean
        self.flow_std = flow_std

    def __len__(self):
        return len(self.rgb_dataset)

    def load_flow_frames(self, flow_dir, nb_flow_frames=None):
        """Return torch tensor [2*nb_flow_frames, H, W] of dtype float32, normalized."""
        if nb_flow_frames is None:
            nb_flow_frames = self.nb_flow_frames

        npy_files = sorted(glob(os.path.join(flow_dir, "flow_*_*.npy")))[
            :nb_flow_frames
        ]
        if len(npy_files) == 0:
            raise FileNotFoundError(f"No .npy flow files found in {flow_dir}")

        # pad by repeating last if fewer files
        while len(npy_files) < nb_flow_frames:
            npy_files.append(npy_files[-1])

        flow_frames = []
        for npy_path in npy_files:
            flow = np.load(npy_path).astype("float32")  # keep float values
            # handle shapes [2,H,W] or [H,W,2]
            if flow.ndim == 3:
                if flow.shape[0] == 2:
                    u = flow[0]
                    v = flow[1]
                elif flow.shape[-1] == 2:
                    u = flow[..., 0]
                    v = flow[..., 1]
                else:
                    raise ValueError(
                        f"Unexpected flow shape {flow.shape} in {npy_path}"
                    )
            else:
                raise ValueError(f"Invalid flow shape {flow.shape} in {npy_path}")

            # convert to torch
            u_t = torch.from_numpy(u).float()
            v_t = torch.from_numpy(v).float()

            # per-sample normalization: scale to [-1,1] by max abs (avoid division by zero)
            max_abs = max(u_t.abs().max().item(), v_t.abs().max().item(), 1e-6)
            u_t = u_t / max_abs
            v_t = v_t / max_abs

            # optional global mean/std subtraction (if provided)
            if self.flow_mean is not None and self.flow_std is not None:
                # flow_mean expected as (mean_u, mean_v), flow_std as (std_u, std_v)
                u_t = (u_t - float(self.flow_mean[0])) / (
                    float(self.flow_std[0]) + 1e-8
                )
                v_t = (v_t - float(self.flow_mean[1])) / (
                    float(self.flow_std[1]) + 1e-8
                )

            # Now, if user provided a torchvision transform for flow (unlikely), apply it:
            # BUT flow_transform must expect a tensor, so skip PIL conversions.
            if self.flow_transform is not None:
                # we expect a transform that accepts torch Tensor, but many torchvision transforms accept PIL images.
                # Best practice: keep flow_transform to be ToTensor / Lambda only. If not, skip.
                try:
                    # put channels-first [2,H,W]
                    fv = torch.stack([u_t, v_t], dim=0)  # [2,H,W]
                    # if transform is a compose with Lambda expecting tensor, we can call it:
                    fv = self.flow_transform(fv)
                    # if transform returns shape [2,H,W] keep as is
                    if fv.ndim == 3:
                        flow_frames.append(fv)
                        continue
                except Exception:
                    # fallback: continue with raw tensors
                    pass

            # default: stack u and v as [2,H,W]
            flow_frames.append(torch.stack([u_t, v_t], dim=0))

        # concatenate along channel axis -> [2*nb_flow_frames, H, W]
        return torch.cat(flow_frames, dim=0)

    def __getitem__(self, idx):
        rgb_stack, label = self.rgb_dataset[idx]  # rgb_stack shape: [C, T, H, W]
        rgb_video_path = self.rgb_dataset.video_paths[idx]
        class_name = rgb_video_path.split("/")[-2]
        video_name = os.path.basename(rgb_video_path).replace(".avi", "")
        flow_video_dir = os.path.join(
            self.flows_dir, self.split, class_name, video_name
        )

        flow_stack = self.load_flow_frames(
            flow_video_dir, nb_flow_frames=self.nb_flow_frames
        )

        # Permute rgb_stack -> [T, C, H, W] for the model pipeline
        rgb_stack = rgb_stack.permute(1, 0, 2, 3)

        return rgb_stack, flow_stack, label


if __name__ == "__main__":
    """
    Quick sanity check for CombinedVideoDataset:
    - Loads one batch (train split)
    - Prints tensor shapes, dtype, value ranges
    - Verifies that flow and RGB normalization produce finite values
    """
    import torch
    from torch.utils.data import DataLoader

    print("=== Testing CombinedVideoDataset ===")

    # --- configuration ---
    root_dir = "/dtu/datasets1/02516/ucf101_noleakage"
    flows_dir = os.path.join(root_dir, "flows")
    n_frames = 10
    nb_flow_frames = 9
    batch_size = 2

    # --- transforms ---
    from torchvision import transforms as T

    rgb_transform = T.Compose(
        [
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # flow_transform will just re-scale [-1,1]
    flow_transform = T.Compose(
        [T.Lambda(lambda x: x)]
    )  # no-op, flows normalized inside loader

    # --- instantiate dataset ---
    dataset = CombinedVideoDataset(
        root_dir=root_dir,
        split="train",
        rgb_transform=rgb_transform,
        flow_transform=flow_transform,
        n_sampled_frames=n_frames,
        flows_dir=flows_dir,
        nb_flow_frames=nb_flow_frames,
    )

    # --- DataLoader ---
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # --- fetch one batch ---
    rgb_stack, flow_stack, labels = next(iter(loader))

    print("\nSample batch loaded successfully!")
    print(f"RGB stack shape     : {tuple(rgb_stack.shape)}  (expected [B, T, 3, H, W])")
    print(
        f"Flow stack shape    : {tuple(flow_stack.shape)}  (expected [B, 2*{nb_flow_frames}, H, W])"
    )
    print(f"Labels shape        : {tuple(labels.shape)}  (expected [B])")
    print(
        f"Labels dtype        : {labels.dtype}, unique values: {torch.unique(labels)}"
    )

    # --- sanity checks ---
    print("\nRGB stats:")
    print(
        f"  min={rgb_stack.min().item():.4f}, max={rgb_stack.max().item():.4f}, mean={rgb_stack.mean().item():.4f}"
    )
    print(f"Flow stats:")
    print(
        f"  min={flow_stack.min().item():.4f}, max={flow_stack.max().item():.4f}, mean={flow_stack.mean().item():.4f}"
    )

    print("\nNaN / Inf checks:")
    print(
        f"  RGB NaN?  {torch.isnan(rgb_stack).any().item()} | Inf? {torch.isinf(rgb_stack).any().item()}"
    )
    print(
        f"  Flow NaN? {torch.isnan(flow_stack).any().item()} | Inf? {torch.isinf(flow_stack).any().item()}"
    )

    print(
        "\nDataloader test passed ✅"
        if not torch.isnan(flow_stack).any()
        else "\n⚠️ Flow contains NaNs!"
    )
