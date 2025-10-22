# dataloader.py
from glob import glob
import os
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms as T
import numpy as np


class FrameVideoDataset(torch.utils.data.Dataset):
    """
    Dataset for RGB video frames.
    Loads a stack of sampled frames for each video.
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
        self.transform = transform
        self.stack_frames = stack_frames
        self.n_sampled_frames = n_sampled_frames

    def __len__(self):
        return len(self.video_paths)

    def _get_meta(self, attr, value):
        return self.df.loc[self.df[attr] == value]

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        video_name = os.path.basename(video_path).split(".avi")[0]
        video_meta = self._get_meta("video_name", video_name)
        label = video_meta["label"].item()

        video_frames_dir = video_path.replace("videos", "frames").replace(".avi", "")
        video_frames = self.load_frames(video_frames_dir)

        if self.transform:
            frames = [self.transform(frame) for frame in video_frames]
        else:
            frames = [T.ToTensor()(frame) for frame in video_frames]

        if self.stack_frames:
            frames = torch.stack(frames).permute(1, 0, 2, 3)  # [C, T, H, W]

        return frames, label

    def load_frames(self, frames_dir):
        frames = []
        for i in range(1, self.n_sampled_frames + 1):
            frame_file = os.path.join(frames_dir, f"frame_{i}.jpg")
            frame = Image.open(frame_file).convert("RGB")
            frames.append(frame)
        return frames


class CombinedVideoDataset(torch.utils.data.Dataset):
    """
    Returns both RGB frames and Optical Flow stacks for the same video.
    For the HPC structure:
        /dtu/datasets1/02516/ucf101_noleakage/
            frames/{split}/{class}/{video}/frame_*.jpg
            flows/{split}/{class}/{video}/flow_1_2.npy ...
            metadata/{split}.csv
    """

    def __init__(
        self,
        root_dir="/dtu/datasets1/02516/ucf101_noleakage",
        split="train",
        transform=None,
        n_sampled_frames=10,
        flows_dir=None,
    ):
        super().__init__()
        self.rgb_dataset = FrameVideoDataset(
            root_dir=root_dir,
            split=split,
            transform=transform,
            stack_frames=True,
            n_sampled_frames=n_sampled_frames,
        )
        self.split = split
        self.transform = transform
        self.n_sampled_frames = n_sampled_frames
        self.root_dir = root_dir
        self.flows_dir = (
            flows_dir if flows_dir is not None else os.path.join(root_dir, "flows")
        )

    def __len__(self):
        return len(self.rgb_dataset)

    def load_flow_frames(self, flow_dir):
        """
        Load optical flow .npy files of shape [H, W, 2].
        Expected filenames: flow_1_2.npy, flow_2_3.npy, ..., flow_9_10.npy
        Returns a tensor of shape [2*T, H, W]
        """
        flow_frames = []
        npy_files = sorted(glob(os.path.join(flow_dir, "flow_*_*.npy")))

        if len(npy_files) == 0:
            raise FileNotFoundError(f"No .npy flow files found in {flow_dir}")

        # Sample the first n_sampled_frames flow files if there are more
        npy_files = npy_files[: self.n_sampled_frames]

        for npy_path in npy_files:
            flow = np.load(npy_path)  # shape (H, W, 2)
            if flow.shape[-1] != 2:
                raise ValueError(f"Unexpected flow shape {flow.shape} in {npy_path}")
            u, v = flow[..., 0], flow[..., 1]
            u = Image.fromarray(u).convert("L")
            v = Image.fromarray(v).convert("L")

            if self.transform:
                u = self.transform(u)
                v = self.transform(v)
                if u.shape[0] == 3:  # ensure grayscale
                    u = u[0:1]
                    v = v[0:1]
            else:
                u = T.ToTensor()(u)
                v = T.ToTensor()(v)

            flow_frames.append(torch.cat([u, v], dim=0))  # [2, H, W]

        flow_stack = torch.cat(flow_frames, dim=0)  # [2*T, H, W]
        return flow_stack

    def __getitem__(self, idx):
        rgb_stack, label = self.rgb_dataset[idx]
        rgb_video_path = self.rgb_dataset.video_paths[idx]

        # Derive flow directory path
        class_name = rgb_video_path.split("/")[-2]
        video_name = os.path.basename(rgb_video_path).replace(".avi", "")
        flow_video_dir = os.path.join(
            self.flows_dir, self.split, class_name, video_name
        )

        if not os.path.exists(flow_video_dir):
            raise FileNotFoundError(f"Flow directory not found: {flow_video_dir}")

        flow_stack = self.load_flow_frames(flow_video_dir)
        rgb_stack = rgb_stack.permute(1, 0, 2, 3)  # [T, 3, H, W]

        return rgb_stack, flow_stack, label


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    transform = T.Compose([T.Resize((64, 64)), T.ToTensor()])
    dataset = CombinedVideoDataset(
        root_dir="/dtu/datasets1/02516/ucf101_noleakage",
        split="train",
        transform=transform,
        n_sampled_frames=5,
    )
    loader = DataLoader(dataset, batch_size=2, shuffle=False)

    for rgb_stack, flow_stack, labels in loader:
        print("RGB:", rgb_stack.shape)  # [B, T, 3, H, W]
        print("Flow:", flow_stack.shape)  # [B, 2*T, H, W]
        print("Labels:", labels)
        break
