from glob import glob
import os
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms as T
import numpy as np


class FrameImageDataset(torch.utils.data.Dataset):
    def __init__(
        self, root_dir="/work3/ppar/data/ucf101", split="train", transform=None
    ):
        self.frame_paths = sorted(glob(f"{root_dir}/frames/{split}/*/*/*.jpg"))
        self.df = pd.read_csv(f"{root_dir}/metadata/{split}.csv")
        self.split = split
        self.transform = transform

    def __len__(self):
        return len(self.frame_paths)

    def _get_meta(self, attr, value):
        return self.df.loc[self.df[attr] == value]

    def __getitem__(self, idx):
        frame_path = self.frame_paths[idx]
        video_name = frame_path.split("/")[-2]
        video_meta = self._get_meta("video_name", video_name)
        label = video_meta["label"].item()

        frame = Image.open(frame_path).convert("RGB")

        if self.transform:
            frame = self.transform(frame)
        else:
            frame = T.ToTensor()(frame)

        return frame, label


class FrameVideoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root_dir="/work3/ppar/data/ucf101",
        split="train",
        transform=None,
        stack_frames=True,
    ):

        self.video_paths = sorted(glob(f"{root_dir}/videos/{split}/*/*.avi"))
        self.df = pd.read_csv(f"{root_dir}/metadata/{split}.csv")
        self.split = split
        self.transform = transform
        self.stack_frames = stack_frames

        self.n_sampled_frames = 10

    def __len__(self):
        return len(self.video_paths)

    def _get_meta(self, attr, value):
        return self.df.loc[self.df[attr] == value]

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        video_name = video_path.split("/")[-1].split(".avi")[0]
        video_meta = self._get_meta("video_name", video_name)
        label = video_meta["label"].item()

        video_frames_dir = (
            self.video_paths[idx].split(".avi")[0].replace("videos", "frames")
        )
        video_frames = self.load_frames(video_frames_dir)

        if self.transform:
            frames = [self.transform(frame) for frame in video_frames]
        else:
            frames = [T.ToTensor()(frame) for frame in video_frames]

        if self.stack_frames:
            frames = torch.stack(frames).permute(1, 0, 2, 3)

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
    Expected directory structure:
    root_dir/
        frames/{split}/{class}/{video_name}/frame_1.jpg ...
        flows/{split}/{class}/{video_name}/flow_1_u.png / flow_1_v.png ...
        metadata/{split}.csv
    """

    def __init__(
        self,
        root_dir="/work3/ppar/data/ucf101",
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
        Load optical flow pairs (u,v) per frame.
        Supports flow_{i}_u.png and flow_{i}_v.png, or flow_{i}.npy.
        Returns torch.Tensor [2*T, H, W]
        """
        flow_frames = []
        for i in range(1, self.n_sampled_frames + 1):
            u_path = os.path.join(flow_dir, f"flow_{i}_u.png")
            v_path = os.path.join(flow_dir, f"flow_{i}_v.png")
            npy_path = os.path.join(flow_dir, f"flow_{i}.npy")

            if os.path.exists(npy_path):
                flow = np.load(npy_path)
                if flow.shape[-1] == 2:
                    u = flow[..., 0]
                    v = flow[..., 1]
                else:
                    u, v = flow[0], flow[1]
                u = Image.fromarray(u).convert("L")
                v = Image.fromarray(v).convert("L")
            else:
                u = Image.open(u_path).convert("L")
                v = Image.open(v_path).convert("L")

            if self.transform:
                u = self.transform(u)
                v = self.transform(v)
                if u.shape[0] == 3:
                    u = u[0:1]
                    v = v[0:1]
            else:
                u = T.ToTensor()(u)
                v = T.ToTensor()(v)

            flow_frames.append(torch.cat([u, v], dim=0))  # [2, H, W]

        flow_stack = torch.cat(flow_frames, dim=0)  # [2*T, H, W]
        return flow_stack

    def __getitem__(self, idx):
        # RGB frames and labels
        rgb_stack, label = self.rgb_dataset[idx]

        # Derive flow directory
        rgb_video_path = self.rgb_dataset.video_paths[idx]
        flow_video_dir = rgb_video_path.replace("videos", "flows").replace(".avi", "")
        if self.flows_dir not in flow_video_dir:
            flow_video_dir = flow_video_dir.replace(self.root_dir, self.flows_dir)

        if not os.path.exists(flow_video_dir):
            raise FileNotFoundError(f"Flow directory not found: {flow_video_dir}")

        flow_stack = self.load_flow_frames(flow_video_dir)

        # rgb_stack: [C, T, H, W], flow_stack: [2*T, H, W]
        rgb_stack = rgb_stack.permute(1, 0, 2, 3)  # [T, 3, H, W]
        rgb_stack = rgb_stack  # keep as [T, 3, H, W]
        return rgb_stack, flow_stack, label


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    transform = T.Compose([T.Resize((64, 64)), T.ToTensor()])
    dataset = CombinedVideoDataset(
        root_dir="/work3/ppar/data/ucf101",
        split="val",
        transform=transform,
        n_sampled_frames=10,
    )
    loader = DataLoader(dataset, batch_size=2, shuffle=False)

    for rgb_stack, flow_stack, labels in loader:
        print("RGB:", rgb_stack.shape)  # [B, T, 3, H, W]
        print("Flow:", flow_stack.shape)  # [B, 2*T, H, W]
        print("Labels:", labels.shape)
        break
