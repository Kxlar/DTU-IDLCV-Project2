# DualStreamNetwork.py
import torch
import torch.nn as nn
import torchvision.models as models


class SpatialStreamConvNet(nn.Module):
    """
    Spatial stream. Optionally initialised from a pretrained VGG16_BN
    to mimic ImageNet pretraining (we keep the VGG features and add FCs).
    """

    def __init__(self, nb_class=10, use_pretrained=True, dropout=0.5):
        super().__init__()
        self.use_pretrained = use_pretrained
        if use_pretrained:
            vgg = models.vgg16_bn(pretrained=True)
            # use the feature extractor (convs + pooling) from VGG
            self.features = vgg.features  # outputs shape [B,512,7,7] for 224x224 input
            # Freeze features initially (we will fine-tune last layers optionally)
            for param in self.features.parameters():
                param.requires_grad = False
            in_fc = 512 * 7 * 7
        else:
            # define convolutional stack similar to original implementation
            self.features = nn.Sequential(
                nn.Conv2d(3, 96, 7, 2, 3),
                nn.BatchNorm2d(96),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(96, 256, 5, 2, 2),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(256, 512, 3, 1, 1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.Conv2d(512, 512, 3, 1, 1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.Conv2d(512, 512, 3, 1, 1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )
            in_fc = 512 * 7 * 7

        self.flatten = nn.Flatten()
        self.fc1 = nn.Sequential(nn.Linear(in_fc, 4096), nn.ReLU(), nn.Dropout(dropout))
        self.fc2 = nn.Sequential(nn.Linear(4096, 2048), nn.ReLU(), nn.Dropout(dropout))
        self.fc_out = nn.Linear(2048, nb_class)

    def forward(self, x):
        # x expected [B*T, 3, H, W] for frame-by-frame processing
        x = self.features(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        logits = self.fc_out(x)
        return logits


class TemporalStreamConvNet(nn.Module):
    """
    Temporal stream. Input channels = 2 * nb_flow_frames (u1,v1,u2,v2,...).
    If training from scratch, higher dropout is recommended (e.g. 0.9).
    """

    def __init__(self, nb_class=10, nb_flow_frames=9, dropout=0.5):
        super().__init__()
        in_channels = 2 * nb_flow_frames
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 96, 7, 2, 3),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 256, 5, 2, 2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.flatten = nn.Flatten()
        self.fc1 = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096), nn.ReLU(), nn.Dropout(dropout)
        )
        self.fc2 = nn.Sequential(nn.Linear(4096, 2048), nn.ReLU(), nn.Dropout(dropout))
        self.fc_out = nn.Linear(2048, nb_class)

    def forward(self, x):
        # x expected [B, 2*L, H, W]
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        logits = self.fc_out(x)
        return logits


class DualStreamConvNet(nn.Module):
    """
    Wrapper that contains both streams. forward returns (logits_rgb, logits_flow)
    so fusion (softmax averaging or SVM) can be done outside as in the paper.
    """

    def __init__(
        self,
        nb_class=10,
        nb_rgb_frames=10,
        nb_flow_frames=9,
        spatial_pretrained=True,
        temporal_dropout=0.5,
    ):
        super().__init__()
        # spatial: option to use pretrained vgg16_bn
        self.spatial_stream = SpatialStreamConvNet(
            nb_class, use_pretrained=spatial_pretrained, dropout=0.5
        )
        self.temporal_stream = TemporalStreamConvNet(
            nb_class, nb_flow_frames=nb_flow_frames, dropout=temporal_dropout
        )

    def forward(self, x_rgb, x_flow):
        """
        x_rgb: [B, T, 3, H, W]
        x_flow: [B, 2*L, H, W]
        Returns: logits_rgb [B, nb_class], logits_flow [B, nb_class]
        """
        B, T, C, H, W = x_rgb.shape
        # process RGB frame-by-frame
        x_rgb_frames = x_rgb.view(B * T, C, H, W)  # [B*T, C, H, W]
        logits_rgb_frames = self.spatial_stream(x_rgb_frames)  # [B*T, nb_class]
        logits_rgb = logits_rgb_frames.view(B, T, -1).mean(
            dim=1
        )  # average over frames -> [B, nb_class]

        logits_flow = self.temporal_stream(x_flow)  # [B, nb_class]

        return logits_rgb, logits_flow
