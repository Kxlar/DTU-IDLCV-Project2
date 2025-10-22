import torch
import torch.nn as nn
import torch.nn.functional as F


# Spacial stream ConvNet
class SpacialStreamConvNet(nn.Module):
    def __init__(self, nb_class=10):
        super(SpacialStreamConvNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 96, (7, 7), stride=2, padding="same"),
            nn.LocalResponseNorm(5),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 256, (5, 5), stride=2, padding="same"),
            nn.LocalResponseNorm(5),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 512, (3, 3), stride=1, padding="same"), nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(512, 512, (3, 3), stride=1, padding="same"), nn.ReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 512, (3, 3), stride=1, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.flatten = nn.Flatten()
        self.fc1 = nn.Sequential(
            nn.Linear(
                512 * 7 * 7, 4096
            ),  # modify 7*7 according to image size after poolings
            nn.ReLU(),
            nn.Dropout2d(0.3),
        )
        self.fc2 = nn.Sequential(nn.Linear(4096, 2048), nn.ReLU(), nn.Dropout2d(0.3))
        self.fc_out = nn.Sequential(nn.Linear(2048, nb_class))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc2(x)
        x = self.fc_out(x)
        return x


# Temporal stream ConvNet
class TemporalStreamConvNet(nn.Module):
    def __init__(self, nb_class=10):
        super(TemporalStreamConvNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(20, 96, (7, 7), stride=2, padding="same"),
            nn.LocalResponseNorm(5),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 256, (5, 5), stride=2, padding="same"),
            nn.LocalResponseNorm(5),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 512, (3, 3), stride=1, padding="same"), nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(512, 512, (3, 3), stride=1, padding="same"), nn.ReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 512, (3, 3), stride=1, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.flatten = nn.Flatten()
        self.fc1 = nn.Sequential(
            nn.Linear(
                512 * 7 * 7, 4096
            ),  # modify 7*7 according to image size after poolings
            nn.ReLU(),
            nn.Dropout2d(0.3),
        )
        self.fc2 = nn.Sequential(nn.Linear(4096, 2048), nn.ReLU(), nn.Dropout2d(0.3))
        self.fc_out = nn.Sequential(nn.Linear(2048, nb_class))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc2(x)
        x = self.fc_out(x)
        return x


# Dual stream Net
class DualStreamConvNet(nn.Module):
    def __init__(self, nb_class=10):
        super(SpacialStreamConvNet, self).__init__()
        self.spacial_stream = SpacialStreamConvNet(nb_class)
        self.temporal_stream = TemporalStreamConvNet(nb_class)

    def forward(self, x_rgb, x_flow):
        logits_rgb = self.spatial_stream(x_rgb)
        logits_flow = self.temporal_stream(x_flow)
        # Late fusion: moyenne des logits
        logits_fused = (logits_rgb + logits_flow) / 2
        return logits_fused
