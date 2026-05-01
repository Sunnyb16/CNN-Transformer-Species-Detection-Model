import torch
import torch.nn as nn
import torchvision.models as models

class BirdResNet(nn.Module):
    def __init__(self, num_classes, dropout=0.2):
        super().__init__()

        self.model = models.resnet18(pretrained=True)

        # Fix input channels (audio = 1 channel, not 3)
        self.model.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # Replace classifier head
        self.model.fc = nn.Sequential(
            nn.BatchNorm1d(self.model.fc.in_features),  # optional but strong
            nn.Dropout(p=dropout),
            nn.Linear(self.model.fc.in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)
