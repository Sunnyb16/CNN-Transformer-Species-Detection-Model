# 🐦 Bird Audio Classification with Mel Spectrograms + ResNet

This repository implements a deep learning pipeline for **multi-label bird species classification** from raw audio using **mel spectrogram representations** and a fine-tuned **ResNet-18 architecture**.

---

## 📌 Overview

Environmental audio is inherently noisy, non-stationary, and high-dimensional. This project transforms raw audio into structured time-frequency representations and leverages convolutional neural networks to learn discriminative acoustic patterns.

**Pipeline:**

Raw Audio → Mel Spectrogram → CNN (ResNet-18) → Multi-label Predictions

---

## 🧠 Model Architecture

The core model is a modified **ResNet-18** adapted for audio input:

- Input channels changed from **3 → 1** (spectrograms are single-channel)
- Pretrained weights used for transfer learning
- Custom classification head:
  - BatchNorm
  - Dropout
  - Linear layer → multi-label outputs

### Implementation

```python
import torch
import torch.nn as nn
import torchvision.models as models

class BirdResNet(nn.Module):
    def __init__(self, num_classes, dropout=0.2):
        super().__init__()

        self.model = models.resnet18(pretrained=True)

        # Modify first layer for 1-channel input
        self.model.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # Replace classification head
        self.model.fc = nn.Sequential(
            nn.BatchNorm1d(self.model.fc.in_features),
            nn.Dropout(p=dropout),
            nn.Linear(self.model.fc.in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)
