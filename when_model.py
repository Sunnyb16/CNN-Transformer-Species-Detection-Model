import torch
import torch.nn as nn


class WhenNet(nn.Module):
    def __init__(self, n_mels=128, conv_channels=(32, 64, 128), gru_hidden_size=128):
        super().__init__()

        c1, c2, c3 = conv_channels

        self.features = nn.Sequential(
            nn.Conv2d(1, c1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            nn.Conv2d(c1, c2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            nn.Conv2d(c2, c3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
        )

        reduced_mels = max(1, n_mels // 4)
        self.gru = nn.GRU(
            input_size=c3 * reduced_mels,
            hidden_size=gru_hidden_size,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = nn.Linear(gru_hidden_size * 2, 1)

    def forward(self, x):
        x = self.features(x)
        batch_size, channels, freq_bins, time_steps = x.shape
        x = x.permute(0, 3, 1, 2).contiguous().view(batch_size, time_steps, channels * freq_bins)
        x, _ = self.gru(x)
        logits = self.classifier(x).squeeze(-1)
        return logits
