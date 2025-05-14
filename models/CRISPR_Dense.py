import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(DenseLayer, self).__init__()
        self.bn = nn.BatchNorm1d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv1d(in_channels, growth_rate, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.conv(self.relu(self.bn(x)))
        return torch.cat([x, out], dim=1)

class DenseBlock(nn.Module):
    def __init__(self, num_layers, in_channels, growth_rate):
        super(DenseBlock, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(DenseLayer(in_channels + i * growth_rate, growth_rate))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionLayer, self).__init__()
        self.bn = nn.BatchNorm1d(in_channels)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.pool = nn.AvgPool1d(kernel_size=2)

    def forward(self, x):
        x = self.conv(F.relu(self.bn(x)))
        return self.pool(x)

class DenseNetRegressor_6_heads(nn.Module):
    def __init__(self, input_channels=4, sequence_length=60, growth_rate=16):
        super(DenseNetRegressor_6_heads, self).__init__()

        # Initial convolution
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=3, padding=1)

        # Dense Block 1
        self.block1 = DenseBlock(num_layers=3, in_channels=32, growth_rate=growth_rate)
        self.trans1 = TransitionLayer(in_channels=32 + 3 * growth_rate, out_channels=64)

        # Dense Block 2
        self.block2 = DenseBlock(num_layers=3, in_channels=64, growth_rate=growth_rate)
        self.trans2 = TransitionLayer(in_channels=64 + 3 * growth_rate, out_channels=128)

        # Global average pooling to flatten time axis
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Regression heads
        self.regression_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid()
            ) for _ in range(6)
        ])

    def forward(self, x):
        # x: [batch_size, input_channels, sequence_length]
        x = self.conv1(x)
        x = self.trans1(self.block1(x))
        x = self.trans2(self.block2(x))
        x = self.global_pool(x).squeeze(-1)

        outputs = [head(x) for head in self.regression_heads]
        outputs = torch.cat(outputs, dim=1)  # -> [B, 6]
        return outputs

