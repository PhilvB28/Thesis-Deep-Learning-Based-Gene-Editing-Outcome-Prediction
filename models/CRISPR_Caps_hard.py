import torch
import torch.nn as nn
import torch.nn.functional as F

class CapsuleLayer(nn.Module):
    def __init__(self, num_capsules, in_dim, out_dim, num_routing):
        super(CapsuleLayer, self).__init__()
        self.num_capsules = num_capsules
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_routing = num_routing

        # Weight matrix for all output capsules
        self.W = nn.Parameter(torch.randn(1, num_capsules, in_dim, out_dim))

    def forward(self, x):
        batch_size = x.size(0)
        num_primary_caps = x.size(1)

        # Expand x for proper multiplication:
        x = x.unsqueeze(2)  # [B, num_primary_caps, 1, in_dim]

        # Repeat the weight matrix for each sample in the batch:
        W = self.W.repeat(batch_size, 1, 1, 1)  # [batch_size, num_capsules, in_dim, out_dim]
        W = W.unsqueeze(1)  # [batch_size, 1, num_capsules, in_dim, out_dim]

        # Prepare x for multiplication:
        x = x.unsqueeze(-1)  # [batch_size, num_primary_caps, 1, in_dim, 1]
        x = x.transpose(-2, -1)  # [batch_size, num_primary_caps, 1, 1, in_dim]

        # Multiplication
        u_hat = torch.matmul(x, W)  # [batch_size, num_primary_caps, num_capsules, 1, out_dim]
        u_hat = u_hat.squeeze(-2)   # [batch_size, num_primary_caps, num_capsules, out_dim]

        # Initialize routing logits:
        b_ij = torch.zeros(batch_size, num_primary_caps, self.num_capsules, device=x.device)

        # Dynamic routing:
        for routing_iteration in range(self.num_routing):
            c_ij = F.softmax(b_ij, dim=2)  # coupling coefficients
            s_j = (c_ij.unsqueeze(-1) * u_hat).sum(dim=1)
            v_j = self.squash(s_j)
            if routing_iteration < self.num_routing - 1:
                b_ij = b_ij + (u_hat * v_j.unsqueeze(1)).sum(dim=-1).detach()

        return v_j  # [batch_size, num_capsules, out_dim]

    @staticmethod
    def squash(x):
        squared_norm = (x ** 2).sum(dim=-1, keepdim=True)
        scale = squared_norm / (1 + squared_norm) / torch.sqrt(squared_norm + 1e-8)
        return scale * x

class CrisprCaps_hard(nn.Module):  #CapsNetRegressor_HardSharing_3_layer
    def __init__(self, input_channels=4, sequence_length=60, in_dim=8, out_dim=16, num_routing=2):
        super(CrisprCaps_hard, self).__init__()

        # Convolutional feature extractor using strided convolutions
        # Sequence length: 60 -> 30 -> 15
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=3, stride=2, padding=1)  # 60 -> 30
        self.bn1 = nn.BatchNorm1d(32)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2)  # 30 -> 15
        self.bn2 = nn.BatchNorm1d(64)

        # Primary Capsule Layer: 32 capsules of dimension 8
        self.primary_caps = nn.Conv1d(64, 32 * 8, kernel_size=15, stride=15)

        self.intermediate_caps = CapsuleLayer(
            num_capsules=8,
            in_dim=8,
            out_dim=out_dim,
            num_routing=num_routing
        )

        # Digit Capsule Layer: 6 output capsules
        self.digit_caps = CapsuleLayer(num_capsules=6, in_dim=out_dim, out_dim=out_dim, num_routing=num_routing)

        # Regression heads: one small MLP per capsule
        self.regression_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(out_dim, 8),
                nn.ReLU(),
                nn.Linear(8, 1),
                nn.Sigmoid()
            )
            for _ in range(6)
        ])

    def forward(self, x):
        # x: [batch_size, input_channels, sequence_length]
        x = F.relu(self.bn1(self.conv1(x)))  # -> [batch_size, 32, 30]
        x = F.relu(self.bn2(self.conv2(x)))  # -> [batch_size, 64, 15]

        # Primary capsules
        x = self.primary_caps(x)  # -> [batch_size, 256, 1]
        x = x.squeeze(-1)  # -> [batch_size, 256]
        x = x.view(x.size(0), 32, 8)  # -> [batch_size, 32, 8]

        # Intermediate Capsule
        x = self.intermediate_caps(x)

        # Digit capsules (dynamic routing)
        x = self.digit_caps(x)  # -> [batch_size, 6, 16]

        # Apply regression head to each capsule
        outputs = [head(x[:, i, :]) for i, head in enumerate(self.regression_heads)]
        outputs = torch.cat(outputs, dim=1)  # -> [batch_size, 6]
        return outputs