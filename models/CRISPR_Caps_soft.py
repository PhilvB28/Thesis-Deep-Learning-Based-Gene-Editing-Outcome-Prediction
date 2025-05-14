import torch
import torch.nn as nn
import torch.nn.functional as F

class CapsuleLayer(nn.Module):
    def __init__(self, num_capsules, in_dim, out_dim, num_routing=3):
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

class CapsNetTask(nn.Module):
    def __init__(self, input_channels=4, sequence_length=60, in_dim=16, out_dim=16, num_routing=2):
        super(CapsNetTask, self).__init__()
        # Shared convolutional feature extractors
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=3, stride=2, padding=1)
        self.bn1   = nn.BatchNorm1d(32)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2)
        self.bn2   = nn.BatchNorm1d(64)

        # Primary capsules: convert conv features to 32 capsules of dim 8
        self.primary_caps = nn.Conv1d(64, 32 * 8, kernel_size=15, stride=15, padding=0)

        # Intermediate capsule layer: 8 Capsules
        self.intermediate_caps = CapsuleLayer(
            num_capsules=8,
            in_dim=8,
            out_dim=out_dim,
            num_routing=num_routing
        )

        # Final digit capsule: 8 -> 1 capsule of dimension out_dim
        self.digit_caps = CapsuleLayer(
            num_capsules=1,
            in_dim=out_dim,
            out_dim=out_dim,
            num_routing=num_routing
        )

        # Task-specific head: maps final capsule to a scalar output
        self.head = nn.Sequential(
            nn.Linear(out_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [batch, input_channels, sequence_length]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        # Primary capsule convolution and reshape
        x = self.primary_caps(x).squeeze(-1)      # [batch, 256]
        x = x.view(x.size(0), 32, 8)              # [batch, 32 capsules, 8-dim]

        # Intermediate capsule transformation
        x = self.intermediate_caps(x)             # [batch, 8 capsules, out_dim]

        # Final digit capsule
        x = self.digit_caps(x)                    # [batch, 1 capsule, out_dim]
        x = x.squeeze(1)                          # [batch, out_dim]

        # Task-specific prediction
        return self.head(x)                       # [batch, 1]


class CrisprCaps_soft(nn.Module): #CapsNetRegressor_SoftSharing_3layer
    def __init__(self, input_channels=4, sequence_length=60, in_dim=16, out_dim=16, num_routing=3):
        super(CrisprCaps_soft, self).__init__()
        # Instantiate one CapsNetTask per outcome (6 tasks)
        self.tasks = nn.ModuleList([
            CapsNetTask(input_channels, sequence_length, in_dim, out_dim, num_routing)
            for _ in range(6)
        ])

    def forward(self, x):
        # Independent task network forwarding
        outputs = [task(x) for task in self.tasks]
        return torch.cat(outputs, dim=1)

    def compute_soft_sharing_loss(self, p=2):
        """
        Compute soft parameter sharing regularization:
        encourages weights of different task nets to be similar.
        """
        reg_loss = 0.0
        num_tasks = len(self.tasks)
        for i in range(num_tasks):
            for j in range(i + 1, num_tasks):
                for w_i, w_j in zip(self.tasks[i].parameters(), self.tasks[j].parameters()):
                    if p == 2:
                        reg_loss += torch.sum((w_i - w_j) ** 2)
                    else:
                        reg_loss += torch.norm(w_i - w_j, p=p)
        # Normalize by number of pairs
        num_pairs = num_tasks * (num_tasks - 1) / 2
        return reg_loss / num_pairs