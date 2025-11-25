import torch.nn as nn
import torch
from torch.utils.data import DataLoader, TensorDataset
class DiffusionMLP(nn.Module):
        def __init__(self, input_dim=3, hidden_dim=64, num_layers=4):
            super().__init__()
            layers = []
            # First layer from input_dim to hidden_dim
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.SiLU())
            # Hidden layers
            for _ in range(num_layers - 1):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.SiLU())
            # Output layer
            layers.append(nn.Linear(hidden_dim, 2))
            self.net = nn.Sequential(*layers)
            
        def forward(self, x, t):
            # x: [B, 2], t: [B]
            t = t.unsqueeze(-1)  # [B, 1]
            x_t = torch.cat([x, t], dim=-1)  # [B, 3]
            return self.net(x_t) 