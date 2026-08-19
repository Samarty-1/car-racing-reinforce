import torch
import torch.nn as nn


class PolicyNet(nn.Module):
    """Looks at the 96x96 track image and outputs a probability for each possible move."""

    def __init__(self, num_actions: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 96, 96)
            flat_size = self.conv(dummy).numel()

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions),
        )

    def forward(self, x):
        return self.head(self.conv(x))
