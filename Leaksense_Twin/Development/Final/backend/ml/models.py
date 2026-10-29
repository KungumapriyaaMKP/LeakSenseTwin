"""
LeakSense Twin — Neural Network Model
LeakNet: single 6-class network (0=Healthy, 1-5=Zone). Binary leak detection
and confidence are derived from this same output (argmax != 0, 1 - P(healthy))
rather than trained as a separate network — see ml/benchmark.py for the
comparison data this design is based on.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LeakNet(nn.Module):
    """
    Unified leak detection + zone localization network.
    Classes: 0=Healthy, 1=Zone 1, 2=Zone 2, 3=Zone 3, 4=Zone 4, 5=Zone 5

    Returns raw logits (not softmax) — use F.softmax(model(x), dim=1) at
    inference time, and nn.CrossEntropyLoss directly during training.
    """

    def __init__(self, input_dim: int = 31, n_classes: int = 6):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, n_classes)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        return self.fc3(x)
