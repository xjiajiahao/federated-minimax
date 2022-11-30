import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP_784_200_200_10(nn.Module):

    def __init__(self):
        super(MLP_784_200_200_10, self).__init__()
        # kernel
        self.fc0 = nn.Linear(784, 200)
        self.fc1 = nn.Linear(200, 200)
        self.fc2 = nn.Linear(200, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
