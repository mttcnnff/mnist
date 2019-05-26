import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearClassifier(nn.Module):
    def __init__(self):
        super(LinearClassifier, self).__init__()
        # define layers of net
        self.fc1 = nn.Linear(400, 10)

    def forward(self, x):
        # define forward prop
        a1 = self.fc1(x)
        return F.log_softmax(a1, dim=1)
