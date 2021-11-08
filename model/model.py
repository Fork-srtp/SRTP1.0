import torch
import torch.nn as nn
import torch.nn.functional as F
from GCN import GCN
from mlp import MLP

class Net(nn.Module):
    def __init__(self, input_dim, output_dim, num_features_nonzero):
        super().__init__()

        self.model = nn.Sequential(
                        GCN(input_dim, output_dim, num_features_nonzero),
                        MLP(input_dim))

    def forward(self, x):
        x = model(x)
        return x
