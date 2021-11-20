import torch
import torch.nn as nn
import torch.nn.functional as F
from model.GraphConv import GraphConv


class GCN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.layer1 = GraphConv(input_dim, 16, dropout=0.5)
        self.layer2 = GraphConv(16, output_dim, dropout=0.5)

    def forward(self, A, features):
        x = self.layer1(A, features)
        x = self.layer2(A, x)

        return x
