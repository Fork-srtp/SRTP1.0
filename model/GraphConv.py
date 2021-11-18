import torch
import torch.nn as nn
import torch.nn.functional as F
from model.utils import sparse_dropout

class GraphConv(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.):
        super().__init__()

        self.dropout = dropout

        self.weight = nn.Parameter(torch.randn(input_dim, output_dim), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(output_dim), requires_grad=True)

    def forward(self, A, features):
        I = torch.eye(A.shape[0])
        A_hat = A + I

        D = torch.sum(A_hat, axis=0)
        D = torch.diag(D)
        D_inv = torch.inverse(D)

        A_hat = torch.mm(torch.mm(D_inv, A_hat), D_inv)

        features = F.dropout(features, 1 - self.dropout)

        aggregate = torch.mm(A_hat, features)

        propagate = torch.mm(aggregate, self.weight) + self.bias

        out = F.relu(propagate)
        return out
        