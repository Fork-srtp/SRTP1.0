import torch
import torch.nn as nn
import torch.nn.functional as F
from model.utils import sparse_dropout

class GraphConv(nn.Module):
    def __init__(self, input_dim, output_dim, num_features_nonzero,
                 is_sparse=False,
                 bias=False, 
                 dropout=0.,
                 featureless=False):
        super().__init__()

        self.is_sparse = is_sparse
        self.bias = bias
        self.dropout = dropout
        self.num_features_nonzero = num_features_nonzero
        self.featureless = featureless

        self.weight = nn.Parameter(torch.randn(input_dim, output_dim))
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, input):
        x, sup = input

        if self.training and self.is_sparse:
            x = sparse_dropout(x, 1 - self.dropout, self.num_features_nonzero)
        elif self.training:
            x = F.dropout(x, 1 - self.dropout)

        """Convolve"""
        if not self.featureless:
            if self.is_sparse:
                x = torch.sparse.mm(x, self.weight)
            else:
                x = torch.mm(x, self.weight)
        else:
            x = self.weight
        
        out = torch.sparse.mm(sup, x)

        if self.bias is not None:
            out += self.bias

        out = F.relu(out)
        return out, sup
        