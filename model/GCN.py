import torch
import torch.nn as nn
import torch.nn.functional as F
from GraphConv import GraphConv

class GCN(nn.Module):
    def __init__(self, input_dim, output_dim, num_features_nonzero):
        super().__init__()

        self.layers = nn.Sequential(
                            GraphConv(input_dim, 16, num_features_nonzero,
                                      is_sparse=True,
                                      dropout=0.5),
                            GraphConv(16, output_dim, num_features_nonzero,
                                      is_sparse=False,
                                      dropout=0.5)
                        )

    def forward(self, input):
        x, sup = input

        x = self.layers((x, sup))

        return x