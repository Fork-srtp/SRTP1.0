import torch
import torch.nn as nn
import torch.nn.functional as F
from model.GCN import GCN
from model.mlp import MLP
from sklearn.metrics.pairwise import cosine_similarity


class Net(nn.Module):
    def __init__(self, input_dim, output_dim, num_features_nonzero):
        super().__init__()

        self.GNN = GCN(input_dim, output_dim, num_features_nonzero)
        self.umlp = MLP(input_dim)
        self.imlp = MLP(input_dim)

    def forward(self, A, features):
        x = self.GNN(A, features)
        m, n = 1, 7
        user, item = x.split([m, n], dim=0)
        user = self.umlp(user)
        item = self.imlp(item)
        out = torch.tensor(cosine_similarity(user.detach(), item.detach()), requires_grad=True)
        return out
