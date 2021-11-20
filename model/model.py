import torch
import torch.nn as nn
import torch.nn.functional as F
from model.GCN import GCN
from model.mlp import MLP
from sklearn.metrics.pairwise import cosine_similarity


class Net(nn.Module):
    def __init__(self, input_dim, output_dim, usernum, itemnum):
        super().__init__()

        self.GNN = GCN(input_dim, output_dim)
        self.umlp = MLP(input_dim)
        self.imlp = MLP(input_dim)
        self.usernum = usernum
        self.itemnum = itemnum

    def forward(self, A, features):
        x = self.GNN(A, features)
        m, n = self.usernum, self.itemnum
        user, item = x.split([m, n], dim=0)
        user = self.umlp(user)
        item = self.imlp(item)
        out = torch.tensor(cosine_similarity(user.detach(), item.detach()), requires_grad=True)
        return out
