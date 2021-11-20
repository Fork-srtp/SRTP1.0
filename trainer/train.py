import torch
import torch.nn as nn
from model.model import Net
from torch import optim
from model.utils import NDCG, HR

def train(feature, adj, usernum, itemnum, epochs=200):
    feat_dim = feature.shape[1]
    num_features_nonzero = 0
    model = Net(feat_dim, feat_dim, usernum, itemnum)
    # model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.02)
    rating = adj[0:usernum, usernum:usernum+itemnum]

    # lossF = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        out = model(adj, feature)

        # sorted, indices = torch.sort(out, descending=True, stable=True)
        #
        # with torch.no_grad():
        #     for i in range(indices.size(dim=0)):
        #         for j in range(indices.size(dim=1)):
        #             if j < K:
        #                 out[i][indices[i][j]] = 1
        #             else:
        #                 out[i][indices[i][j]] = 0

        loss = -(rating * torch.log(out) + (torch.ones_like(rating) - rating) * torch.log(torch.ones_like(out) - out))
        # loss = lossF(out, rating)

        optimizer.zero_grad()
        loss.backward(loss.clone().detach())
        optimizer.step()

        if epoch % 10 == 0:
            print(epoch, loss)

    model.eval()

    sorted, indices = torch.sort(rating, descending=True, stable=True)
    K = 10
    for i in range(indices.size(dim=0)):
        for j in range(indices.size(dim=1)):
            if j < K:
                rating[i][indices[i][j]] = 1
            else:
                rating[i][indices[i][j]] = 0

    out = model(adj, feature)
    sorted, indices = torch.sort(out, descending=True, stable=True)
    with torch.no_grad():
        for i in range(indices.size(dim=0)):
            for j in range(indices.size(dim=1)):
                if j < K:
                    out[i][indices[i][j]] = 1
                else:
                    out[i][indices[i][j]] = 0
    out = out.clone().detach()
    out = out.type(torch.LongTensor)
    rating = rating.type(torch.LongTensor)
    print("HR@10: ", HR(rating, out, K))
    print("NDCG@10: ", NDCG(rating, out))

