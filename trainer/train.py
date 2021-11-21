import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from model.model import Net
from torch import optim
from model.utils import NDCG, HR

def train(feature, adj, usernum, itemnum, epochs=200):
    feat_dim = feature.shape[1]
    model = Net(feat_dim, feat_dim, usernum, itemnum)
    # model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.02)
    rating = adj[0:usernum, usernum:usernum+itemnum]
    model.train()
    for epoch in range(epochs):
        out = model(adj, feature)

        totloss = 0

        for i in range(itemnum):
            y = rating[:,i]
            y_hat = out[:,i]
            loss = -(y * torch.log(y_hat) + (torch.ones_like(y) - y) * torch.log(torch.ones_like(y_hat) - y_hat))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            totloss += torch.sum(loss)

        # loss = -(rating * torch.log(out) + (torch.ones_like(rating) - rating) * torch.log(torch.ones_like(out) - out))

        # optimizer.zero_grad()
        # loss.backward(rating.clone().detach())
        # optimizer.step()

        if (epoch + 1) % 10 == 0:

            # print("epoch:", epoch + 1, "loss:", torch.sum(torch.sum(loss, dim=0)).item() / (loss.size(dim=0) * loss.size(dim=1)))
            print("epoch:", epoch + 1, "loss:", totloss)

    model.eval()

    out = model(adj, feature)

    K = 10
    print("NDCG@10: ", NDCG(rating, out, K))

    sorted, indices = torch.sort(rating, descending=True, stable=True)
    for i in range(indices.size(dim=0)):
        for j in range(indices.size(dim=1)):
            if j < K:
                rating[i][indices[i][j]] = 1
            else:
                rating[i][indices[i][j]] = 0

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


