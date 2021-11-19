import torch
from model.model import Net
from torch import optim
from model.utils import NDCG, HR

def train(feature, adj, epochs=200):
    feat_dim = feature.shape[1]
    num_features_nonzero = 0
    model = Net(feat_dim, feat_dim, num_features_nonzero)
    # model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.02)
    rating = adj[0:1, 0:7]
    sorted, indices = torch.sort(rating, descending=True, stable=True)
    for i in range(indices.size(dim=0)):
        for j in range(indices.size(dim=1)):
            if j < 3:
                rating[i][indices[i][j]] = 1
            else:
                rating[i][indices[i][j]] = 0
    for epoch in range(epochs):
        out = model(adj, feature)

        sorted, indices = torch.sort(out, descending=True, stable=True)

        with torch.no_grad():
            for i in range(indices.size(dim=0)):
                for j in range(indices.size(dim=1)):
                    if j < 3:
                        out[i][indices[i][j]] = 1
                    else:
                        out[i][indices[i][j]] = 0

        loss = rating * torch.log(out) + (torch.ones_like(rating) - rating) * torch.log(torch.ones_like(out) - out)

        optimizer.zero_grad()
        loss.backward(loss.clone().detach())
        optimizer.step()

        if epoch % 10 == 0:
            print(epoch, loss)

    model.eval()

    out = model(adj, feature)
    with torch.no_grad():
        for i in range(indices.size(dim=0)):
            for j in range(indices.size(dim=1)):
                if j < 3:
                    out[i][indices[i][j]] = 1
                else:
                    out[i][indices[i][j]] = 0
    
    print("HR@10: ", HR(rating, out))
    print("NDCG@10: ", NDCG(rating, out))

