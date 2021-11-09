import torch
from model.model import Net
from torch import optim
from model.utils import NDCG, HR

def train(feature, adj, epochs=200):
    # feature = torch.sparse.FloatTensor(feature)
    feat_dim = feature.shape[1]
    num_features_nonzero = 0
    model = Net(feat_dim, feat_dim, num_features_nonzero)
    # model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.02)
    for epoch in range(epochs):
        out = model((feature, adj))

        loss = rating * torch.log(out) + (torch.ones_like(rating) - rating) * torch.log(torch.ones_like(out) - out)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(epoch, loss.item())

    model.eval()

    out = model((feature, adj))

    print("NDCG@10: ", NDCG(rating, out))
    print("HR@10: ", HR(rating, out))
