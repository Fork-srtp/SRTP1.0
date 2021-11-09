import model.model.Net as Net
from torch import optim
from utils import NDCG, HR

def train(feature, adj, epochs=epochs):
    model = Net(input_dim, output_dim, num_features_nonzero)
    model.to(device)
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
