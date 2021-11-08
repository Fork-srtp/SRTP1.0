import model.model.Net as Net
from torch import optim

def train(epochs=epochs):
    # TODO
    # Data

    model = Net(input_dim, output_dim, num_features_nonzero)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.02)
    for epoch in range(epochs):
        out = model((feature, adj))

        loss = rating * torch.log(out) + (torch.ones_like(rating) - rating) * torch.log(torch.ones_like(out) - out)

        # acc = ?

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(epoch, loss.item())

    model.eval()
