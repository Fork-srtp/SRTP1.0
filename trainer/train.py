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

        loss = "fds"

        acc = "hyh"

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(epoch, loss.item(), acc.item())

    model.eval()
