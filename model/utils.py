import torch

def DCG(a):

    return torch.sum(a[:,0] + torch.sum(a[:,1:]/torch.log2(torch.arange(2,a.size(dim=1)+1))))

def NDCG(y, y_pred, k=10):
    sorted, indices = torch.sort(y_pred, descending=True, stable=True)
    pred = torch.zeros(0)
    for i in range(y_pred.size(dim=0)):
        pred = torch.cat((pred, y[i][indices[i][0:k]].view(1,-1)), dim=0)

    sorted, indices = torch.sort(y, descending=True, stable=True)
    ideal = torch.zeros(0)
    for i in range(y_pred.size(dim=0)):
        ideal = torch.cat((ideal, y[i][indices[i][0:k]].view(1,-1)), dim=0)

    dcg = DCG(pred)
    idcg = DCG(ideal)

    ndcg = dcg / idcg
    return ndcg

def HR(y, y_pred, k=10):
    hit = torch.bitwise_and(y, y_pred)
    hit = torch.sum(torch.sum(hit, dim=0))

    total = y.size(dim=0) * 10

    return hit / total
