import torch

def DCG(a):
    return torch.sum(a[:,0] + torch.sum(a[:,1:]/torch.log2(torch.arange(2,a.size(dim=1)+1))))

def NDCG(y, y_pred, k=10):
    dcg = DCG(y_pred)
    idcg = DCG(y)

    ndcg = dcg / idcg
    return ndcg

def HR(y, y_pred, k=10):
    # sorted, indices = torch.sort(y, descending=True, stable=True)
    # sorted_pred, indices_pred = torch.sort(y_pred, descending=True, stable=True)
    #
    # indices = indices[:,:k]
    # indices_pred = indices_pred[:,:k]

    hit = torch.bitwise_and(y, y_pred)
    hit = torch.sum(torch.sum(hit, dim=0))

    total = y.size(dim=0) * 10

    return hit / total
