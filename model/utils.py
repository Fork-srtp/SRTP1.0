import torch

def sparse_dropout(x, keep_prob, noise_shape):
    # Dropout for sparse tensors
    random_tensor = keep_prob
    random_tensor += torch.rand(noise_shape).to(x.device)
    dropout_mask = torch.floor(random_tensor).byte()

    i = x._indices()
    v = x._values()
    i = i[:, dropout_mask]
    v = v[dropout_mask]

    pre_out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
    return pre_out * (1./keep_prob)

def DCG(a):
    return torch.sum(a[:,0] + torch.sum(a[:,1:]/torch.log2(torch.arange(2,a.size(dim=1)+1))))

def NDCG(y, y_pred, k=10):
    sorted, indices = torch.sort(y, descending=True, stable=True)
    sorted_pred, indices_pred = torch.sort(y_pred, descending=True, stable=True)

    sorted = sorted[:,:10]
    indices_pred = indices_pred[:,:10]
    
    for i in y.size(dim=0):
        if i == 0:
            sorted_pred = y[i][indices_pred[i]]
        else:
            sorted_pred = torch.cat((sorted_pred, y[i][indices_pred[i]]), 0)

    dcg = DCG(sorted_pred)
    idcg = DCG(sorted)

    ndcg = dcg / idcg
    return ndcg

def HR(y, y_pred, k=10):
    sorted, indices = torch.sort(y, descending=True, stable=True)
    sorted_pred, indices_pred = torch.sort(y_pred, descending=True, stable=True)

    indices = indices[:,:10]
    indices_pred = indices_pred[:,:10]

    count = 0

    for i in indices.size(dim=0):
        for item in indices[i]:
            if item in indices_pred[i]:
                count += 1

    total = indices.size(dim=0) * indices.size(dim=1)

    return count / total
