import torch

def sparse_dropout(x, keep_prob, noise_shape):
    # Dropout for sparse tensors
    random_tensor = keep_prob
    random_tensor += torch.rand(noise_shape)
    dropout_mask = torch.floor(random_tensor).byte()

    i = x._indices()
    v = x._values()
    i = i[:, dropout_mask]
    v = v[dropout_mask]

    pre_out = torch.sparse.FloatTensor(i, v, x.shape)
    return pre_out * (1./keep_prob)

def DCG(a):
    return torch.sum(a[:,0] + torch.sum(a[:,1:]/torch.log2(torch.arange(2,a.size(dim=1)+1))))

def NDCG(y, y_pred, k=10):
    dcg = DCG(y_pred)
    idcg = DCG(y)

    ndcg = dcg / idcg
    return ndcg

def HR(y, y_pred, k=10):
    sorted, indices = torch.sort(y, descending=True, stable=True)
    sorted_pred, indices_pred = torch.sort(y_pred, descending=True, stable=True)

    indices = indices[:,:3]
    indices_pred = indices_pred[:,:3]

    count = 0

    for i in range(indices.size(dim=0)):
        for item in indices[i]:
            if item in indices_pred[i]:
                count += 1

    total = indices.size(dim=0) * indices.size(dim=1)

    return count / total
