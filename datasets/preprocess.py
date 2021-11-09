from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from datasets.datareader import Datareader
from nltk.tokenize import word_tokenize
import networkx as nx
import torch
import scipy.sparse as sp
import numpy as np

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)) # D
    d_inv_sqrt = np.power(rowsum, -0.5).flatten() # D^-0.5
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt) # D^-0.5
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo() # D^-0.5AD^0.5

def sparse_to_tuple(sparse_mx):
    """
    Convert sparse matrix to tuple representation.
    """
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx

def preprocess():
    dr = Datareader()
    music_rating_dict, music_review_dict, item_music_dict, \
            fashion_rating_dict, fashion_review_dict, item_fashion_dict = dr.read_data()
    tagged_data = []
    for user, reviews in music_review_dict.items():
        str = ""
        for each in reviews:
            str += " "
            str += each[1]
        tagged_data.append(TaggedDocument(words=word_tokenize(str), tags=[user]))

    for item, reviews in item_music_dict.items():
        str = ""
        for each in reviews:
            str += " "
            str += each[1]
        tagged_data.append(TaggedDocument(words=word_tokenize(str), tags=[item]))
        

    max_epochs = 100
    vec_size = 20
    alpha = 0.025

    model = Doc2Vec(dm=1,
                    alpha=alpha,
                    vector_size=vec_size,
                    min_alpha=0.00025,
                    min_count=1)

    model.build_vocab(tagged_data)

    for epoch in range(max_epochs):
        model.train(tagged_data,
                    total_examples=model.corpus_count,
                    epochs=model.epochs)
        # decrease the learning rate
        model.alpha -= 0.0002
        # fix the learning rate, no decay
        model.min_alpha = model.alpha
        if epoch % 10 == 0:
            print('iteration {0}'.format(epoch))

    # print(model.dv['A2UUEV4MYOJY66'])

    nodeset = [_key for _key in music_review_dict.keys()]
    nodeset += [_key for _key in item_music_dict.keys()]

    nodedict = {}

    G = nx.Graph()
    for index, node in enumerate(nodeset):
        nodedict[node] = index
        G.add_node(index, feature=model.dv[node])

    # add edge weight
    for key, value in music_rating_dict.items():
        G.add_weighted_edges_from([(nodedict[key], nodedict[items[0]], items[1]) for items in value])

    # print(G)
    nodelist = list(G.nodes(data=True))
    feature = []
    for each in nodelist:
        feature.append(each[1]['feature'])

    feature = torch.FloatTensor(feature)

    adj_matrix = nx.adjacency_matrix(G)
    adj_normalized = normalize_adj(adj_matrix + sp.eye(adj_matrix.shape[0]))
    adj_matrix = sparse_to_tuple(adj_normalized)
    i = torch.from_numpy(adj_matrix[0]).long()
    v = torch.from_numpy(adj_matrix[1])
    adj = torch.sparse.FloatTensor(i.t(), v, adj_matrix[2]).float()

    return adj, feature

adj, feat = preprocess()

print(1)