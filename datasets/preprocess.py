import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from datasets.datareader import Datareader
from nltk.tokenize import word_tokenize
import networkx as nx
import torch
import scipy.sparse as sp
import numpy as np

def preprocess():
    dr = Datareader()
    music_rating_dict, music_review_dict, item_music_dict, \
            fashion_rating_dict, fashion_review_dict, item_fashion_dict = dr.read_data()
    tagged_data = []
    usernum = 0
    for user, reviews in music_review_dict.items():
        usernum += 1
        str = ""
        for each in reviews:
            str += " "
            str += each[1]
        tagged_data.append(TaggedDocument(words=word_tokenize(str), tags=[user]))

    itemnum = 0
    for item, reviews in item_music_dict.items():
        itemnum += 1
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

    feature = torch.LongTensor(feature)

    adj_matrix = nx.adjacency_matrix(G)
    e = list(G.edges.data())
    adj = torch.zeros(usernum + itemnum, usernum + itemnum)

    for edge in e:
        adj[edge[0]][edge[1]] = edge[2]['weight']
        adj[edge[1]][edge[0]] = edge[2]['weight']
    adj = adj.type(torch.LongTensor)

    return adj, feature
