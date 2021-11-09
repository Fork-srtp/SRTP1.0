from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from datareader import Datareader
from nltk.tokenize import word_tokenize
import networkx as nx

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
        print(str, user)
        tagged_data.append(TaggedDocument(words=word_tokenize(str), tags=[user]))

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

    print(model.dv['A2UUEV4MYOJY66'])

    idset = [_key for _key in music_review_dict.keys()]
    itemset = []
    for _key in idset:
        itemset.extend([r[0] for r in music_review_dict[_key]])

    idset.extend(itemset)
    idset = list(set(idset))  # delete the repeated elements
    idstr2num = {}

    G = nx.Graph()
    for index, node in enumerate(idset):
        idstr2num[node] = index
        G.add_node(index, feature=model.dv[node])

    # add edge weight
    for key, value in music_rating_dict.items():
        G.add_weighted_edges_from([(idstr2num[key], idstr2num[items[0]], items[1]) for items in value])

    # print(G)

    return G.adj, G.feature
