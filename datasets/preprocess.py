from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from datareader import Datareader
from nltk.tokenize import word_tokenize

if __name__ == '__main__':
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
