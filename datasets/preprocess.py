import gensim

from gensim.models.doc2vec import Doc2Vec

from datareader import Datareader




if __name__ == '__main__':
    TaggededDocument = gensim.models.doc2vec.TaggedDocument
    dr = Datareader()
    music_rating_dict, music_review_dict, item_music_dict, fashion_rating_dict, fashion_review_dict, item_fashion_dict = dr.read_data()
    
    # model = Doc2Vec()
    print(len(music_review_dict))
    a = 1
