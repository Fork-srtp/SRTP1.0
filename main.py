import torch
from trainer.train import train
from datasets.preprocess import preprocess

if __name__ == '__main__':
    adj, feature, usernum, itemnum = preprocess()
    epochs = 1
    
    train(feature, adj, usernum, itemnum)
