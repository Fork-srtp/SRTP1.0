from datareader import Datareader
from torch.utils.data import Dataset

class RecDataset(Dataset):
    def __init__(self, datareader = Datareader()):      
        self.music_rating_dict, self.music_review_dict, \
            self.item_music_dict, self.fashion_rating_dict, \
                self.fashion_review_dict, self.item_fashion_dict = datareader.read_data()

    
    def __len__(self):
        return len(self.item_music_dict.keys()) + len(self.music_rating_dict.keys())

    def __getitem__(self, idx):
        #TODO
        return 