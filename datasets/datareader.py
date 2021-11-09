from torch.utils.data import Dataset
from tqdm import tqdm
import json
import torch
import random

class Datareader:
    def __init__(self):
        # source domain
        self.music_rating_dict = {}
        self.music_review_dict = {}
        self.item_music_dict = {}

        # target domain
        self.fashion_rating_dict = {}
        self.fashion_review_dict = {}
        self.item_fashion_dict = {}

    def read_data(self):
        # need_col = ['reviewerID', 'asin', 'summary', 'overall']

        music_dict = {}
        with open("datasets/Digital_Music_5.json") as fp:
            for music in fp.readlines():
                line = json.loads(music)
                reviewer = line['reviewerID']
                item = line['asin']
                if 'summary' not in line.keys():
                    continue
                review = line['summary']
                rating = line['overall']
                if reviewer not in music_dict:
                    music_dict[reviewer] = []
                music_dict[reviewer] += [[item, review, rating]]
            print(len(music_dict))

        fashion_dict = {}
        with open("datasets/AMAZON_FASHION_5.json") as fp:
            for fashion in fp.readlines():
                line = json.loads(fashion)
                reviewer = line['reviewerID']
                item = line['asin']
                review = line['summary']
                rating = line['overall']
                if reviewer not in fashion_dict:
                    fashion_dict[reviewer] = []
                fashion_dict[reviewer] += [[item, review, rating]]

        # data filter

        # filter inactive users
        temp_keys = list(music_dict.keys())
        for reviewer in tqdm(temp_keys):
            if len(music_dict[reviewer]) < 5:
                music_dict.pop(reviewer)

        # for reviewer in tqdm(fashion_dict):
        #     if len(fashion_dict[reviewer]) < 5:
        #         fashion_dict.pop(reviewer)

        # filter uncommon users
        temp_keys = list(music_dict.keys())
        for reviewer in tqdm(temp_keys):
            if reviewer not in fashion_dict:
                music_dict.pop(reviewer)

        temp_keys = list(fashion_dict.keys())
        for reviewer in tqdm(temp_keys):
            if reviewer not in music_dict:
                fashion_dict.pop(reviewer)

        for reviewer in tqdm(music_dict.keys()):
            for each in music_dict[reviewer]:
                item = each[0]
                review = each[1]
                rating = each[2]
                if reviewer not in self.music_rating_dict:
                    self.music_rating_dict[reviewer] = []
                    self.music_review_dict[reviewer] = []
                self.music_review_dict[reviewer] += [[item, review]]
                self.music_rating_dict[reviewer] += [[item, rating]]


        for reviewer in tqdm(fashion_dict.keys()):
            for each in fashion_dict[reviewer]:
                item = each[0]
                review = each[1]
                rating = each[2]
                if reviewer not in self.fashion_rating_dict:
                    self.fashion_rating_dict[reviewer] = []
                    self.fashion_review_dict[reviewer] = []
                self.fashion_review_dict[reviewer] += [[item, review]]
                self.fashion_rating_dict[reviewer] += [[item, rating]]

        print(self.music_review_dict)
        print('user:', len(self.music_review_dict.keys()))

        # construct item dictionary

        for reviewer in tqdm(music_dict.keys()):
            for each in music_dict[reviewer]:
                if each[0] not in self.item_music_dict:
                    self.item_music_dict[each[0]] = []
                self.item_music_dict[each[0]] += [[reviewer, each[1]]]

        for reviewer in tqdm(fashion_dict.keys()):
            for each in fashion_dict[reviewer]:
                if each[0] not in self.item_fashion_dict:
                    self.item_fashion_dict[each[0]] = []
                self.item_fashion_dict[each[0]] += [[reviewer, each[1]]]

        print("music_dict len:", len(music_dict), "fashion_dict len:", len(fashion_dict))
        return self.music_rating_dict, self.music_review_dict, self.item_music_dict, \
            self.fashion_rating_dict, self.fashion_review_dict, self.item_fashion_dict
