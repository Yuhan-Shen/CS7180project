#!/usr/bin/python3.7

import numpy as np
import random
import os
import time

# self.features[video]: the feature array of the given video (dimension x frames)
# self.transcrip[video]: the transcript (as label indices) for each video
# self.input_dimension: dimension of video features
# self.n_classes: number of classes
class SemiDataset(object):

    def __init__(self, base_path, task, video_list, un_video_list, label2index, shuffle = False, batch_size=1, un_batch_size=None):
        self.features = dict()
        self.un_features = dict()
        self.transcript = dict()
        self.shuffle = shuffle
        self.idx = 0
        self.un_idx = 0
        self.batch_size = batch_size
        self.un_batch_size = un_batch_size if type(un_batch_size) == int else batch_size
        # read features for each video
        for video in video_list:
            # video features
            self.features[video] = np.load(base_path + '/features/' +  task + '/' + video + '.npy')[:,::15]
            # transcript
            with open(base_path + '/transcripts/' + task + '/' + video + '.txt') as f:
                self.transcript[video] = [ label2index[line] for line in f.read().split('\n')[0:-1] ]

        # read features for each un video
        for video in un_video_list:
            # video features
            self.un_features[video] = np.load(base_path + '/features/' +  task + '/' + video + '.npy')[:,::15]

        # selectors for random shuffling
        self.selectors = list(self.features.keys())
        self.un_selectors = list(self.un_features.keys())
        # print(self.selectors)
        if self.shuffle:
            random.shuffle(self.selectors)
            random.shuffle(self.un_selectors)
        # set input dimension and number of classes
        self.input_dimension = list(self.features.values())[0].shape[0]
        self.n_classes = len(label2index)
        self.n_videos = len(video_list)
        self.n_un_videos = len(un_video_list)

    def videos(self):
        return list(self.features.keys())

    def __len__(self):
        return len(self.features)

    def __iter__(self):
        return self

    def __next__(self):
        if self.idx == self.n_videos:
            self.idx = 0
            if self.shuffle:
                random.shuffle(self.selectors)
            raise StopIteration
        if self.un_idx == self.n_un_videos:
            self.un_idx = 0
            if self.shuffle:
                random.shuffle(self.un_selectors)

        next_idx = min(self.idx+self.batch_size, self.n_videos)
        video_idx = [self.selectors[i] for i in range(self.idx, next_idx)]
        features = [self.features[i] for i in video_idx]
        transcripts = [self.transcript[i] for i in video_idx]
        self.idx = next_idx

        next_un_idx = min(self.un_idx+self.un_batch_size, self.n_un_videos)
        un_video_idx = [self.un_selectors[i] for i in range(self.un_idx, next_un_idx)]
        un_features = [self.un_features[i] for i in un_video_idx]
        self.un_idx = next_un_idx
        return features, transcripts, un_features

    def get(self):
        #return next(self)
        try:
            return next(self)
        except StopIteration:
            return self.get()

class Dataset(object):

    def __init__(self, base_path, task, video_list, label2index, shuffle = False):
        self.features = dict()
        self.transcript = dict()
        self.shuffle = shuffle
        self.idx = 0
        # read features for each video
        for video in video_list:
            # video features
            self.features[video] = np.load(base_path + '/features/' +  task + '/' + video + '.npy')[:,::15]
            # transcript
            with open(base_path + '/transcripts/' + task + '/' + video + '.txt') as f:
                self.transcript[video] = [ label2index[line] for line in f.read().split('\n')[0:-1] ]
        # selectors for random shuffling
        self.selectors = list(self.features.keys())
        # print(self.selectors)
        if self.shuffle:
            random.shuffle(self.selectors)
        # set input dimension and number of classes
        self.input_dimension = list(self.features.values())[0].shape[0]
        self.n_classes = len(label2index)

    def videos(self):
        return list(self.features.keys())

    def __len__(self):
        return len(self.features)

    def __iter__(self):
        return self

    def __next__(self):
        if self.idx == len(self):
            self.idx = 0
            if self.shuffle:
                random.shuffle(self.selectors)
            raise StopIteration
        else:
            video = self.selectors[self.idx]
            self.idx += 1
            return self.features[video], self.transcript[video]

    def get(self):
        try:
            return next(self)
        except StopIteration:
            return self.get()

