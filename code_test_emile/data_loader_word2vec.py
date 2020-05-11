#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 13:59:07 2020

@author: emile
"""
from nltk.corpus import stopwords
import numpy as np
from torch.utils.data import Dataset
import torch
import pickle
import os

from txt_preprocessing import get_doc_sentences
import config
from utils import get_size_of_torch, get_size_of_numpy

CAP_PER_IMAGE = 5

def load_flickr_captions(image_nrs):
    stop_words = set(stopwords.words('english'))
    im2idx = dict(zip(image_nrs, range(len(image_nrs))))
    im2captions = {}
    with open(config.CAPTIONS_PATH, 'r') as f:
        for line in f:
            line = line.strip().lower().split()
            im = line[0].split('.')[0]
            if im in image_nrs:
                if im not in im2captions:
                    im2captions[im] = []

                im2captions[im].append([token for token in line[1:-1] if token not in stop_words])  # last token = '.', thus [1:-1]

    assert(len(im2idx) == len(im2captions))
    captions = []
    cap2im = []
    for im, idx in im2idx.items():
        im_captions = im2captions[im]
        captions += im_captions
        cap2im.append(np.ones(len(im_captions), np.int32) * idx)

    cap2im = np.hstack(cap2im)
    return captions, cap2im, im2idx

class im_word2vec_dataset(Dataset):
    def __init__(self, img_nrs, img_feat_file, cache_file_name):
        self.img_nrs = img_nrs
        self.img_feats = torch.Tensor(np.load(img_feat_file))
        self.captions, self.cap2im, self.im2idx = load_flickr_captions(img_nrs)
        vecs = self.load_vocab(cache_file_name)
        self.vecs = torch.from_numpy(vecs)
        
    def load_vocab(self, cache_filename):
        """
        raises an error when no cached vocabulary is available
        """
        vocab_data = pickle.load(open(cache_filename, 'rb'))
        self.max_length = vocab_data['max_length']
        self.tok2idx = vocab_data['tok2idx']
        vecs = vocab_data['vecs']
            
        self.sent_feats = torch.zeros(len(self.captions), self.max_length, dtype = torch.int64)
        for i, caption in enumerate(self.captions):
            tokens = torch.LongTensor([self.tok2idx[token] for token in caption if token in self.tok2idx])
            self.sent_feats[i, :len(tokens)] = tokens

        return vecs
        
    def __len__(self):
        return len(self.captions)
    
    def __getitem__(self, index):
        i = index//CAP_PER_IMAGE
        return (index, self.img_feats[i], self.sent_feats[index])
    
    def get_vecs(self):
        return self.vecs