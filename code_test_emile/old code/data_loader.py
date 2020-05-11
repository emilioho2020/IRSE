#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 19:16:58 2020

@author: emile
"""
from torch.utils.data import Dataset
import torch
import numpy as np
import pickle

from txt_preprocessing import get_doc_sentences
import config
from utils import get_size_of_torch, get_size_of_numpy

CAP_PER_IMAGE = 5

class im_sent_dataset(Dataset):
    """
    This dataset stores the pre-computed image and text features directly = data intensive!
    """
    def __init__(self, set_numbers, img_feat_file, txt_feat_file):
        self.img_feats = torch.from_numpy(np.load(img_feat_file))
        print("size of image data: {}".format(get_size_of_torch(self.img_feats)))
        self.txt_feats = torch.from_numpy(np.load(txt_feat_file))
        print("size of image data: {}".format(get_size_of_torch(self.txt_feats)))
        self.file_numbers = set_numbers
        print("feature files loaded")

    def __len__(self):
        return len(self.txt_feats)
    
    def __getitem__(self, index):
        """
        Returns
        -------
        tuple
            (index of image-caption pair, image features vector, text BOW vector).

        """
        i = index//CAP_PER_IMAGE
        return (index, self.get_img_feats()[i], self.get_sent_vecs()[index])
          
    def get_num_imgs(self):
        return len(self.file_numbers)
    
    def get_voc_length(self):
        return len(self.vectorizer.get_feature_names())
    
    def get_img_feats(self):
        return self.img_feats
    
    def get_sent_vecs(self):
        return self.txt_feats
            
    
    
class im_sent_dataset_vectorizer(Dataset):
    """
    This dataset stores the pre-computed image and computes the txt features individually
    Advantage: less storage needed
    Disadvantage: requires lot of computing
    """
    def __init__(self, set_numbers, img_feat_file, vectorizer = config.VECTORIZER_PATH, \
                 use_trunc = True, truncator = config.TRUNCATOR_PATH):
        self.img_feats = np.load(img_feat_file)
        self.vectorizer = pickle.load(open(vectorizer, "rb"))
        self.use_trunc = use_trunc
        if use_trunc: self.truncator = pickle.load(open(truncator, "rb"))
        self.file_numbers = set_numbers

        print("feature files loaded")
        print("size of image data: {}".format(get_size_of_numpy(self.img_feats)))

    def __len__(self):
        return len(self.file_numbers*CAP_PER_IMAGE)
    
    def __getitem__(self, index):
        """
        Returns
        -------
        tuple
            (index of image, index of caption, image features vector, text BOW vector).

        """
        i = index//CAP_PER_IMAGE
        file_nr = self.file_numbers[i]
        sent_nr = index%CAP_PER_IMAGE
        txt_feat = self.vectorizer.transform([get_doc_sentences(file_nr)[sent_nr]])
        if self.use_trunc: txt_feat = self.truncator.transform(txt_feat)
        return (i, index, self.get_img_feats()[i],txt_feat)
          
    def get_num_imgs(self):
        return len(self.file_numbers)
    
    def get_voc_length(self):
        return len(self.vectorizer.get_feature_names())
    
    def get_img_feats(self):
        return self.img_feats