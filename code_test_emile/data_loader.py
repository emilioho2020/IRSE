#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 19:16:58 2020

@author: emile
"""
from torch.utils.data import Dataset
import numpy as np
import sys
#from sklearn.feature_extraction.text import TfidfTransformer
import pickle

from txt_preprocessing import get_doc_sentences
import config

CAP_PER_IMAGE = 5

class im_sent_dataset(Dataset):
    
    def __init__(self, set_numbers, im_feat_file, vectorizer_path = config.VECTORIZER_PATH):
        self.im_feats = np.load(im_feat_file)
        self.set_numbers = set_numbers
        self.vectorizer = pickle.load(open(vectorizer_path, "rb"))

        print("feature files loaded")
        print("size of data: {}".format(sys.getsizeof(self)))
        
    def __len__(self):
        return len(self.set_numbers*CAP_PER_IMAGE)
    
    def __getitem__(self, index): ##TODO start indexing at 0?
        i = index//CAP_PER_IMAGE #otherwise +1
        img_nr = self.set_numbers[i]
        sent_nr = index%CAP_PER_IMAGE
        return (i, index, self.im_feats[i],self.vectorizer.transform([get_doc_sentences(img_nr)[sent_nr]]))
    
    def get_voc_length(self):
        return len(self.vectorizer.get_feature_names())