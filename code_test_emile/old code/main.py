#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 21:14:41 2020

@author: emile
"""
from torch.utils.data import DataLoader
import torch

from model import feat_nn
from data_loader import im_sent_dataset
from data_loader_word2vec import im_word2vec_dataset
from txt_preprocessing import get_train
import config

loader = im_word2vec_dataset(get_test(), config.IMG_TEST_PATH, config.VOCAB_PATH)
img_network = feat_nn(2048, 1024)
txt_network = feat_nn(train_set.get_voc_length(), 8192)

#%%
"""TESTING"""
for i in range(6):
    print(loader.__getitem__(i))
mapper_i = img_network
mapper_t = txt_network
F = mapper_i.forward(torch.from_numpy(train_set.get_img_feats()).float())
F = mapper_i.forward(torch.Tensor([img_feat for _,_,img_feat,_ in train_set]))

G = mapper_

for i,j, b, c in train_set:
    if i%1000==0 :print(i)