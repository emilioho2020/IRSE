#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch

from model_word2vec import MapperI, MapperT
from data_loader_word2vec import im_word2vec_dataset
from txt_preprocessing import get_train, get_test
import config
from train import train


#%%
torch.backends.cudnn.benchmark = True
train_dataset = im_word2vec_dataset(get_test(), config.IMG_TRAIN_PATH, config.VOCAB_PATH)
vecs = train_dataset.get_vecs()
img_network = MapperI(2048, 1024, config.CODE_LENGTH)
txt_network = MapperT(vecs, config.TXT_EMBEDDING_LENGTH, config.TXT_EMBEDDING_LENGTH, config.CODE_LENGTH)

train(train_dataset, img_network, txt_network, 10)