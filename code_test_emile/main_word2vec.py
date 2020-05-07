#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from torch.utils.data import DataLoader
import torch

from model_word2vec import MapperI, MapperT
from data_loader_word2vec import im_word2vec_dataset
from txt_preprocessing import get_train, get_test
import config

def predict(dataset, img_network, txt_network, batch_size = 2048):
    loader = DataLoader(dataset, batch_size = batch_size, shuffle = False, num_workers = 4, pin_memory = True)
    F = torch.empty(config.CODE_LENGTH, len(dataset), device='cuda')
    G = torch.empty(config.CODE_LENGTH, len(dataset), device='cuda')
    for (i,(im_feats, txt_vecs)) in enumerate(loader):
        im_feats, txt_vecs = im_feats.cuda(), txt_vecs.cuda()
        F[:,i*batch_size:(i+1)*batch_size]=img_network.forward(im_feats).transpose(0,1)
        G[:,i*batch_size:(i+1)*batch_size]=txt_network.forward(txt_vecs).transpose(0,1)
    return F,G
    
#cudnn.benchmark = True
test_dataset = im_word2vec_dataset(get_test(), config.IMG_TEST_PATH, config.VOCAB_PATH)
vecs = test_dataset.get_vecs()
img_network = MapperI(2048, 1024, config.CODE_LENGTH)
txt_network = MapperT(vecs, config.TXT_EMBEDDING_LENGTH, config.TXT_EMBEDDING_LENGTH, config.CODE_LENGTH)

F,G = predict(test_dataset, img_network, txt_network)
