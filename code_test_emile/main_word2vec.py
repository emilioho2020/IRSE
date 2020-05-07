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
    for (i,(_, im_feats, txt_vecs)) in enumerate(loader):
        im_feats, txt_vecs = im_feats.cuda(), txt_vecs.cuda()
        F[:,i*batch_size:(i+1)*batch_size]=img_network.forward(im_feats).transpose(0,1)
        G[:,i*batch_size:(i+1)*batch_size]=txt_network.forward(txt_vecs).transpose(0,1)
    return F,G
    
def compute_S(indices, set_length):
    S = torch.zeros(config.BATCH_SIZE, set_length, dtype = torch.bool)
    for i,j in enumerate(indices):
        S[i,j] = True
    return S

def compute_dJdFi_batch(F_batch, G, S_batch, B):
    return 0

def compute_dJdfi(i,F_batch, theta_im_batch, G, S_batch, F):
    a = ((torch.sigmoid(theta_im_batch[i])-S_batch[i])*G).sum(dim=1)
    F1 = torch.sign(F)
    b = 2*config.GAMMA*F_batch[:,i]-B[:,i]+2*config.ETA*F1
    return a+b

def compute_B(F,G):
    return torch.sign(config.GAMMA*(F+G))

#cudnn.benchmark = True
train_dataset = im_word2vec_dataset(get_test(), config.IMG_TEST_PATH, config.VOCAB_PATH)
vecs = train_dataset.get_vecs()
img_network = MapperI(2048, 1024, config.CODE_LENGTH)
txt_network = MapperT(vecs, config.TXT_EMBEDDING_LENGTH, config.TXT_EMBEDDING_LENGTH, config.CODE_LENGTH)

F,G = predict(train_dataset, img_network, txt_network)
B = compute_B(F,G).cuda()
#%%
train_loader = DataLoader(train_dataset, batch_size = config.BATCH_SIZE, shuffle = True, num_workers = 4, pin_memory = True)
for (i, (indices, im_feats, txt_vecs)) in enumerate(train_loader):
    im_feats = im_feats.cuda()
    F_batch = img_network.forward(im_feats).transpose(0,1)
    S_batch = compute_S(indices, len(train_dataset))