#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch

import config

def compute_S(m):
    """
    Parameters
    ----------
    m : int
        number of images.
    Returns
    -------
    torch.sparse.ByteTensor
        sparse S matrix.
    """
    a = torch.LongTensor([[i for i in range(m) for _  in range(5)], range(m*5)])
    b = torch.ByteTensor([1]*m*5)
    return torch.sparse.ByteTensor(a,b)

def compute_dJdF_batch(FT_batch, G, S_batch):
    torch.sigmoid(theta_im_batch(FT_batch,G))*
    
def compute_dJdfi(i,F_batch, theta_im_batch, G, S_batch):
    a = ((torch.sigmoid(theta_im_batch[i])-S_batch[i])*G).sum(dim=1)
    b = 2*config.GAMMA*F_batch[:,i]-B[:,i]+2*config.ETA*F1
    return res

def compute_theta_im_batch(F_batch_T, G):
    return 1/2*torch.mm(F_batch_T,G)
    
def train(train_set, train_loader, mapper_i, mapper_t, optimizer_i=None, optimizer_t=None, nb_epochs=10):
    F = mapper_i.forward(torch.from_numpy(train_set.get_img_feats())).transpose(0,1).cuda()
    G = mapper_t.forward(train_set.get_sent_vecs()).transpose(0,1).cuda()
    #G = torch.empty(config.CODE_LENGTH, len(train_loader), device = "cuda")
    #F = torch.empty_like(G)
    B = config.GAMMA*torch.sign(F+G)
    S = torch.eye(len(train_loader))
    
    for indices, sent_indices, batch_im_feats, batch_sent_feats in train_loader:
        batch_im_feats= batch_im_feats.cuda()
        FT_batch = mapper_i.forward(batch_im_feats)
        S_batch = S[[indices]]

