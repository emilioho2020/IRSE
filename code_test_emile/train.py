#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 10 17:40:02 2020

@author: emile
"""
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm

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
    
def get_S_img(indices, n):
    """
    Parameters
    ----------
    indices : list of indices i of images
    n : total number of samples
    
    Returns
    -------
    S : S matrix for batch! size = (batch_size, n)
    """
    S = torch.zeros(config.BATCH_SIZE, n, dtype = torch.bool)
    for i,j in enumerate(indices):
        S[i,j] = True
    return S.cuda()

def get_S_txt(indices, n):
    S = torch.zeros(n, config.BATCH_SIZE, dtype = torch.bool)
    for i,j in enumerate(indices):
        S[j,i] = True
    return S.cuda()

def get_dJdF(F_b, G, S_b, B_b, F1): #F_b : (c, batch_size), S_b : (batch_size, n)
    theta_b = 1/2*torch.matmul(F_b.transpose(0,1),G) #theta_b : (batch_size, n)
    dJdF = torch.empty(config.BITS,F_b.shape[1],device='cuda')
    for i in range(F_b.shape[1]):
        dJdF[:,i] = get_dJdFi(F_b[:,i], theta_b[i,:], B_b[:,i], G, S_b[i,:],F1)
    return dJdF #size (c)
        
def get_dJdFi(F_i, theta_i, B_i, G, S_i, F1):
    a = ((torch.sigmoid(theta_i)*G)-S_i*G).sum(dim=1)
    b = 2*config.GAMMA*F_i-B_i+2*config.ETA*F1
    return a+b # size (c)

def get_dJdG(G_b, F, S_b, B_b, G1):
    theta_b = 1/2*torch.matmul(F.transpose(0,1),G_b) #theta_b : (n, batch_size)
    dJdG = torch.empty(config.BITS,G_b.shape[1],device='cuda')
    for j in range(G_b.shape[1]):
        dJdG[:,j] = get_dJdGj(G_b[:,j], theta_b[:,j], B_b[:,j], F, S_b[:,j],G1)
    return dJdG #size (c)
        
def get_dJdGj(G_j, theta_j, B_j, F, S_j, G1):
    a = ((torch.sigmoid(theta_j)*F)-S_j*F).sum(dim=1)
    b = 2*config.GAMMA*G_j-B_j+2*config.ETA*G1
    return a+b # size (c)

def get_B(F,G):
    return torch.sign(config.GAMMA*(F+G))

def get_J():
    return 0

def get_loss(data_loader, img_network, txt_network):
    F, G = predict(data_loader.dataset, img_network, txt_network)
    B = get_B(F,G)
    theta = 1.0 / 2 * torch.matmul(F,G.transpose(0,1))
    """
    note: formula = Sum(S_ij*Theta_ij-log(1+exp(Theta_ij))
    Entire theta matrix is too big to compute, however we need only diagonal elements as 
    S is the identity matrix.
    Therefore I compute each Theta_ii individually:
    """
    theta_diag = torch.sum(F*G, dim=0) #size (n)
    loss1 = -torch.sum(theta_diag - torch.log(1.0 + torch.exp(theta_diag)))
    loss2 = config.GAMMA * (torch.sum(torch.pow(B - F, 2)) + torch.sum(torch.pow(B - G, 2))) #Frobenius norm A = sqrt(sum(A))
    F1 = torch.sum(F, dim=1)
    G1 = torch.sum(G, dim=1)
    loss3 = config.ETA*(torch.sum(torch.pow(F1,2)) + torch.sum(torch.pow(G1,2)))
    return loss1.item(), loss2.item(), loss3.item()
    
def train(train_dataset, img_network, txt_network, epochs):
    F,G = predict(train_dataset, img_network, txt_network)
    B = get_B(F,G).cuda()
    train_loader = DataLoader(train_dataset, batch_size = config.BATCH_SIZE, shuffle = True, num_workers = 4, pin_memory = True)
    history = []
    
    for e in tqdm(range(epochs)):
        #Train on images
        img_network.train()
        for (i, (indices, im_feats, txt_vecs)) in enumerate(train_loader):
            im_feats = im_feats.cuda()
            F_b = img_network.forward(im_feats).transpose(0,1)
            F[:,indices] = F_b #F matrix is updated
            F1 = torch.sum(F, dim=1) # F1 : (c) #TODO could increase performance by computing difference instead
            S_b = get_S_img(indices, len(train_dataset))
            dJdF = get_dJdF(F_b, G, S_b, B[:,indices], F1)
            img_network.zero_grad()
            out = img_network(im_feats)
            out.backward(dJdF.transpose(0,1)) # can't aggregate on batch?
            
        #Train on text
        txt_network.train()
        for (i, (indices, im_feats, txt_vecs)) in enumerate(train_loader):
            txt_vecs = txt_vecs.cuda()
            G_b = txt_network.forward(txt_vecs).transpose(0,1)
            G[:,indices] = G_b
            G1 = torch.sum(G, dim=1) # F1 : (c) #TODO could increase performance by computing difference instead
            S_b = get_S_txt(indices, len(train_dataset))
            dJdG = get_dJdG(G_b, F, S_b, B[:,indices], G1)
            txt_network.zero_grad()
            out = txt_network(txt_vecs)
            out.backward(dJdG.transpose(0,1)) # can't aggregate on batch?
    
        #compute B
        B = get_B(F,G)
        
        print("trained model for epoch {}, computing loss...".format(e))
        loss1, loss2, loss3 = get_loss(train_loader, img_network, txt_network)
        total_loss = loss1+loss2+loss3
        evaluation = {'epoch' : e, 'loss1' : loss1, 'loss2' : loss2, 'loss3' : loss3, 'total loss' : total_loss}
        history.append([evaluation])
        print(evaluation)
    
def get_dJdF_efficient(F_b, G, S_b, B_b, F1):
    theta_b = 1/2*torch.matmul(F_b.transpose(0,1),G) #theta_b : (batch_size, n)
    dJdF_b = torch.matmul(G,theta_b.transpose(0,1)) # dJdF_b : (c, batch_size)
    dJdF = torch.sum(dJdF_b, dim=1) # (c)
    dJdF += config.BATCH_SIZE*(2*config.GAMMA*(F_b-B_b)+2*config.ETA*F1)
    
