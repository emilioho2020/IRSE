#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch.nn as nn
import torch
import config

class MapperI(nn.Module):
    def __init__(self, feat_dim, hidden_dim, out_dim):
        super(MapperI, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(feat_dim, hidden_dim),
                        nn.BatchNorm1d(hidden_dim),
                        nn.ReLU(inplace=True),
                        nn.Dropout(p=0.5))
        self.fc2 = nn.Sequential(nn.Linear(hidden_dim, out_dim))
        if config.CUDA:
            self.cuda()
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        # L2 normalize each feature vector
        x = nn.functional.normalize(x)
        return x
    
class MapperT(nn.Module):
    def __init__(self, vecs, feat_dim, hidden_dim, out_dim):
        super(MapperT, self).__init__()
        self.fc = nn.Sequential(nn.Linear(feat_dim, hidden_dim),
                                nn.BatchNorm1d(hidden_dim),
                                nn.ReLU(inplace=True),
                                nn.Dropout(p=0.5),
                                nn.Linear(hidden_dim, out_dim))
        n_tokens, token_dim = vecs.shape
        self.vecs = vecs  # unique word and its embedding vector -- [#words, embed_dim]
        self.words = nn.Embedding(n_tokens, token_dim)
        self.words.weight = nn.Parameter(vecs)
        if config.CUDA:
            self.cuda()
            self.vecs = vecs.cuda()

    def forward(self, tokens):
        words = self.words(tokens)
        n_words = torch.sum(tokens > 0, 1).float() + 1e-10
        sum_words = words.sum(1).squeeze()
        sentences = sum_words / n_words.unsqueeze(1)  # take an average embed vector across all embed vectors of given caption
        x = self.fc(sentences)
        x = nn.functional.normalize(x)  # L2 normalize each feature vector
        return x