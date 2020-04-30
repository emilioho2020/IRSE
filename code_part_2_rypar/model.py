# -*- coding: utf-8 -*-
## LIBRARIES IMPORT

import torch
print(torch.__version__)
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

from tqdm import tqdm

## SIMPLE FC NN
class MapperI(nn.Module):
    def __init__(self, feat_dim, args):
        super(MapperI, self).__init__()
        self.fc = nn.Sequential(nn.Linear(int(feat_dim), int(args.dim_embed)),
                                nn.BatchNorm1d(args.dim_embed),
                                nn.ReLU(inplace=True),
                                nn.Dropout(p=0.5),
                                nn.Linear(int(args.dim_embed), int(args.bit)) )
        if args.cuda: self.cuda()
        
    def forward(self, images):
        x = self.fc(images)
        x = nn.functional.normalize(x)  # L2 normalize each feature vector
        return x

class MapperT(nn.Module):
    def __init__(self, feat_dim, vecs, args):
        super(MapperT, self).__init__()
        self.fc = nn.Sequential(nn.Linear(int(feat_dim), int(args.dim_embed)),
                                nn.BatchNorm1d(args.dim_embed),
                                nn.ReLU(inplace=True),
                                nn.Dropout(p=0.5),
                                nn.Linear(int(args.dim_embed), int(args.bit)) )
        n_tokens, token_dim = vecs.shape
        self.vecs = vecs
        self.words = nn.Embedding(n_tokens, token_dim)
        self.words.weight = nn.Parameter(vecs)
        if args.cuda:
            self.cuda()
            self.vecs = vecs.cuda()

    def forward(self, tokens):
        words = self.words(tokens)
        n_words = torch.sum(tokens > 0, 1).float() + 1e-10
        sum_words = words.sum(1).squeeze()
        sentences = sum_words / n_words.unsqueeze(1)
        x = self.fc(sentences)
        x = nn.functional.normalize(x)  # L2 normalize each feature vector
        return x