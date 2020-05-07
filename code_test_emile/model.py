import torch.nn as nn
import config

class feat_nn(nn.Module):
    def __init__(self, feat_dim, hidden_dim, out_dim):
        super(feat_nn, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(feat_dim, hidden_dim),
                        nn.BatchNorm1d(hidden_dim),
                        nn.ReLU(inplace=True),
                        nn.Dropout(p=0.5))
        self.fc2 = nn.Sequential(nn.Linear(hidden_dim, out_dim))
        if config.CUDA: self.cuda()
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        # L2 normalize each feature vector
        x = nn.functional.normalize(x)
        return x
    