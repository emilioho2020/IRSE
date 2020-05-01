## LIBRARIES IMPORT
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import os, sys, pickle, shutil#, numbers

import torch
print(torch.__version__)
import torch.optim as optim
import torch.utils.data
import torch.backends.cudnn as cudnn

from training import *

##TODO: Refactor all of this

## ARGUMENTS
class ARGS:
    def __init__(self):
        self.name = 'Two_Branch_Network'
        self.seed = 11
        self.cuda = True
        self.feat_path = ""
        self.save_dir = 'models/'
        self.resume = 'none'#'models/Two_Branch_Network/checkpoint.pth.tar'
        self.test = ''
        self.display_interval=0.25
        self.given_im_features = False  # use given features else use resnet feat
        self.embedding_length=300
        self.dim_embed=2048
        # LEARNING  
        self.lr=1e-4
        self.text_lr_multi=2.0
        self.batch_size=256
        self.sample_size=1
        self.max_num_epoch=1
        self.no_gain_stop=4
        self.minimum_gain=0.1
        self.num_neg_sample=10
        self.margin=0.05
        
        self.bit=32
        self.gamma=1
        self.eta=1

ARGS = ARGS()

if ARGS.cuda:
    torch.cuda.manual_seed(ARGS.seed)

# choose the features -- one given by TA or the ones used in the paper
# as default used the ones from paper -- better performance
if not (os.path.exists(ARGS.feat_path + 'train_im_features.npy') and os.path.exists(ARGS.feat_path + 'test_im_features.npy') and os.path.exists(ARGS.feat_path + 'val_im_features.npy')):
    generate_npy(ARGS.feat_path+'image_features.csv', ARGS.feat_path)
    
train_loader = DatasetLoader(ARGS, 'train')

vocab_filename = os.path.join(ARGS.feat_path, 'vocab.pkl')
word_embeddings_filename = os.path.join(ARGS.feat_path, 'mt_grovle.txt')

print('Loading vocab')
vecs = train_loader.build_vocab(vocab_filename, word_embeddings_filename, ARGS.embedding_length)
print('Loading complete')

kwargs = {'num_workers': 8, 'pin_memory': True} if ARGS.cuda else {}
train_loader = torch.utils.data.DataLoader(train_loader, batch_size=ARGS.batch_size, shuffle=True, **kwargs)
test_loader = DatasetLoader(ARGS, 'test')
val_loader = DatasetLoader(ARGS, 'val')

# Assumes the train_loader has already built the vocab and can be loaded from the cached file.
test_loader.build_vocab(vocab_filename)
val_loader.build_vocab(vocab_filename)

image_feature_dim = train_loader.dataset.im_feats.shape[-1]
n_tokens, token_dim = vecs.shape
vecs = torch.from_numpy(vecs)
if ARGS.cuda: vecs = vecs.cuda()

mapper_t = MapperT(token_dim, vecs, ARGS)
mapper_i = MapperI(image_feature_dim, ARGS)


# optionally resume from a checkpoint
start_epoch, best_acc = 1, 0.0  # load_checkpoint(mappe, ARGS.resume)
cudnn.benchmark = True

parameters_i = [{'params' : mapper_i.fc.parameters()}]

parameters_t = [{'params' : mapper_t.words.parameters(), 'weight_decay' : 0.},
                {'params' : mapper_t.fc.parameters(), 
                'lr' : ARGS.lr*ARGS.text_lr_multi}]

optimizer_i = optim.Adam(parameters_i, lr=ARGS.lr, weight_decay=0.001)
scheduler_i = torch.optim.lr_scheduler.ExponentialLR(optimizer_i, gamma = 0.794)
optimizer_t = optim.Adam(parameters_t, lr=ARGS.lr, weight_decay=0.001)
scheduler_t = torch.optim.lr_scheduler.ExponentialLR(optimizer_t, gamma = 0.794)

n_parameters = sum([p.data.nelement() for model in [mapper_i, mapper_t] for p in model.parameters()])
print('  + Number of params: {}'.format(n_parameters))

save_directory = os.path.join(ARGS.save_dir, ARGS.name)
if not os.path.exists(save_directory):
    os.makedirs(save_directory)