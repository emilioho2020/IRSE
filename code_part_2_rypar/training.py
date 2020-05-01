import torch
print(torch.__version__)
import torch.utils.data
from torch.autograd import Variable

import numpy as np

def calculate_S(label1, label2, args):
    # calculate the similar matrix (S) -- neighbour matrix
    S = (label1.matmul(label2.transpose(0, 1)) > 0).type(torch.FloatTensor)
    return S

def derivative(F, G, B, S, args):
    sigma = torch.sigmoid(torch.matmul(F, G.transpose(0, 1)) / 2.)
    term1 = torch.sum( (torch.matmul(sigma, F) - torch.matmul(S, F)) / 2.)  # similarity
    term2 = 2*args.gamma * torch.sum(F - B)  # preserve similarity
    term3 = 2*args.eta * torch.sum(F)  # preserve balance
    loss = term1 + term2 + term3
    return loss

def compound_loss(F, G, B, S, args):
    theta = torch.matmul(F, G.transpose(0, 1)) / 2.  # D*D resp T*T (D = unique image-caption pairs, T = image-caption pairs in batch)
    term1 = torch.sum(S*theta - torch.log(1. + torch.exp(theta)))  # similarity
    term2 = torch.sum(torch.pow(B - F, 2) + torch.pow(B - G, 2))  # preserve similarity
    term3 = torch.sum(torch.pow(F.sum(dim=0), 2) + torch.pow(G.sum(dim=0), 2))  # preserve balance
    loss = term1 + args.gamma * term2 + args.eta * term3
    return loss

def generate_image_code(mapper_i, x, args):
    img = mapper_i(x)
    return torch.sign(img)

def generate_text_code(mapper_t, y, args):
    text = mapper_t(y)
    return torch.sign(text)

def train(train_loader, mapper_i, mapper_t, optimizer_i, optimizer_t, epoch, args):
    steps_per_epoch = len(train_loader.dataset) // args.batch_size
    display_interval = int(np.floor(steps_per_epoch * args.display_interval))
    D = len(train_loader.dataset)   

    # Initialize
    # F_tmp = torch.rand((args.batch_size, args.bit))
    # G_tmp = torch.rand((args.batch_size, args.bit))
    # B = torch.sign(F_tmp + G_tmp)
    # if args.cuda:
    #     F_tmp, G_tmp, B = F_tmp.cuda(), G_tmp.cuda(), B.cuda()

    mapper_i.train()
    for batch_idx, (im_feats, sent_feats) in enumerate(train_loader):
        labels = torch.from_numpy(np.eye(im_feats.size(0), dtype=np.float32))
        if args.cuda:
            im_feats, sent_feats, labels = im_feats.cuda(), sent_feats.cuda(), labels.cuda()

        im_feats, labels = Variable(im_feats), Variable(labels)
        sent_feats = sent_feats.view(labels.size(0), -1)

        S = calculate_S(labels, labels, args)
        F = mapper_i(im_feats)
        G = mapper_t(sent_feats)
        B = torch.sign(args.gamma * (F+G))
        if args.cuda:
            F, G, S = F.cuda(), G.cuda(), S.cuda()

        F, G = Variable(F), Variable(G)  #  requires_grad=True
        derivative_F = derivative(F, G, B, S, args)
        # derivative_F /= (args.batch_size * D)

        # compute gradient and do optimizer step
        optimizer_i.zero_grad()
        derivative_F.backward()  #retain_graph=True
        optimizer_i.step()

    mapper_t.train()
    for batch_idx, (im_feats, sent_feats) in enumerate(train_loader):
        labels = torch.from_numpy(np.eye(im_feats.size(0), dtype=np.float32))
        if args.cuda:
            im_feats, sent_feats, labels = im_feats.cuda(), sent_feats.cuda(), labels.cuda()

        sent_feats, labels =  Variable(sent_feats), Variable(labels)
        sent_feats = sent_feats.view(labels.size(0), -1)

        S = calculate_S(labels, labels, args)
        F = mapper_i(im_feats)
        G = mapper_t(sent_feats)
        B = torch.sign(args.gamma * (F+G))
        if args.cuda:
            F, G, S = F.cuda(), G.cuda(), S.cuda()

        F, G = Variable(F), Variable(G)  # requires_grad=True
        derivative_G = derivative(G, F, B, S, args)
        # derivative_G /= (args.batch_size * D)

        # compute gradient and do optimizer step
        optimizer_t.zero_grad()
        derivative_G.backward()  #retain_graph=True
        optimizer_t.step()


    B = torch.sign(args.gamma * (F + G))
    loss = compound_loss(F, G, B, Variable(S), args)
    print('Epoch: {:d} Loss: {:f}, Image Loss: {:f}, Text Loss: {:f} '.format(epoch, loss, 
                                                                              derivative_F,
                                                                              derivative_G))  #i_average_loss.avg()
