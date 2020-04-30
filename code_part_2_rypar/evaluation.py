import numpy as np
import torch
import torch.utils.data
from torch.autograd import Variable

from training import generate_image_code, generate_text_code


def test(test_loader, mapper_i, mapper_t, args):
    mapper_i.eval()
    mapper_t.eval()
    sent_feats = torch.from_numpy(test_loader.sent_feats)  # sent_feats are indices
    im_feats = torch.from_numpy(test_loader.im_feats)
    if ARGS.cuda:
        sent_feats, im_feats = sent_feats.cuda(), im_feats.cuda()

    sent_feats, im_feats = Variable(sent_feats), Variable(im_feats)

    i_hash = generate_image_code(mapper_i, im_feats, args)
    t_hash = generate_text_code(mapper_t, sent_feats, args)
    labels = torch.from_numpy(test_loader.labels)
    im2sent = map_k(i_hash, t_hash, labels.t())
    sent2im = map_k(t_hash, i_hash, labels)
    MAP = (im2sent + sent2im)/2
    print('\n{} set im2sent: {:.2f}%, sent2im: {:.2f}%, total: {:.2f}% \n'.format(test_loader.split,
                                                                               im2sent*100, sent2im*100, MAP*100))
    return MAP

def map_k(queries, targets, labels, k=10):
    mAP = 0.
    is_correct_target = torch.repeat_interleave(labels, 5, dim=0)  # [#images*5, #captions] -- img vs. caption
    correct_target_num = (is_correct_target[0].sum()).type(torch.LongTensor).item()
    assert correct_target_num > 0

    hamming_dist = torch.matmul(queries, targets.t())
    _, hd_sorted_idx = hamming_dist.sort(axis=1, descending=True)  # -1*-1=1 and 1*1=1 looking max value = max similarity = min hamm_dist
    total = min(k, correct_target_num)
    count = torch.arange(1, correct_target_num+1)  # calculate the weight for MAP calculation

    for i in range(len(queries)):
        query_result = is_correct_target[i, hd_sorted_idx[i]]    
        tindex = torch.nonzero(query_result)[:total].squeeze() + 1.  # get non zero indices
        mAP += torch.mean(count.type(torch.FloatTensor)/tindex.type(torch.FloatTensor))
    return mAP/len(queries)

def recallAtK(dist_matrix, labels):
    assert len(dist_matrix) == len(labels)
    thresholds = [1, 5, 10]
    successAtK = np.zeros(len(thresholds), np.float32)
    _, indices = dist_matrix.topk(max(thresholds), largest=False)
    for i, k in enumerate(thresholds):
        for sample_indices, sample_labels in zip(indices[:, :k], labels):
            successAtK[i] += sample_labels[sample_indices].max()

    if len(indices) > 0:
        successAtK /= len(indices)

    successAtK = np.round(successAtK*100, 1)
    return successAtK