import nltk
nltk.download('stopwords')
import numpy as np
import pickle, os

import torch
print(torch.__version__)
import torch.utils.data

# GET SENTENCE FEATURES -- INDICES OF TOKENS -- BASED ON CONSTRUCTED VOCABULARY
# converts given sentence to word embeddings
def get_sentence_features(sentences, args):
    if os.path.exists(args.feat_path+'vocab.pkl'):
        vocab_data = pickle.load(open(args.feat_path+'vocab.pkl', 'rb'))
        max_length = vocab_data['max_length']
        tok2idx = vocab_data['tok2idx']
        vecs = vocab_data['vecs']
    else:
        print('could find data/vocab.pkl file!')
    
    sent_feats = np.zeros((len(sentences), max_length), np.int64)
    for i, caption in enumerate(sentences):
        tokens = [tok2idx[token.lower().rstrip(',.!?:')] for token in caption.split() if token.lower().rstrip(',.!?:') in tok2idx]
        sent_feats[i, :len(tokens)] = tokens
    return sent_feats