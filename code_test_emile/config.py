#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CONFIGURATION
"""
CUDA = True
"""
HYPERPARAMETERS
"""
TXT_EMBEDDING_LENGTH = 300
CODE_LENGTH = 32

GAMMA = 1
ETA = 1
"""
PATHS
"""
SENT_PATH = '../flickr30k_entities/Sentences/'

IMG_TRAIN_PATH = '../Data/img_train.npy'
IMG_TEST_PATH = '../Data/img_test.npy'
IMG_VAL_PATH = '../Data/img_val.npy'

#if using truncated BOW representation
VECTORIZER_PATH = './models/vectorizer.pickle'
TRUNCATOR_PATH = './models/trunc_2048.pickle'
TXT_TRAIN_PATH = '../Data/txt_train_trunc_2048.npy'
TXT_VAL_PATH = '../Data/txt_val_trunc_2048.npy'
TXT_TEST_PATH = '../Data/txt_test_trunc_2048.npy'

CAPTIONS_PATH = '../Data/captions.txt'
VOCAB_PATH = '../Data/vocab.pkl'
