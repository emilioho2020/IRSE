#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 16:02:53 2020

@author: emile
"""

from keras.models import Sequential, Model
from keras.layers import Dense, Input, Dropout, Activation, concatenate
from keras.layers.normalization import BatchNormalization
from keras import optimizers
from scipy.spatial.distance import cosine
import tensorflow as tf
import keras.backend as K

EMB_DIM = 512
BATCH_SIZE = 500

def create_img_encoder():
    x = Sequential()
    x.add(Dense(1024, input_dim=2048))
    x.add(Activation('relu'))
    x.add(Dropout(0.5))
    x.add(Dense(EMB_DIM, input_dim = 1024))
    x.add(BatchNormalization())
    return x

def create_txt_encoder(input_dimension):
    x = Sequential()
    x.add(Dense(2048, input_dim = input_dimension))
    x.add(Activation('relu'))
    x.add(Dropout(0.5))
    x.add(Dense(EMB_DIM, input_dim = 1024))
    x.add(BatchNormalization())
    return x

def cosine_similarity(a,b):
    return (tf.tensordot(a,b,1))/(tf.norm(a)*tf.norm(b))

"""
s: similarity function
"""

def pdist(x1, x2):
    """
        x1: Tensor of shape (h1, w)
        x2: Tensor of shape (h2, w)
        Return pairwise distance for each row vector in x1, x2 as
        a Tensor of shape (h1, h2)
    """
    x1_square = tf.reshape(tf.reduce_sum(x1*x1, axis=1), [-1, 1])
    x2_square = tf.reshape(tf.reduce_sum(x2*x2, axis=1), [1, -1])
    return tf.sqrt(x1_square - 2 * tf.matmul(x1, tf.transpose(x2)) + x2_square + 1e-4)

def bi_ranking_loss(fake, outputs, margin = 0.12, num_neg_sample=50):
    BATCH_SIZE = K.shape(outputs)[0]
    txt_outputs = outputs[:,:EMB_DIM]
    img_outputs = outputs[:,EMB_DIM:]
    sent_im_dist = pdist(txt_outputs, img_outputs)
        
    # image loss: sentence, positive image, and negative image
    pos_pair_dist = tf.reshape(tf.linalg.diag_part(sent_im_dist), [BATCH_SIZE, 1])
    neg_pair_dist = tf.reshape(tf.boolean_mask(sent_im_dist, ~tf.eye(BATCH_SIZE, dtype=tf.bool)), [BATCH_SIZE, -1])
    im_loss = tf.clip_by_value(margin + pos_pair_dist - neg_pair_dist, 0, 1e6)
    im_loss = tf.reduce_mean(tf.nn.top_k(im_loss, k=num_neg_sample)[0])
    # sentence loss: image, positive sentence, and negative sentence
    neg_pair_dist = tf.reshape(tf.boolean_mask(tf.transpose(sent_im_dist), ~tf.eye(BATCH_SIZE, dtype=tf.bool)), [BATCH_SIZE, -1])
    sent_loss = tf.clip_by_value(margin + pos_pair_dist - neg_pair_dist, 0, 1e6)
    sent_loss = tf.reduce_mean(tf.nn.top_k(sent_loss, k=num_neg_sample)[0])
    loss = im_loss + sent_loss
    return loss
    
def create_model(text_dim):
    txt_input = Input(shape=(text_dim,))
    img_input = Input(shape = (2048,))
    
    txt_enc = create_txt_encoder(text_dim)
    img_enc = create_img_encoder()
    
    encoded_txt = txt_enc(txt_input)
    encoded_img = img_enc(img_input)
    
    combined = concatenate([encoded_txt, encoded_img])
    model = Model([txt_input,img_input], combined)
    return model

def compile_model(model):
    sgd = optimizers.SGD(lr=0.1, decay=1e-5, momentum=0.9, nesterov=True)
    model.compile(optimizer = sgd, loss = bi_ranking_loss)
