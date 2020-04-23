#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 18:40:26 2020

@author: emile
"""

from models import bi_ranking_loss
import keras.losses
keras.losses.bi_ranking_loss = bi_ranking_loss #ugly fix: unable to save entire model otherwise
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from flickr30k_entities_utils import get_sentence_data
import numpy as np
from keras.models import load_model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import pickle

EMB_DIM = 512
TXT_DIM = 4092

IMG_DIR = '/media/emile/shared_dual/Documents/IRSE/project/Data/flickr30k-images/'
TEST_FILE = '../flickr30k_entities/test.txt'
TXT_TEST = '../Data/txt_test_trunc_{}.npy'.format(TXT_DIM) #location of pre-processed txt test file
IMG_TEST = '../Data/img_test.npy'.format(TXT_DIM) #location of pre-processed txt file
MODEL_FILE = "./models/main_{}.h5".format(TXT_DIM)
VECTORIZER_PATH = "./models/vectorizer.pickle".format(TXT_DIM)
SVD_PATH = "./models/trunc_{}.pickle".format(TXT_DIM)


def get_test():
    with open(TEST_FILE, "r") as f:
        return f.read().splitlines()

def get_n_closest_images(vect, img_emb, n = 10):
  distances = cdist([vect], img_emb, 'cosine')[0]
  sorted_indices = np.argsort(distances)
  #print(sorted_indices[:n])
  test_imgs = get_test()
  img_nrs = [test_imgs[i] for i in sorted_indices[:n]]
  return img_nrs

def show_img(img_nr):
    img=mpimg.imread(IMG_DIR+str(img_nr)+'.jpg')
    plt.imshow(img)
    plt.show()
    
def eval_img(index, txt_emb):
    nr = get_test()[index]
    sentences = get_sentence_data("Sentences/"+nr+".txt")
    print(sentences)
    vect = txt_emb[index]
    img_nrs = get_n_closest_images(vect)
    for img in img_nrs:
        show_img(img)


def main():
    model = load_model(MODEL_FILE)
    print("model loaded")
    txt_test = np.load(TXT_TEST)
    img_test = np.load(IMG_TEST)
    out = model.predict([txt_test, img_test])
    img_emb = out[:,EMB_DIM:]
    print("embeddings calculated")
    vectorizer = pickle.load(open(VECTORIZER_PATH, "rb"))
    svd = pickle.load(open(SVD_PATH, "rb"))
    
    caption = input("Please write a caption or sentence: ")
    a = vectorizer.transform([caption])
    a = svd.transform(a)
    prediction = model.predict([a,np.ones((1,2048))])
    txt_vector = prediction[0,:EMB_DIM]    
    img_nrs = get_n_closest_images(txt_vector, img_emb, 10)
    for img in img_nrs:
        show_img(img)
    
if __name__ == "__main__":
    main()