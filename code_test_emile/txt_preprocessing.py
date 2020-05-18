#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 14:41:30 2020

@author: emile
"""

from flickr30k_entities_utils import get_sentence_data
from nltk.corpus import stopwords
import numpy as np
from nltk.stem import LancasterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import pickle

import config
 
STOP_WORDS = stopwords.words("english")

TXT_DIMENSION = 2048

"""
USING sklearn
"""

def main():
    vectorizer = TfidfVectorizer(stop_words="english") #optional: stemmervectorizer
    X = vectorizer.fit_transform(get_sentences(get_train()))
    print(X.shape)    
    nb_of_words = X.shape[1]
    print("Size of vocabulary = {}".format(nb_of_words))
    pickle.dump(vectorizer, open("./models/vectorizer.pickle","wb"))
    truncated_representation(X,vectorizer)
    return vectorizer.get_feature_names()

def truncated_representation(X, vectorizer):
    trunc = TruncatedSVD(TXT_DIMENSION)
    txt_trunc = trunc.fit_transform(X)
    pickle.dump(trunc, open("./models/trunc_{}.pickle".format(TXT_DIMENSION),"wb"))
    print(txt_trunc.shape)

    np.save("../Data/txt_train_trunc_{}.npy".format(TXT_DIMENSION), txt_trunc)
    
    val = vectorizer.transform(get_sentences(get_val()))
    val_trunc = trunc.transform(val)
    np.save("../Data/txt_val_trunc_{}.npy".format(TXT_DIMENSION), val_trunc)
    
    test = vectorizer.transform(get_sentences(get_test()))
    test_trunc = trunc.transform(test)
    np.save("../Data/txt_test_trunc_{}.npy".format(TXT_DIMENSION), test_trunc)
    print("truncs saved")

def get_train():
    with open(config.TRAIN_FILE, "r") as f:
        return f.read().splitlines()
    
def get_val():
    with open(config.VAL_FILE, "r") as f:
        return f.read().splitlines()
    
def get_test():
    with open(config.TEST_FILE, "r") as f:
        return f.read().splitlines()

def get_doc_sentences(file_nr):
    sentences_data = get_sentence_data("../flickr30k_entities/Sentences/"+file_nr+".txt")
    strings = []
    for sd in sentences_data:
        strings.append(sd["sentence"].rstrip("."))
    return strings

def get_sentences(file_numbers):
    for file_nr in file_numbers:
        for s in get_doc_sentences(file_nr):
            yield s
        
def get_input(df, input_set):
    imgs = [nr+".jpg" for nr in input_set]
    res = df.loc[imgs]
    return res.to_numpy()

def stemmer_vectorizer():    
    stemmer = LancasterStemmer()
    analyzer = TfidfVectorizer(stop_words = "english").build_analyzer()
    
    def stemmed_words(doc):
        return [stemmer.stem(w) for w in analyzer(doc)]
    
    vectorizer = TfidfVectorizer(stop_words="english", analyzer=stemmed_words)
    return vectorizer