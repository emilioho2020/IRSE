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

TRAIN_FILE = "../flickr30k_entities/train.txt"
TEST_FILE = "../flickr30k_entities/test.txt"
VAL_FILE = "../flickr30k_entities/val.txt"
 
STOP_WORDS = stopwords.words("english")

TXT_DIMENSION = 4092

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

def get_train():
    with open(TRAIN_FILE, "r") as f:
        return f.read().splitlines()
    
def get_val():
    with open(VAL_FILE, "r") as f:
        return f.read().splitlines()
    
def get_test():
    with open(TEST_FILE, "r") as f:
        return f.read().splitlines()

def get_doc_sentences(file_nr):
    sentences_data = get_sentence_data("../flickr30k_entities/Sentences/"+file_nr+".txt")
    string = ""
    for sd in sentences_data:
        s = sd["sentence"].rstrip(".")
        string += s
    return string

def get_sentences(file_numbers):
    for file_nr in file_numbers:
        yield get_doc_sentences(file_nr)
        
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
    
if __name__ == "__main__":
    main()