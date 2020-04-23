#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 11:58:11 2020

@author: emile
"""
from flickr30k_entities_utils import get_sentence_data
from nltk.corpus import stopwords
STOP_WORDS = stopwords.words("english")
import math

def get_bow(file_nr: str):
    sentences_data = get_sentence_data("Sentences/"+file_nr+".txt")
    num_of_words = dict()
    for sd in sentences_data:
        s = sd["sentence"]
        word_list = s.lower().rstrip(" .").split(" ")
        for word in word_list:
            if word not in STOP_WORDS:
                num_of_words[word] = num_of_words.get(word,0) + 1
    #return num_of_words
    total = sum(num_of_words.values())
    tf = {key:value/total for (key,value) in num_of_words.items()}
    return tf

def computeIDF(documents):
    N = len(documents)
    idfDict = dict()
    
    for document in documents:
        for word, val in get_bow(document).items():
            if val > 0:
                idfDict[word] = idfDict.get(word,0) + 1
        for word, val in idfDict.items():
            idfDict[word] = math.log(N / float(val))
    return idfDict

def computeTFIDF(tfBagOfWords, idfs):
    tfidf = {}
    for word, val in tfBagOfWords.items():
        tfidf[word] = val * idfs[word]
    return tfidf
