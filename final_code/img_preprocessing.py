#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 16:05:15 2020

@author: emile
"""
import pandas as pd
from generator import get_train, get_val, get_test
import numpy as np

IMAGES_FILE = "/media/emile/shared_dual/Documents/IRSE/project/image_features.csv"

def generate_img_numpy(IMAGES_FILE, file_numbers):
    df = pd.read_csv(IMAGES_FILE, header = None, sep = " ", index_col = 0, dtype={0:str})
    indices = [nr+".jpg" for nr in file_numbers]
    result_df = df.loc[indices]
    print(df.shape)
    print(result_df.shape)
    img_input = result_df.to_numpy()
    return img_input

a=generate_img_numpy(IMAGES_FILE, get_test())
np.save("img_test.npy", a)
