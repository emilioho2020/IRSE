#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 23:12:57 2020

@author: emile
"""
import torch
import numpy as np

def get_size_of_torch(t : torch.tensor):
    return t.nelement()*t.element_size()

def get_size_of_numpy(a : np.ndarray):
    return (a.size * a.itemsize)