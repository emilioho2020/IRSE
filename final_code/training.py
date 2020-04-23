#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 16:57:26 2020

@author: emile
"""
from models import create_model, compile_model
from keras.models import load_model, save_model
import numpy as np

BATCH_SIZE = 1000
TXT_DIM = 4092
MODEL_PATH = "./models/main_{}.h5".format(TXT_DIM)

"""
change file paths here
"""
txt_train = np.load('../Data/txt_train_trunc_{}.npy'.format(TXT_DIM))
txt_val = np.load('../Data/txt_val_trunc_{}.npy'.format(TXT_DIM))
img_train = np.load('../Data/img_train.npy')
img_val = np.load('../Data/img_val.npy')

model = create_model(TXT_DIM)
print(model.count_params())
compile_model(model)

print("done")

history = model.fit(x=[txt_train,img_train], y=[np.ones((txt_train.shape[0],1024))], \
                    validation_data = ([txt_val, img_val],[np.ones((1000,1024))]), \
                        batch_size = BATCH_SIZE, epochs = 50, verbose = 2, shuffle = True)
    
# Plot history
import matplotlib.pyplot as plt
plt.plot(history.history['loss'], label='loss (testing data)')
plt.plot(history.history['val_loss'], label='loss (validation data)')
plt.title('Training losses')
plt.ylabel('loss value')
plt.xlabel('No. epoch')
plt.legend(loc="upper left")
plt.show()

save_model(model, MODEL_PATH)