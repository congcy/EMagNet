'''
Train a fully convolutional network for earthquake magnitude.
'''

from __future__ import print_function
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5,6"
import keras
from keras.datasets import mnist
from keras import losses
from keras import optimizers
from keras.models import Sequential
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten,Activation,Reshape
from keras.layers import Conv1D,Conv2D,MaxPooling1D,MaxPooling2D,BatchNormalization
from keras.layers import UpSampling1D,UpSampling2D,AveragePooling1D,AveragePooling2D 
from keras.layers import ZeroPadding1D,ZeroPadding2D 
from keras.layers.advanced_activations import LeakyReLU
from keras import backend as K
from keras.utils import plot_model
import scipy.stats as stats
import read2
import numpy as np
import time
import matplotlib.pyplot as plt
import sys
import math
np.random.seed(7)

num=1000000 # num of training samples 
num2=10000  # num of test samples
sm,sn,x_train,y_train=read2.load_data(sgynam='TRAIN DATA PATH',sgyf1=1,sgyt1=num,step1=1,sgyf2=1,sgyt2=1,step2=1,shuffle='true')
sm,sn,x_test,y_test=read2.load_data(sgynam='TEST DATA PATH',sgyf1=1,sgyt1=num2,step1=1,sgyf2=1,sgyt2=1,step2=1,shuffle='true')

batch_size = 4
epochs = 100

# input image dimensions
img_rows, img_cols = sm, sn

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test  = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
y_train = y_train.astype('float32')
y_test = y_test.astype('float32')

main_input = Input(shape=input_shape,name='main_input')
x=Conv2D(8, kernel_size=(3,3),padding='valid')(main_input)
x=MaxPooling2D(pool_size=(1,4),padding='valid')(x)
x=LeakyReLU(alpha=0.2)(x)
x=Conv2D(16, kernel_size=(3,3),padding='valid')(x)
x=MaxPooling2D(pool_size=(1,4),padding='valid')(x)
x=LeakyReLU(alpha=0.2)(x)
x=Conv2D(32, kernel_size=(3,3),padding='valid')(x)
x=MaxPooling2D(pool_size=(1,4),padding='valid')(x)
x=LeakyReLU(alpha=0.2)(x)
x=Conv2D(64, kernel_size=(3,3),padding='valid')(x)
x=MaxPooling2D(pool_size=(1,2),padding='valid')(x)
x=LeakyReLU(alpha=0.2)(x)
x=Conv2D(128, kernel_size=(3,3),padding='valid')(x)
x=MaxPooling2D(pool_size=(2,2),padding='valid')(x)
x=LeakyReLU(alpha=0.2)(x)
x=Conv2D(256, kernel_size=(3,3),padding='valid')(x)
x=MaxPooling2D(pool_size=(2,2),padding='valid')(x)
x=LeakyReLU(alpha=0.2)(x)
x=Conv2D(512, kernel_size=(1,3),padding='valid')(x)
x=MaxPooling2D(pool_size=(1,2),padding='valid')(x)
x=LeakyReLU(alpha=0.2)(x)
x=Conv2D(1024, kernel_size=(1,3),padding='valid')(x)
x=MaxPooling2D(pool_size=(1,2),padding='valid')(x)
x=LeakyReLU(alpha=0.2)(x)
x=Conv2D(2048, kernel_size=(1,1),padding='valid')(x)
x=UpSampling2D(size=(1,2))(x) #1
x=LeakyReLU(alpha=0.2)(x)
x=Conv2D(1024, kernel_size=(1,3),padding='same')(x)
x=UpSampling2D(size=(1,2))(x) #2
x=LeakyReLU(alpha=0.2)(x)
x=Conv2D(512, kernel_size=(1,3),padding='same')(x)
x=UpSampling2D(size=(1,2))(x) #3
x=LeakyReLU(alpha=0.2)(x)
x=Conv2D(256, kernel_size=(1,3),padding='same')(x)
x=UpSampling2D(size=(1,2))(x) #4
x=LeakyReLU(alpha=0.2)(x)
x=Conv2D(128, kernel_size=(1,3),padding='same')(x)
x=UpSampling2D(size=(1,2))(x) #5
x=LeakyReLU(alpha=0.2)(x)
x=Conv2D(64, kernel_size=(1,3),padding='same')(x)
x=UpSampling2D(size=(1,2))(x) #6
x=LeakyReLU(alpha=0.2)(x)
x=Conv2D(32, kernel_size=(1,3),padding='same')(x)
x=UpSampling2D(size=(1,2))(x) #7
x=LeakyReLU(alpha=0.2)(x)
x=Conv2D(16, kernel_size=(1,3),padding='same')(x)
x=UpSampling2D(size=(1,2))(x) #8
x=LeakyReLU(alpha=0.2)(x)
x=Conv2D(8, kernel_size=(1,3),padding='same')(x)
x=UpSampling2D(size=(1,2))(x) #9
x=LeakyReLU(alpha=0.2)(x)
x=Conv2D(4, kernel_size=(1,3),padding='same')(x)
x=UpSampling2D(size=(1,2))(x) #9
x=LeakyReLU(alpha=0.2)(x)
main_output=Conv2D(1, kernel_size=(1,3),padding='same')(x)
main_output=Reshape((1024,))(main_output)

model = Model(inputs=[main_input],outputs=[main_output])
optimizer = keras.optimizers.Adadelta(lr=0.2,rho=0.95,epsilon=1e-06)

model.compile(loss='mse',
              optimizer=optimizer,
              metrics=['accuracy'])
history_callback=model.fit([x_train], 
                           [y_train],
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=([x_test], [y_test]))

model.save('test.cnn')
