#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 14:04:53 2018

@author: Patrick Motylinski
"""

from keras.models import Sequential, Model
from keras.layers import Activation, Conv2D, Dense, Flatten
from keras.activations import relu, softmax
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam, RMSprop, SGD
from keras.regularizers import l2
from keras import backend as K
import numpy as np

S = 224     # no. of pixels in each dimension
InDepth = 3 # no. of channels in image, 3 since it is RGB

InitConvChannels1  = 64
InitConvChannels2  = 128
InitConvChannels3  = 256
InitConvChannels45 = 512
ConvLayerFilter = (3, 3)
ConvLayerFilterSize = 3
StridesConv = (1, 1)
StridesPool = (2, 2)
PoolingSize = (2, 2)

#input_image = Input(shape=(S, S, InDepth))
input_shape = (S, S, InDepth)

## Initiate model
model = Sequential()
## 1st Conv. Layer
model.add(Conv2D(InitConvChannels1, ConvLayerFilter, strides=StridesConv, padding='same', input_shape=input_shape))
model.add(Activation('relu'))
model.add(Conv2D(InitConvChannels1, ConvLayerFilter, strides=StridesConv, padding='same'))
model.add(Activation('relu'))
# 1st MaxPool
model.add(MaxPooling2D(pool_size=PoolingSize, strides=StridesPool))

## 2nd Conv. Layer
model.add(Conv2D(InitConvChannels2, ConvLayerFilter, strides=StridesConv, padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(InitConvChannels2, ConvLayerFilter, strides=StridesConv, padding='same'))
model.add(Activation('relu'))
# 1st MaxPool
model.add(MaxPooling2D(pool_size=PoolingSize, strides=StridesPool))

## 3rd Conv. Layer
model.add(Conv2D(InitConvChannels3, ConvLayerFilter, strides=StridesConv, padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(InitConvChannels3, ConvLayerFilter, strides=StridesConv, padding='same'))
model.add(Activation('relu'))
# 1st MaxPool
model.add(MaxPooling2D(pool_size=PoolingSize, strides=StridesPool))

## 4th Conv. Layer
model.add(Conv2D(InitConvChannels45, ConvLayerFilter, strides=StridesConv, padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(InitConvChannels45, ConvLayerFilter, strides=StridesConv, padding='same'))
model.add(Activation('relu'))
# 1st MaxPool
model.add(MaxPooling2D(pool_size=PoolingSize, strides=StridesPool))

## 5th Conv. Layer
model.add(Conv2D(InitConvChannels45, ConvLayerFilter, strides=StridesConv, padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(InitConvChannels45, ConvLayerFilter, strides=StridesConv, padding='same'))
model.add(Activation('relu'))
# 1st MaxPool
model.add(MaxPooling2D(pool_size=PoolingSize, strides=StridesPool))
model.add(Flatten())

# 1st fully conected (FC) layer
model.add(Dense(4096))
model.add(Activation('relu'))

# 2nd fully conected (FC) layer
model.add(Dense(4096))
model.add(Activation('relu'))

# 3rd and final: fully conected (FC) layer
model.add(Dense(1000))
model.add(Activation('relu'))

# Create model
#model = Model(inputs, X)

model.summary()