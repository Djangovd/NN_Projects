#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 14:04:53 2018

@author: Patrick Motylinski
"""

from keras.models import Sequential, Model
from keras.layers import Activation, Conv2D, Dense
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

input_image = Input(shape=(S, S, InDepth))

## 1st Conv. Layer
X = Conv2D(InitConvChannels1, ConvLayerFilter, strides=StridesConv, padding='valid')(input_image)
X = Activation('relu')(X)
X = Conv2D(InitConvChannels1, ConvLayerFilter, strides=StridesConv, padding='valid')(X)
X = Activation('relu')(X)
# 1st MaxPool
X = MaxPooling2D(pool_size=PoolingSize, strides=StridesPool)(X)

## 2nd Conv. Layer
X = Conv2D(InitConvChannels2, ConvLayerFilter, strides=StridesConv, padding='valid')(X)
X = Activation('relu')(X)
X = Conv2D(InitConvChannels2, ConvLayerFilter, strides=StridesConv, padding='valid')(X)
X = Activation('relu')(X)
# 1st MaxPool
X = MaxPooling2D(pool_size=PoolingSize, strides=StridesPool)(X)

## 3rd Conv. Layer
X = Conv2D(InitConvChannels3, ConvLayerFilter, strides=StridesConv, padding='valid')(X)
X = Activation('relu')(X)
X = Conv2D(InitConvChannels3, ConvLayerFilter, strides=StridesConv, padding='valid')(X)
X = Activation('relu')(X)
# 1st MaxPool
X = MaxPooling2D(pool_size=PoolingSize, strides=StridesPool)(X)

## 4th Conv. Layer
X = Conv2D(InitConvChannels45, ConvLayerFilter, strides=StridesConv, padding='valid')(X)
X = Activation('relu')(X)
X = Conv2D(InitConvChannels45, ConvLayerFilter, strides=StridesConv, padding='valid')(X)
X = Activation('relu')(X)
# 1st MaxPool
X = MaxPooling2D(pool_size=PoolingSize, strides=StridesPool)(X)

## 5th Conv. Layer
X = Conv2D(InitConvChannels45, ConvLayerFilter, strides=StridesConv, padding='valid')(X)
X = Activation('relu')(X)
X = Conv2D(InitConvChannels45, ConvLayerFilter, strides=StridesConv, padding='valid')(X)
X = Activation('relu')(X)
# 1st MaxPool
X = MaxPooling2D(pool_size=PoolingSize, strides=StridesPool)(X)
X = Flatten()(X)

# 1st fully conected (FC) layer
X = Dense(4096)(X)
X = Activation('relu')(X)

# 2nd fully conected (FC) layer
X = Dense(4096)(X)
X = Activation('relu')(X)

# 3rd and final: fully conected (FC) layer
X = Dense(1000)(X)
X = Activation('relu')(X)

# Create model
model = Model(inputs, X,)