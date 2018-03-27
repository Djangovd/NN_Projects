# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import tensorflow as tf
import numpy as np
import scipy.io as sio

a = np.array([[1,2,3]])
print(a.reshape((1,1,1,3)))

vgg = sio.loadmat('./imagenet-vgg-verydeep-19.mat')
print (vgg['layers'])