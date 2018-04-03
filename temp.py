# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import tensorflow as tf
import numpy as np
import scipy.io as sio

#a = np.array([[1,2,3]])
#print(a.reshape((1,1,1,3)))
fl = open("el_crapo.txt", 'w')
vgg = sio.loadmat('./imagenet-vgg-verydeep-19.mat')
lay = vgg['layers']
print (lay[0])
fl.write(str(lay))
fl.close()