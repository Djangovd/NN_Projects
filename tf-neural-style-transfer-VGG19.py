#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 22:52:09 2018

Heavily inspired by the excellent guide found here:
    http://www.chioka.in/tensorflow-implementation-neural-algorithm-of-artistic-style

@author: patrickm
"""
import os
import sys
import numpy as np
import scipy
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from PIL import image
#% matplotlib inline

## Image specific settings
# Folder for output images
OUTPUT_DIR = 'output/'
# Image used for style
STYLE_IMG = 'images/'
# Image containing the content
CONTENT_IMG = 'images/'
# Image dimensions
IMG_W = 800
IMG_H = 600
COLOR_CHAN = 3
###
# CNN specific settings
#
# Percentage of weight of noise for intermixing with content img.
NOISE_RATIO = 0.6
# Constant used to emphasise content loss
BETA = 5
# Constant used to emphasise style loss
ALPHA = 100
# The VGG model itself
VGG_MODEL = 'imagenet-vgg-verydeep-19.mat'
# Mean values to subtract from the inputs to the VGG. Used during training
MEAN_VALUES = np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3))

def vgg_model(path_to_imagenet_file):
    vgg = scipy.io.loadmat(path_to_imagenet_file)
    
    vgg_layers = vgg['layers']







































