#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 22:52:09 2018

Heavily inspired by/based on the excellent guide found here:
    http://www.chioka.in/tensorflow-implementation-neural-algorithm-of-artistic-style

Detailed description of the VGG network(s) can be found in:
    https://arxiv.org/pdf/1409.1556.pdf

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
#   Load the VGG neural network in question
    vgg = scipy.io.loadmat(path_to_imagenet_file)
    
    vgg_layers = vgg['layers']


# define wrappers, which makes the code easier to read and work with

    def _weights(layer, expected_layer_name):
        W = vgg_layers[0][layer][0][0][0][0][0]
        b = vgg_layers[0][layer][0][0][0][0][1]
        layer_name = vgg_layers[0][layer][0][0][-2]
        assert layer_name == expected_layer_name
        return W, b

    def _relu(layer):
        return tf.nn.relu(layer)
    
    def _sigmoid(layer):
        return tf.sigmoid(layer)
    
    def _tanh(layer):
        return tf.sigmoid(layer)
    
    def _conv2d(prev_layer, layer, layer_name):
        W, b = _weights(layer, layer_name)
        W = tf.constant(W)
        b = tf.constant(np.reshape(b,(b.size)))
        return tf.nn.conv2d(prev_layer, filter=W, strides=[1,1,1,1], padding='SAME') + b

    def _conv2d_relu(prev_layer, layer, layer_name):
        return _relu(_conv2d(prev_layer, layer, layer_name))
    
    def _conv2d_sigmoid(prev_layer, layer, layer_name):
        return _sigmoid(_conv2d(prev_layer, layer, layer_name))
    
    def _conv2d_tanh(prev_layer, layer, layer_name):
        return _tanh(_conv2d(prev_layer, layer, layer_name))
    
    def _avgpool(prev_layer):
        return tf.nn.avg_pool(prev_layer, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    
    def _maxpool(prev_layer):
        return tf.nn.max_pool(prev_layer, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

# Construct the network in terms of graph (defined as a python dictionary)
# The graph follows the outline in the paper of a 19 layers deep 
# CNN with 16 convolutional layers, (ignoring the three fully connected (FC) layers, that normally follow)
# See graph.readme for an overview
    graph = {}
    graph['input'] = tf.Variable(np.zeros((1, IMG_W, IMG_H, COLOR_CHAN)), dtype='float32')
    graph['conv1_1'] = _conv2d_relu(graph['input'], 0, 'conv1_1')
    graph['conv1_2'] = _conv2d_relu(graph['input'], 2, 'conv1_2')
    graph['avgpool1'] = _avgpool(graph['conv1_2'])
    #graph['maxpool1'] = _maxpool(graph['conv1_2'])
    graph['conv2_1'] = _conv2d_relu(graph['input'], 5, 'conv2_1')
    graph['conv2_2'] = _conv2d_relu(graph['input'], 7, 'conv2_2')
    graph['avgpool2'] = _avgpool(graph['conv2_2'])
    #graph['maxpool2'] = _maxpool(graph['conv1_2'])
    graph['conv3_1'] = _conv2d_relu(graph['input'], 10, 'conv3_1')
    graph['conv3_2'] = _conv2d_relu(graph['input'], 12, 'conv3_2')
    graph['conv3_3'] = _conv2d_relu(graph['input'], 14, 'conv3_3')
    graph['conv3_4'] = _conv2d_relu(graph['input'], 16, 'conv3_4')
    graph['avgpool3'] = _avgpool(graph['conv3_4'])
    #graph['maxpool3'] = _maxpool(graph['conv3_4'])
    graph['conv4_1'] = _conv2d_relu(graph['input'], 19, 'conv4_1')
    graph['conv4_2'] = _conv2d_relu(graph['input'], 21, 'conv4_2')
    graph['conv4_3'] = _conv2d_relu(graph['input'], 23, 'conv4_3')
    graph['conv4_4'] = _conv2d_relu(graph['input'], 25, 'conv4_4')
    graph['avgpool4'] = _avgpool(graph['conv4_4'])
    #graph['maxpool4'] = _maxpool(graph['conv4_4'])
    graph['conv5_1'] = _conv2d_relu(graph['input'], 28, 'conv5_1')
    graph['conv5_2'] = _conv2d_relu(graph['input'], 30, 'conv5_2')
    graph['conv5_3'] = _conv2d_relu(graph['input'], 32, 'conv5_3')
    graph['conv5_4'] = _conv2d_relu(graph['input'], 34, 'conv5_4')
    graph['avgpool5'] = _avgpool(graph['conv5_4'])
    #graph['maxpool5'] = _maxpool(graph['conv5_4'])
    
