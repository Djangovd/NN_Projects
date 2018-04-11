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
import scipy.io
import scipy.misc
import imageio
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow, matshow
from PIL import Image
#% matplotlib inline

## Image specific settings
# Folder for output images
OUTPUT_DIR = 'output/'
# Image used for style
STYLE_IMG = 'images/Dali_melting_clocks.jpg'
#STYLE_IMG = 'images/guernica.jpg'
# Image containing the content
CONTENT_IMG = 'images/Mirror_Queen.jpg'
#CONTENT_IMG = 'images/hongkong.jpg'
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
        wb = vgg_layers[0][layer][0][0][2]
        W = wb[0][0]
        b = wb[0][1]
        layer_name = vgg_layers[0][layer][0][0][0][0]
        assert layer_name == expected_layer_name
        return W, b
## Below was found in the tutorial but doesn't seem to correspond to the format of the imagenet .mat file
#        W = vgg_layers[0][layer][0][0][0][0][0]
#        print("weights : ", W)
#        b = vgg_layers[0][layer][0][0][0][0][1]
#        print("biases : ", b)
#        #layer_name = vgg_layers[0][layer][0][0][-2] # Doesn't appear to be correct
#        layer_name = vgg_layers[0][layer][0][0][0][0]
#        print("layer_name: ", layer_name)
#        print("expected_layer_name: ", expected_layer_name)
#        assert layer_name == expected_layer_name
#        return W, b

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
    graph['input'] = tf.Variable(np.zeros((1, IMG_H, IMG_W, COLOR_CHAN)), dtype='float32')
    graph['conv1_1'] = _conv2d_relu(graph['input'], 0, 'conv1_1')
    graph['conv1_2'] = _conv2d_relu(graph['conv1_1'], 2, 'conv1_2')
    graph['avgpool1'] = _avgpool(graph['conv1_2'])
    #graph['maxpool1'] = _maxpool(graph['conv1_2'])
    graph['conv2_1'] = _conv2d_relu(graph['avgpool1'], 5, 'conv2_1')
    graph['conv2_2'] = _conv2d_relu(graph['conv2_1'], 7, 'conv2_2')
    graph['avgpool2'] = _avgpool(graph['conv2_2'])
    #graph['maxpool2'] = _maxpool(graph['conv1_2'])
    graph['conv3_1'] = _conv2d_relu(graph['avgpool2'], 10, 'conv3_1')
    graph['conv3_2'] = _conv2d_relu(graph['conv3_1'], 12, 'conv3_2')
    graph['conv3_3'] = _conv2d_relu(graph['conv3_2'], 14, 'conv3_3')
    graph['conv3_4'] = _conv2d_relu(graph['conv3_3'], 16, 'conv3_4')
    graph['avgpool3'] = _avgpool(graph['conv3_4'])
    #graph['maxpool3'] = _maxpool(graph['conv3_4'])
    graph['conv4_1'] = _conv2d_relu(graph['avgpool3'], 19, 'conv4_1')
    graph['conv4_2'] = _conv2d_relu(graph['conv4_1'], 21, 'conv4_2')
    graph['conv4_3'] = _conv2d_relu(graph['conv4_2'], 23, 'conv4_3')
    graph['conv4_4'] = _conv2d_relu(graph['conv4_3'], 25, 'conv4_4')
    graph['avgpool4'] = _avgpool(graph['conv4_4'])
    #graph['maxpool4'] = _maxpool(graph['conv4_4'])
    graph['conv5_1'] = _conv2d_relu(graph['avgpool4'], 28, 'conv5_1')
    graph['conv5_2'] = _conv2d_relu(graph['conv5_1'], 30, 'conv5_2')
    graph['conv5_3'] = _conv2d_relu(graph['conv5_2'], 32, 'conv5_3')
    graph['conv5_4'] = _conv2d_relu(graph['conv5_3'], 34, 'conv5_4')
    graph['avgpool5'] = _avgpool(graph['conv5_4'])
    #graph['maxpool5'] = _maxpool(graph['conv5_4'])
    
    return graph

### Content loss, as described by EQN.1 
def loss_func_content(sess, model):
#   a wrapper of sorts
    def content_loss(p, x):
        # N = number of filters
        N = p.shape[3]
        # M = H x W of feature map
        M = p.shape[1] * p.shape[2]
        #return 1./2. * tf.reduce_sum(tf.pow(p - x, 2))
        return (1./(4. * N * M)) * tf.reduce_sum(tf.pow(p - x, 2))
    
    return content_loss(sess.run(model['conv4_2']), model['conv4_2'])

### Style loss, EQN.5. In contrast to content loss, which solely focused on layer
#   conv4_2, style loss is taken over a much larger range from conv1_1 to conv5_1. 
#   The idea is that the style loss captures input across the various layers, both 
#   the more hard, basic features and the soft, refined ones.
#   We begin by using the layers as described in the paper, and attach weights 
#   corresponding to how much we want the features to count in each layer.

STYLE_LAYERS = [('conv1_1', 0.5),('conv2_1', 1.),('conv3_1', 1.5),('conv4_1', 3.),('conv5_1', 4.)]

def loss_func_style(sess, model):
    # We need a Gram matrix
    def gram_matrix(F, N, M):
        Ft = tf.reshape(F, (M, N))
        return tf.matmul(tf.transpose(Ft),Ft)
    
    def style_loss(a, x):
        # N = number of filters in layer l
        N = a.shape[3]
        # M = H x W of feature map (layer l)
        M = a.shape[1] * a.shape[2]
        # Style representation of original image
        A = gram_matrix(a, N, M)
        # Style representation of generated image
        G = gram_matrix(x, N, M)
        return  (1./(4. * N**2 * M**2)) * tf.reduce_sum(tf.pow(A - G, 2))
    
    E = [style_loss(sess.run(model[layer_name]), model[layer_name]) for layer_name, _ in STYLE_LAYERS]
    W = [w for _, w in STYLE_LAYERS]
    loss = sum([W[l]*E[l] for l in range(len(STYLE_LAYERS))])
    return loss

def noise_img_gen(content_img, noise_ratio = NOISE_RATIO):
    # Noise image, resulting from intermixing content and white noise images at certain proportions
    noise_img = np.random.uniform(-20, 20, (1, IMG_H, IMG_W, COLOR_CHAN)).astype('float32')
    # white noise from content representation: weighted average
    input_img = noise_img * noise_ratio + content_img * (1. - noise_ratio)
    return input_img.astype('uint8')

def load_img(path):
    img = scipy.misc.imread(path)
#    img = imageio.imread(path)
    img = np.reshape(img, ((1,)+img.shape))
#    print("1) img = ", img)
    img = img - MEAN_VALUES # Def. in the beginning
#    print("2) img - MEAN = ", img)
#    print ("3) img.shape = ", img.shape)
#    print ("4) mean.shape = ", MEAN_VALUES.shape)
#    print ("5) mean = ", MEAN_VALUES)
    return img.astype('uint8')

def save_img(path, img):
    img = img + MEAN_VALUES
    img = img[0] # remove 1st (useless) dimension
    img = np.clip(img,0,255).astype('uint8') 
    scipy.misc.imsave(path, img)
    
####
    
sess = tf.InteractiveSession()

# Set content image
content_img = load_img(CONTENT_IMG)
print("Content", content_img.shape)
imshow(content_img[0])

# Style image
style_img = load_img(STYLE_IMG)
print("Style", style_img.shape)
imshow(style_img[0])

### Build the model
print("Build model")
model = vgg_model(VGG_MODEL)
print(model)

# input (noise) image
input_img = noise_img_gen(content_img)
print("Input", input_img.shape)
imshow(input_img[0])

## Init. of the model
sess.run(tf.global_variables_initializer())

## Construct content loss
sess.run(model['input'].assign(content_img))
content_loss = loss_func_content(sess, model)

## Construct style loss
sess.run(model['input'].assign(style_img))
style_loss = loss_func_style(sess, model)

## Combine into total loss (Eqn.7 in paper)
total_loss = BETA * content_loss + ALPHA * style_loss

#
optimizer = tf.train.AdamOptimizer(2.0)
train_step = optimizer.minimize(total_loss)

sess.run(tf.global_variables_initializer())
sess.run(model['input'].assign(input_img))

## Time to train
ITERS = 1000 #* 5

sess.run(tf.global_variables_initializer())
sess.run(model['input'].assign(input_img))
for it in range(ITERS):
    sess.run(train_step)
    if it%100 == 0:
        # Print every 100 iteration.
        mixed_img = sess.run(model['input'])
        print('Iteration %d' % (it))
        print('sum : ', sess.run(tf.reduce_sum(mixed_img)))
        print('cost: ', sess.run(total_loss))

        if not os.path.exists(OUTPUT_DIR):
            os.mkdir(OUTPUT_DIR)

        filename = 'output/%d.png' % (it)
        save_img(filename, mixed_img)

save_img('output/art.jpg', mixed_img)

