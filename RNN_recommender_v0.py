#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 31 14:59:18 2018

@author: patrickm
"""
import sys
import numpy as np
import pandas as pd
import tensorflow as tf

### Arrange data
# Idea: to create "images" for every user. The images are first flattened
# and then used as input a la the MNIST set.




### Hyperparameters
n_neurons = 128
learning_rate = 0.001
batch_size = 10
n_epochs = 10

### Parameters
n_steps = 10
n_inputs = 1000
n_outputs = 1000

### RNN model itself
X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_outputs])

cell = tf.nn.rnn_cell.BasicRNNCell(num_units=n_neurons)
output, state = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

logits = tf.layers.dense(state, n_outputs)
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)

loss =tf.reduce_mean(cross_entropy)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

prediction = tf.nn.in_top_k(logits, y, 1)
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))

####

### Init variables
init = tf.global_variables_initializer()

### Train model
with tf.Session() as sess:
    sess.run(init)
    n_batches = 9
    for epoch in range(n_epochs):
        for batch in range(n_batches):
            