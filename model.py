#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import matplotlib as plt
import config
from time import time

def cnn(image):
    
    # Include these with statements to understand tensorboard better
    with tf.name_scope("CNN"):

        with tf.name_scope("Input_layer"):
            
            image = tf.placeholder(dtype = tf.float32, shape = [None, config.IMG_HEIGHT, config.IMG_WIDTH], name = "image_input")
            
            # Flattens image to transform into input layer
            input_layer = tf.contrib.layers.flatten(image)
            
        with tf.name_scope("Hidden_layer"):
            
            # Examples
            conv1 = tf.layers.conv2d(
                    inputs=input_layer, 
                    filters=32,
                    kernel_size=[5, 5],
                    padding="same", 
                    activation=tf.nn.relu,
                    name = "conv1")
            
            pool1 = tf.layers.max_pooling2d(
                    inputs=conv1,
                    pool_size=[2,2],
                    strides=2,
                    name = "pool1")
            
        with tf.name_scope("Output_layer"):
            
            # Or whatever kind of layer we want to output
            output_layer = tf.layers.conv2d(
                    inputs= pool1, # or whatever last layer before output
                    filters=32,
                    kernel_size=[5, 5],
                    padding="same", 
                    activation=tf.nn.relu,
                    name = "output_layer")

        
        return output_layer


# Goal of model() is to initialize all the variables we will need in training, thus initializing the structure
# Then, wherever the variables go, goal is to minimize loss over some sort of optimization, like the Adam optimizer
def model():
    
    image = tf.placeholder(dtype = tf.float32, shape = [None, config.IMG_HEIGHT, config.IMG_WIDTH], name = "image_input")
    cnn_output = tf.get_variable(cnn(image), name = "CNN_output")
    
    sentence = tf.placeholder(dtype = tf.int32, shape = [config.BATCH_SIZE, config.MAX_CAP_LEN])
    mask = tf.placeholder(dtype = tf.int32, shape = [config.BATCH_SIZE, config.MAX_CAP_LEN])
    
    with tf.name_scope("lstm"):
            
        # Initialize lstm cell
        lstm = tf.contrib.rnn.BasitcLSTMCell()        
        
        with tf.name_scope("word_embedding"):
            
            # Creats matrix the size of the number of possible words to map the probability weights
            embedding_matrix = tf.get_variable(
                    shape = [config.NUM_TOKENS, config.DIM_EMBEDDING],  # DIM_EMBEDDING should be the same size as the max length, as the diminsions reperesent the probability the word apprears in that part of the sequence
                    name = 'embedding_weights')
            
            embedded_word = tf.nn.embedding_loopup(embedding_matrix, last_word)
        
        with tf.name_scope("Inputs"):
            
            # Initialize lstm cell
            lstm = tf.contrib.rnn.BasitcLSTMCell()
            
            # Initialize layers
            hidden_state = tf.zeros([batch_size, lstm.state_size])
            current_state = tf.zeros([batch_size, lstm.state_size])
            
            state = hidden_state, current_state
            
            # I think the 1 here keeps the embedded word from simply being appended as another facet of convolution
            current_input = tf.concat([cnn_output, embedded_word], 1)
    
            
            # Needs to be inside loop
            output, state = lstm(current_input, state)
            
            # Needs to come after everything in the loop and evaluation process so that the variables can be run with the next input
            tf.get_variable_scope().reuse_variales()

# This is what it we'll use to actually train the model, wether in a different train function or train file
def main():
    
    if not os.path.exists(config.SUMMARY_DIRECTORY):
            os.mkdir(config.SUMMARY_DIRECTORY)
    
    
    with tf.Session() as sess:
        
        summ_writer = tf.summary.FileWriter(config.SUMMARY_DIRECTORY,
                                            sess.graph)
        
        # Space to build model and other variables like optimizer
        
        
        # Need to initialize all varialbes after they are declared
        sess.run(tf.global_variables_initializer())
        
        for _ in epoch:
            for _ in batch:
                
                
                
                feed_dict = 
    
    
    
    
    
        summ_writer.close()
    

if __name__ == "__main__":
    main()
