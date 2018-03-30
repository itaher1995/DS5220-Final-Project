#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import matplotlib as plt
import config
from time import time


def example_weight_initialization():
    
    n_features = X_values.shape[1]
    n_classes = len(set(y_flat))
    weights_shape = (n_features, n_classes)
    
    W = tf.Variable(dtype=tf.float32, initial_value=tf.random_normal(weights_shape))  # Weights of the model



# Helper function to simply run a variable
def run_variable(variable):
    
    tf.initialize_all_variables()
    
    with tf.Session() as sess:
        
        return sess.run(variable)



def cnn():
    
    # Include these with statements to understand tensorboard better
    with tf.name_scope("CNN"):

        with tf.name_scope("Input_layer"):
            
            # Below is an example from other code
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
    
    cnn_output = tf.get_variable(cnn(), name = "CNN_output")
    
    with tf.name_scope("lstm"):
            
        # Initialize lstm cell
        lstm = tf.contrib.rnn.BasitcLSTMCell()        
        
        with tf.name_scope("word_embedding"):
            
            embedding_matrix = tf.get_variable(
                    shape = [config.NUM_TOKENS, config.DIM_EMBEDDING],
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
    
            
    output, state = lstm(current_input, state)

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
