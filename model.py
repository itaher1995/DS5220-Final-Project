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
           
            x = tf.placeholder(dtype = tf.float32, shape = [None, config.IMG_HEIGHT, config.IMG_WIDTH], name = "image_input")
            
            
            
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




def model(image):
    
    cnn_output = tf.placeholder(cnn(image), name = "CNN_output")
    
    
    
    with tf.name_scope("lstm"):
    
        with tf.name_scope("Inputs"):
            
            current_input = tf.concat([cnn_output, "placeholder for word"], 1)
            
    
    
    output, state = lstm(current_input, state)




def main():
    CNN()
        
    
    

if __name__ == "__main__":
    main()
