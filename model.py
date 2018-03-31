#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import config
from skimage import data
from time import time


# Borrowed from: https://github.com/jazzsaxmafia/show_attend_and_tell.tensorflow/blob/master/model_tensorflow.py
def init_weight(dim_in, dim_out, name=None, stddev=1.0):
        return tf.Variable(tf.truncated_normal([dim_in, dim_out], stddev=stddev/math.sqrt(float(dim_in))), name=name)


def cnn(image):
    
    # Include these with statements to understand tensorboard better
    with tf.name_scope("CNN"):

        with tf.name_scope("Input_layer"):
            
            # Flattens image to transform into input layer
            input_layer = tf.contrib.layers.flatten(image)
            
        with tf.name_scope("Hidden_layer"):
            
            logits = tf.contrib.layers.fully_connected(input_layer, 62, tf.nn.relu)
            
            """
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
            """
            
        with tf.name_scope("Output_layer"):
            """
            # Or whatever kind of layer we want to output
            output_layer = tf.layers.conv2d(
                    inputs= pool1, # or whatever last layer before output
                    filters=32,
                    kernel_size=[2, 2],
                    padding="same", 
                    activation=tf.nn.relu,
                    name = "output_layer")
            """
            
            output_layer = logits
        
        return output_layer

def lstm_to_word_output(embedded_words, hidden_state, cnn_output, L_0, L_h, L_z):                                          
    
    mult1 = tf.matmul(L_h, hidden_state)
    mult2 = tf.matmul(L_z, cnn_output)
    
    exponent = tf.matmul(L_0, (embedded_words, mult1, mult2))
    
    word = tf.exp(exponent)
    
    return word



# Goal of model() is to initialize all the variables we will need in training, thus initializing the structure
# Then, wherever the variables go, goal is to minimize loss over some sort of optimization, like the Adam optimizer
def train_model():
    
    # Note that these placeholders take in an entire batch of inputs, i.e. 80 images and captions
    image = tf.placeholder(dtype = tf.float32, shape = [None, config.IMG_HEIGHT, config.IMG_WIDTH], name = "image_input")
    sentences = tf.placeholder(dtype = tf.int32, shape = [config.BATCH_SIZE, config.MAX_CAP_LEN])
    mask = tf.placeholder(dtype = tf.int32, shape = [config.BATCH_SIZE, config.MAX_CAP_LEN])
    
    # Creates the actual info from the cnn to be included into the model
    cnn_output = tf.get_variable(cnn(image), name = "CNN_output")
    
    with tf.name_scope("lstm"):
        
        with tf.name_scope("initialize"):
        
            # Initialize lstm cell
            lstm = tf.contrib.rnn.BasitcLSTMCell()
            
            prior_words = tf.zeros([config.BATCH_SIZE], tf.int32)
            initial_hidden_state = tf.zeros([config.BATCH_SIZE, lstm.state_size])
            initial_current_state = tf.zeros([config.BATCH_SIZE, lstm.state_size])
            # Needed to start model
            prior_state = initial_hidden_state, initial_current_state
            
            # Initialize weights for rnn output transformation
            L_0 = init_weight(config.MAX_CAP_LEN, config.NUM_TOKENS)
            L_h = init_weight(config.NUM_TOKENS, config.MAX_CAP_LEN)
            L_Z = init_weight(config.MAX_CAP_LEN, tf.size(cnn_output))
            
            
        
        with tf.name_scope("word_embedding"):
            
            # Creats matrix the size of the number of possible words to map the probability weights
            embedding_matrix = tf.get_variable(
                    shape = [config.NUM_TOKENS, config.DIM_EMBEDDING],  # DIM_EMBEDDING should be the same size as the max length, as the diminsions reperesent the probability the word apprears in that part of the sequence
                    name = 'embedding_weights')
        
        # Need to keep track of prior words for the sentance portion
        prior_words = []
        
        # For training, need to loop through all the of possible positions
        for i in range(config.MAX_CAP_LEN):
        
            with tf.name_scope("word_embedding"):
                # Need to somehow account for prior words since we're doing a sequence to word model
                embedded_words = tf.nn.embedding_loopup(embedding_matrix, prior_words)
        
            with tf.name_scope("Inputs"):
                
                # Initialize lstm cell
                lstm = tf.contrib.rnn.BasitcLSTMCell()
                
                # I think the 1 here keeps the embedded word from simply being appended as another facet of convolution
                current_input = tf.concat([cnn_output, embedded_words], 1)
    
            # Needs to be inside loop
            output, state = lstm(current_input, prior_state)
            
            hidden_state, _ = state
            
            predicted_word = lstm_to_word_output(embedded_words, hidden_state, cnn_output,
                                                 L_0, L_h, L_z)
            
            
            prior_words.append(output_word)
            prior_state = state
            prior_output = output
            
            
            # Needs to come after everything in the loop and evaluation process so that the variables can be run with the next input
            tf.get_variable_scope().reuse_variables()

# This is what it we'll use to actually train the model, wether in a different train function or train file
def main():
    
    filepaths = ["train2014_normalized/COCO_train2014_000000103817.jpg", "train2014_normalized/COCO_train2014_000000200386.jpg"]
    
    images = []
    
    for file in filepaths:
        
        images.append(data.imread(file))
    
    #print(images[0].ndim)
    
    #plt.imshow(image)
    
    image = tf.placeholder(dtype = tf.float32, shape = [None, config.IMG_HEIGHT, config.IMG_WIDTH, 4], name = "image_input")
    
    img_output = cnn(image)
    
    with tf.Session() as sess:
        
        sess.run(tf.global_variables_initializer())
        
        feed_dict = {image: images}
        
        result = sess.run(img_output, feed_dict = feed_dict)
        
        # each result is a result per image
        print(result[0])
        print(result[1])
        print(len(result))
    
    
    
    
    
    """
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
    """
    

if __name__ == "__main__":
    main()
