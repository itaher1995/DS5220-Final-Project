#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import config
import pandas as pd
from skimage import data
from time import time
import os

def string_padder(caption_matrix):

    padded_matrix = []

    for i in range(len(caption_matrix)):

        caption_array = caption_matrix[i]

        # Want to pad up to max length
        num_pad = config.MAX_CAP_LEN - len(caption_array)

        # Pads strings with zero's. Accounted for this when we mapped the word idx's starting at 1
        padded_caption = np.pad(caption_array, (0,num_pad), 'constant', constant_values = (0,0))

        padded_matrix.append(padded_caption)

    return padded_matrix


def cnn(image):
    
    # Include these with statements to understand tensorboard better
    with tf.name_scope("CNN"):

        with tf.name_scope("Input_layer"):
            
            # Flattens image to transform into input layer
            input_layer = tf.contrib.layers.flatten(image)
            
        with tf.name_scope("Hidden_layer"):
            
            logits = tf.contrib.layers.fully_connected(input_layer, config.NUM_CNN_OUTPUTS, tf.nn.relu)
            
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
        
        return tf.reshape(output_layer, shape = [config.BATCH_SIZE, config.NUM_CNN_OUTPUTS])

def lstm_to_word_output(embedded_words, hidden_state, cnn_output, L_0, L_h, L_z):                                          
    
    mult1 = tf.matmul(L_h, hidden_state)
    mult2 = tf.matmul(L_z, cnn_output)
    
    exponent = tf.matmul(L_0, (embedded_words, mult1, mult2))
    
    word = tf.exp(exponent)
    
    return word



# Goal of model() is to initialize all the variables we will need in training, thus initializing the structure
# Then, wherever the variables go, goal is to minimize loss over some sort of optimization, like the Adam optimizer
def train_model():
    
    # Tensor to return and demonstrate program works
    #tester = tf.constant("it works")
    
    # Note that these placeholders take in an entire batch of inputs, i.e. 80 images and captions
    images = tf.placeholder(dtype = tf.float32, shape = [None, config.IMG_HEIGHT, config.IMG_WIDTH, 4], name = "image_input")
    captions = tf.placeholder(dtype = tf.int32, shape = [config.BATCH_SIZE, config.MAX_CAP_LEN], name = "input_captions")
    
    # To include later if we want to help training
    # mask = tf.placeholder(dtype = tf.int32, shape = [config.BATCH_SIZE, config.MAX_CAP_LEN])
    
    # Creates the actual info from the cnn to be included into the model
    cnn_output = cnn(images)
    
    with tf.name_scope("lstm"):
        
        with tf.name_scope("initialize"):
        
            # Initialize lstm cell
            lstm = tf.contrib.rnn.BasicLSTMCell(config.NUM_LSTM_UNITS)

            prior_word = tf.zeros([config.BATCH_SIZE], tf.int32)

            # Initialize input
            current_input = cnn_output

            # The hidden state corresponds the the cnn inputs
            initial_hidden_state = tf.zeros([config.BATCH_SIZE, config.NUM_LSTM_UNITS])
            initial_current_state = tf.zeros([config.BATCH_SIZE, config.NUM_LSTM_UNITS])
            # Needed to start model
            prior_state = initial_hidden_state, initial_current_state
            
        with tf.name_scope("word_embedding"):
            
            # Creats matrix the size of the number of possible words to map the probability weights
            embedding_matrix = tf.get_variable(
                    shape = [config.NUM_TOKENS, config.DIM_EMBEDDING],  # DIM_EMBEDDING should be the same size as the max length, as the diminsions reperesent the probability the word apprears in that part of the sequence
                    name = 'embedding_weights')
        
        predicted_caption = []

        # For training, need to loop through all the of possible positions
        for i in range(config.MAX_CAP_LEN):
            
            if i != 0:
                with tf.name_scope("word_embedding"):
                    # Can't be run on a gpu for some reason
                    with tf.device("/cpu:0"):
                        # Need to somehow account for prior words since we're doing a sequence to word model
                        embedded_word = tf.nn.embedding_lookup(embedding_matrix, prior_word)
    
            # This line executes the actual gates of lstm to update values
            output, state = lstm(current_input, prior_state)

            _, current_state = state
            
            m_t = tf.multiply(output, current_state)

            p_t = tf.nn.softmax(m_t)

            print(m_t.shape)
            print(p_t.shape)

            return p_t, images, captions

            predicted_caption.append(predicted_word)
            
            current_input = captions[:, i-1]
            prior_state = state
            prior_output = output
            
            
            # Needs to come after everything in the loop and evaluation process so that the variables can be run with the next input
            tf.get_variable_scope().reuse_variables()
        
        return predicted_caption, images, captions
    
    
    
    
# This is what it we'll use to actually train the model, wether in a different train function or train file
def main():
    
    # reads in necessary image data
    img_data = pd.read_pickle("train_data.pkl")
    
    # Just gets a couple images and captions for testing right now
    image_filenames = list(img_data['file_name'][0:config.BATCH_SIZE])
    print(image_filenames)
    
    data_directory = "train2014_normalized"
    
    image_data = []
    caption_data = []
    
    for f in image_filenames:
        
        filepath = os.path.join(data_directory, f)
        
        image_data.append(data.imread(filepath))
        
        cap_row = img_data[img_data['file_name'] == f].copy()
        
        # Note that annotations is a pd.Series()
        idx_captions = cap_row['idx_caption_matrix'].item()        
        
        # really only want one annotation per image for testing
        for sentance in idx_captions:
            caption_data.append(sentance)
            break
        
    print(caption_data)
    print(len(image_data))

    model, images, captions = train_model()
    
    with tf.Session() as sess:
        
        sess.run(tf.global_variables_initializer())
        
        # Need to pad captions
        caption_data = string_padder(caption_data)
        
        feed_dict = {images: image_data,
                     captions: caption_data}
        
        result = sess.run(model, feed_dict = feed_dict)
        
        # each result is a result per image
        print(result)
    
    
    
    
    
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
