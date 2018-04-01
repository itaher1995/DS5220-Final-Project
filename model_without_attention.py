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
            
        with tf.name_scope("Output_layer"):
            
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
        loss = 0

        # For training, need to loop through all the of possible positions
        for i in range(config.MAX_CAP_LEN):
            
            """
            Paused at creating onehot labels. Need to successfully create labels, then make sure it works correctly with
            the cross_entropy
            """

            # Taken straight from code in other program, might need to change up
            labels = tf.expand_dims(captions[:,i], 1)
            indices = tf.expand_dims(tf.range(0, config.BATCH_SIZE, 1), 1)
            concated = tf.concat(1, [indices, labels])
            onehot_labels = tf.sparse_to_dense( concated, tf.pack([config.BATCH_SIZE, config.NUM_TOKENS]), 1.0, 0.0)

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

            # Calculates the loss for the training
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logit_words, onehot_labels)
            current_loss = tf.reduce_sum(cross_entropy)
            loss = loss + current_loss

            predicted_word = tf.argmax(p_t, 1)

            predicted_caption.append(predicted_word)

            current_input = captions[:, i-1]
            prior_state = state
            prior_output = output
            
            
            # Needs to come after everything in the loop and evaluation process so that the variables can be run with the next input
            tf.get_variable_scope().reuse_variables()
        
        return loss, images, captions
    
    
    
    
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

    loss, images, captions = train_model()

    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    
    with tf.Session() as sess:
        
        sess.run(tf.global_variables_initializer())
        
        # Need to pad captions
        caption_data = string_padder(caption_data)
        
        feed_dict = {images: image_data,
                     captions: caption_data}
        
        _, loss_result = sess.run([train_op, loss], feed_dict = feed_dict)
        
        # each result is a result per image
        print(loss_result)
    
    
    
    
    
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
