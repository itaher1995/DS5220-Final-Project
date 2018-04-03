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
"""
def lstm_to_word_output(embedded_words, hidden_state, cnn_output, L_0, L_h, L_z):                                          
    
    mult1 = tf.matmul(L_h, hidden_state)
    mult2 = tf.matmul(L_z, cnn_output)
    
    exponent = tf.matmul(L_0, (embedded_words, mult1, mult2))
    
    word = tf.exp(exponent)
    
    return word
"""
class ImageCaptionGenerator():
    
    def __init__(self):
        self.hidden_state = tf.zeros([config.BATCH_SIZE, config.NUM_LSTM_UNITS], name = "global_hidden_state")

        with tf.device("/cpu:0"):
            self.embedding_matrix = tf.Variable(tf.random_uniform([config.NUM_TOKENS, config.DIM_EMBEDDING], -1.0, 1.0), name='embedding_weights')


    # Goal of model() is to initialize all the variables we will need in training, thus initializing the structure
    # Then, wherever the variables go, goal is to minimize loss over some sort of optimization, like the Adam optimizer
    def train_model(self):
        
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
            with tf.variable_scope(tf.get_variable_scope()) as scope:

                with tf.variable_scope("initialize"):
                
                    # Initialize lstm cell
                    lstm = tf.contrib.rnn.BasicLSTMCell(config.NUM_LSTM_UNITS)

                    # BATCH_SIZE x _
                    prior_word = tf.zeros([config.BATCH_SIZE], tf.int32)
                    print("Prior word:", prior_word.shape)

                    # Initialize input, BATCH_SIZE x NUM_LSTM_UNITS
                    current_input = cnn_output
                    print("Current_input", current_input.shape)

                    # The hidden state corresponds the the cnn inputs, both are BATCH_SIZE x NUM_LSTM_UNITS vectors
                    initial_hidden_state = self.hidden_state
                    initial_current_state = tf.zeros([config.BATCH_SIZE, config.NUM_LSTM_UNITS])

                    # Needed to start model, tuple of vectors
                    prior_state = initial_hidden_state, initial_current_state
                    #prior_state = m.initial_state.eval()
                
                predicted_caption = []
                loss = 0

                #with tf.variable_scope(tf.get_variable_scope()) as scope:
                # For training, need to loop through all the of possible positions
                for i in range(config.MAX_CAP_LEN):
                    
                    # Create onehot vector, or vector of entire dictionary where the word in sentance is labeled 1
                    labels = captions[:,i]
                    # BATCH_SIZE x NUM_TOKENS matrix
                    onehot_labels = tf.one_hot(labels, config.NUM_TOKENS,
                                               on_value = 1, off_value = 0,
                                               name = "onehot_labels")
                    #print("onehot:", onehot_labels.shape)

                    if i != 0:
                        with tf.variable_scope("word_embedding"):
                            # Can't be run on a gpu for some reason
                            with tf.device("/cpu:0"):
                                # Accounts for the one_hot vector
                                # BATCH_SIZE x NUM_TOKENS matrix
                                prior_word_probs = tf.nn.embedding_lookup(self.embedding_matrix, prior_word)
                            current_input = tf.multiply(prior_word_probs, tf.cast(onehot_labels, tf.float32))
                    
                    with tf.variable_scope("lstm_function"):
                        # This line executes the actual gates of lstm to update values, output is BATCH_SIZE x NUM_LSTM_UNITS
                        output, state = lstm(current_input, prior_state)

                        _, current_state = state
                    
                    with tf.variable_scope("lstm_output"):
                        # BATCH_SIZE x NUM_LSTM_UNITS
                        m_t = tf.multiply(output, current_state)

                        #logits = 

                        # BATCH_SIZE x NUM_LSTM_UNITS
                        p_t = tf.nn.softmax(m_t, name = "word_probabilities")

                    # Calculates the loss for the training, performs it in a slightly different manner than paper
                    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = p_t, labels = captions[:,i])
                    current_loss = tf.reduce_sum(cross_entropy)
                    loss = loss + current_loss
                    #print("Loop", i, "Loss", loss)

                    predicted_word = tf.argmax(p_t, 1)

                    predicted_caption.append(predicted_word)

                    prior_word = captions[:, i-1]
                    prior_state = state
                    prior_output = output
                    
                    # Needs to come after everything in the loop and evaluation process so that the variables can be run with the next input
                    tf.get_variable_scope().reuse_variables()
        
        hidden_state, _ = prior_state
        self.hidden_state = hidden_state
        print(2)
        return loss, images, captions


        def use_model(self):
            # Tensor to return and demonstrate program works
            #tester = tf.constant("it works")
            
            # Note that these placeholders take in an entire batch of inputs, i.e. 80 images and captions
            images = tf.placeholder(dtype = tf.float32, shape = [None, config.IMG_HEIGHT, config.IMG_WIDTH, 4], name = "image_input")

            # Creates the actual info from the cnn to be included into the model
            cnn_output = cnn(images)
            
            with tf.name_scope("lstm"):
                with tf.variable_scope(tf.get_variable_scope()) as scope:

                    with tf.variable_scope("initialize"):
                    
                        # Initialize lstm cell
                        lstm = tf.contrib.rnn.BasicLSTMCell(config.NUM_LSTM_UNITS)

                        # Initialize input, BATCH_SIZE x NUM_LSTM_UNITS
                        current_input = cnn_output

                        # The hidden state corresponds the the cnn inputs, both are BATCH_SIZE x NUM_LSTM_UNITS vectors
                        initial_hidden_state = self.hidden_state
                        initial_current_state = self.current_state

                        # Needed to start model, tuple of vectors
                        prior_state = initial_hidden_state, initial_current_state
                        #prior_state = m.initial_state.eval()
                    
                    predicted_caption = []

                    # For training, need to loop through all the of possible positions
                    for i in range(config.MAX_CAP_LEN):
                        
                        if i != 0:
                            with tf.variable_scope("word_embedding"):
                                # Can't be run on a gpu for some reason
                                with tf.device("/cpu:0"):
                                    # Accounts for the one_hot vector
                                    # BATCH_SIZE x NUM_TOKENS matrix
                                    prior_word_probs = tf.nn.embedding_lookup(self.embedding_matrix, prior_word)
                                current_input = tf.multiply(prior_word_probs, tf.cast(onehot_labels, tf.float32))
                        
                        with tf.variable_scope("lstm_function"):
                            # This line executes the actual gates of lstm to update values, output is BATCH_SIZE x NUM_LSTM_UNITS
                            output, state = lstm(current_input, prior_state)

                            _, current_state = state
                        
                        with tf.variable_scope("lstm_output"):
                            # BATCH_SIZE x NUM_LSTM_UNITS
                            m_t = tf.multiply(output, current_state)

                            # BATCH_SIZE x NUM_LSTM_UNITS
                            p_t = tf.nn.softmax(m_t, name = "word_probabilities")

                        predicted_word = tf.argmax(p_t, 1)

                        predicted_caption.append(predicted_word)

                        prior_word = predicted_word
                        prior_state = state
                        prior_output = output
                        
                        # Needs to come after everything in the loop and evaluation process so that the variables can be run with the next input
                        tf.get_variable_scope().reuse_variables()

            return generated_captions, images, captions
        
        
# This is what it we'll use to actually train the model, wether in a different train function or train file
def train():
    
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
            #caption_data.append(sentance)
            caption_data.append(list(range(len(sentance))))
            break
        
    print(caption_data)
    print(len(image_data))
    
    # Allows us to save training sessions
    if not os.path.exists(config.SUMMARY_DIRECTORY):
            os.mkdir(config.SUMMARY_DIRECTORY)


    with tf.Session() as sess:
        model = ImageCaptionGenerator()

        summ_writer = tf.summary.FileWriter(config.SUMMARY_DIRECTORY,
                                                sess.graph)
        #saver = tf.train.Saver(max_to_keep=50)
        loss, images, captions = model.train_model()


        print(1)
        train_op = tf.train.AdamOptimizer(config.LEARNING_RATE).minimize(loss)
        print(3)
        # This is where the number of epochs for the LSTM are controlled
        for epoch in range(config.NUM_LSTM_EPOCHS):
            
            sess.run(tf.global_variables_initializer())
            
            # Need to pad captions
            caption_data = string_padder(caption_data)
            
            feed_dict = {images: image_data,
                         captions: caption_data}
                         #m.initial_state = initial_state}
            print(4)
            _, loss_result = sess.run([train_op, loss], feed_dict = feed_dict)
            
            # each result is a result per image
            print(loss_result)
            #saver.save(sess, os.path.join(confg.MODEL_PATH, 'model'), global_step=epoch)
    
    summ_writer.close()
    
def test():

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

    generated_captions, images, captions = use_model()

    with tf.Session() as sess:
        
        #model = tf.train.Saver()
        #model.restore(sess, config.MODEL_PATH)

        sess.run(tf.global_variables_initializer())
        
        # Need to pad captions
        caption_data = string_padder(caption_data)
        
        feed_dict = {images: image_data,
                     initial_hidden_state: self.hidden_state}
                     #m.initial_state = initial_state}
        
        result = sess.run(generated_captions, feed_dict = feed_dict)
        
        # each result is a result per image
        print(result)
        

def main():
    train()

if __name__ == "__main__":
    main()
