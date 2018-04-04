# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 16:44:59 2018

@author: ibiyt
"""

import tensorflow as tf
import config
import math
from ImageEncoder import ImageEncoder

class ImageDecoder():
    '''
    Implementation of a caption generator, which is a neural network where our
    encoder is a convolutional neural network with chunked layers of the 
    convolution step, max pooling and normalization step followed by a
    recurrent neural network with LSTM.
    
    The functionality is as such. We will call this model, which will in turn
    build an LSTM neural network, which will then call our Image Encoder.
    
    Model -> LSTM -> CNN
    
    INPUT: Captions and Images
    OUTPUT: Captions
    '''
    
    def __init__(self):

        self.hidden_state = tf.zeros([config.BATCH_SIZE, config.NUM_LSTM_UNITS], name = "global_hidden_state")

        with tf.device("/cpu:0"):
            self.hidden_state = self.init_weight(config.BATCH_SIZE, config.NUM_LSTM_UNITS, name = "global_hidden_state")
            self.embedding_matrix = tf.Variable(tf.random_uniform([config.NUM_TOKENS, config.DIM_EMBEDDING], -1.0, 1.0), name='embedding_weights')
    
    def init_weight(self, dim_in, dim_out, name=None, stddev=1.0):
        return tf.Variable(tf.truncated_normal([dim_in, dim_out], stddev=stddev/math.sqrt(float(dim_in))), name=name)
    
    def buildModel(self):
        '''
        Builds the LSTM and CNN and links them together.
        '''
        # Tensor to return and demonstrate program works
        #tester = tf.constant("it works")
        
        # Note that these placeholders take in an entire batch of inputs, i.e. 80 images and captions
        images = tf.placeholder(dtype = tf.float32, shape = [None, config.IMG_HEIGHT, config.IMG_WIDTH, 4], name = "image_input")
        captions = tf.placeholder(dtype = tf.int32, shape = [config.BATCH_SIZE, config.MAX_CAP_LEN + 2], name = "input_captions")
        
        # To include later if we want to help training
        # mask = tf.placeholder(dtype = tf.int32, shape = [config.BATCH_SIZE, config.MAX_CAP_LEN])
        
        # Build CNN
        with tf.name_scope("Image_Encoder"):
            cnn = ImageEncoder()
            chunk, weights = cnn.createCNNChunk(images,config.NUM_CHANNELS,
                                                config.FILTER_SIZE, config.NUM_FILTERS,
                                                config.STRIDES, config.POOL_SIZE)
            
            flattenLayer, numFeatures = cnn.flatten(chunk)
            cnnOutput = cnn.fullyConnectedComponent(flattenLayer, numFeatures,
                                                             config.NUM_CNN_OUTPUTS)
        
        #Build RNN
        with tf.name_scope("Image_Decoder"):
            with tf.variable_scope(tf.get_variable_scope()) as scope:

                with tf.variable_scope("LSTM"):
                
                    # Initialize lstm cell
                    lstm = tf.contrib.rnn.BasicLSTMCell(config.NUM_LSTM_UNITS)

                    # BATCH_SIZE x _
                    prior_word = tf.zeros([config.BATCH_SIZE], tf.int32)
                    print("Prior word:", prior_word.shape)

                    # Initialize input, BATCH_SIZE x NUM_LSTM_UNITS
                    current_input = cnnOutput
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
                for i in range(config.MAX_CAP_LEN + 2):
                    
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
                    #prior_output = output
                    
                    # Needs to come after everything in the loop and evaluation process so that the variables can be run with the next input
                    tf.get_variable_scope().reuse_variables()
        
        hidden_state, _ = prior_state
        self.hidden_state = hidden_state
        print(2)
        return loss, images, captions
    
    def test(self):
        return "Incomplete"
        
        
        
    