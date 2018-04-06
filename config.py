#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle

# Image height and width for resizing in data_handling.py and inputs in model.py

SUBSET_PERCENT = .1
IMG_HEIGHT = 100
IMG_WIDTH = 100
IMG_SIZE = 100

# CNN Hyper-parameters
NUM_CHANNELS = 4
FILTER_SIZE = 3
NUM_FILTERS = 5
FULLY_CON_LAYER_SIZE = 128
STRIDES = 1
POOL_SIZE = 2

# For model.py
NUM_TOKENS = 3004
START_TOKEN_IDX = NUM_TOKENS - 3
STOP_TOKEN_IDX = NUM_TOKENS - 2
PAD_TOKEN_IDX = NUM_TOKENS -1
MAX_CAP_LEN = 35
CHANGE_VOCAB = False
CHANGE_IMAGES = False
BATCH_SIZE = 15  # Of images and sentances sent into model
NUM_BATCHES = 5

NUM_CNN_OUTPUTS = NUM_TOKENS	# Arbitrarily set for testing right now

# Both the lstm cell size and the dimisions for word-weight embedding correspond to the cnn output
# because the lstm cell processes a output per cell (I think) and the word weights are based on
# the image features (i.e. per this specific set of image features, what's the probability the features
# correspond to a specific word)
NUM_LSTM_UNITS = NUM_CNN_OUTPUTS 
DIM_EMBEDDING = NUM_CNN_OUTPUTS # Need to set this for embedding_matrix

LEARNING_RATE = .001
NUM_LSTM_EPOCHS = 20
USE_PRETRAINED_MODEL = False

SUMMARY_DIRECTORY = "tensorboard_summaries"   # Where to write the graphs of the model running
USE_PRETRAINED_MODEL = False
MODEL_PATH = "pretrained_models/model-1.ckpt"


def test(self,filterSize,numFilters,strides,k):
        images = tf.placeholder(dtype = tf.float32, shape = [None, config.IMG_HEIGHT, config.IMG_WIDTH, 4], name = "image_input")
        
        # Build CNN
        with tf.name_scope("Image_Encoder"):
            chunk, weights = self.createCNNChunk(images,config.NUM_CHANNELS,
                                                 filterSize, numFilters,
                                                 strides, k,1)

            flattenLayer, numFeatures = self.flatten(chunk)
            cnnOutput = self.fullyConnectedComponent(flattenLayer, numFeatures,
                                                             config.NUM_CNN_OUTPUTS)
        #with tf.name_scope("Image_Decoder"):
        with tf.variable_scope(tf.get_variable_scope()) as scope:

            with tf.variable_scope("LSTM"):
            
                # Initialize lstm cell
                lstm = tf.contrib.rnn.LayerNormBasicLSTMCell(config.NUM_LSTM_UNITS)

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
            
            output, state = lstm(current_input, prior_state)
            with tf.device("/cpu:0"):
                    # Accounts for the one_hot vector
                    # BATCH_SIZE x NUM_TOKENS matrix
                    current_input = tf.nn.embedding_lookup(self.embedding_matrix, tf.zeros[config.BATCH_SIZE, 1])

            predicted_caption = []
            predictions_correct = []

            
            # For training, need to loop through all the of possible positions
            for i in range(config.MAX_CAP_LEN + 2):
                
                with tf.variable_scope("lstm_function"):
                    # This line executes the actual gates of lstm to update values, output is BATCH_SIZE x NUM_LSTM_UNITS
                    output, state = lstm(current_input, prior_state)

                    _, current_state = state
                
                # Calculates the loss for the training, performs it in a slightly different manner than paper
                logit_words = tf.matmul(output, self.embed_word_W) + self.embed_word_b # (batch_size, n_words)

                ## need to fix masks i vs captions i - 1, something seems wrong

                predicted_word = tf.argmax(logit_words, 1)

                with tf.device("/cpu:0"):
                    # Accounts for the one_hot vector
                    # BATCH_SIZE x NUM_TOKENS matrix
                    current_input = tf.nn.embedding_lookup(self.embedding_matrix, predicted_word)
                
                current_input += self.bemb

                predicted_caption.append(predicted_word)
                
                # Needs to come after everything in the loop and evaluation process so that the variables can be run with the next input
                tf.get_variable_scope().reuse_variables()
    
        hidden_state, _ = prior_state

        return predicted_caption, images

