#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle

# Image height and width for resizing in data_handling.py and inputs in model.py

SUBSET_PERCENT = .1
IMG_HEIGHT = 28
IMG_WIDTH = 28
IMG_SIZE = 28

# CNN Hyper-parameters
NUM_CHANNELS = 4
FILTER_SIZE = 3
NUM_FILTERS = 5
FULLY_CON_LAYER_SIZE = 128
STRIDES = 1
POOL_SIZE = 2

# For model.py

with open('tokens.pkl', 'rb') as f: 
    NUM_TOKENS = len(pickle.load(f))
NUM_TOKENS = 3000
MAX_CAP_LEN = 49   # Optained from a print out in tokenize_captions() of data_handling.py    
BATCH_SIZE = 5  # Of images and sentances sent into model

NUM_CNN_OUTPUTS = NUM_TOKENS	# Arbitrarily set for testing right now

# Both the lstm cell size and the dimisions for word-weight embedding correspond to the cnn output
# because the lstm cell processes a output per cell (I think) and the word weights are based on
# the image features (i.e. per this specific set of image features, what's the probability the features
# correspond to a specific word)
NUM_LSTM_UNITS = NUM_CNN_OUTPUTS
DIM_EMBEDDING = NUM_CNN_OUTPUTS  # Need to set this for embedding_matrix

LEARNING_RATE = .1
NUM_LSTM_EPOCHS = 1


SUMMARY_DIRECTORY = "pretrained_models"   # Where to write the graphs of the model running
MODEL_PATH = ""




