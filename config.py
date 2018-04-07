#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle

# Image height and width for resizing in data_handling.py and inputs in model.py

SUBSET_PERCENT = .1
IMG_HEIGHT = 100
IMG_WIDTH = 100
IMG_SIZE = 100

# CNN Hyper-parameters
STRIDE = 2

#Layer 1
NUM_CHANNELS = 4
FILTER_SIZE_1 = 5
NUM_FILTERS_1 = 48


#Layer 2
FILTER_SIZE_2 = 3
NUM_FILTERS_2 = 128


#Layer 3/4
FILTER_SIZE_34 = 2
NUM_FILTERS_34 = 192


#Layer 5
FILTER_SIZE_5 = 3
NUM_FILTERS_5 = 128

#Layer 5

#Pool
POOL_STRIDES = 1
POOL_SIZE = 2

# For model.py
NUM_TOKENS = 3003
START_TOKEN_IDX = NUM_TOKENS - 2
STOP_TOKEN_IDX = NUM_TOKENS - 1
MAX_CAP_LEN = 35
CHANGE_VOCAB = False
CHANGE_IMAGES = False
BATCH_SIZE = 100  # Of images and sentances sent into model
NUM_BATCHES = 5

NUM_CNN_OUTPUTS = NUM_TOKENS	# Arbitrarily set for testing right now

# Both the lstm cell size and the dimisions for word-weight embedding correspond to the cnn output
# because the lstm cell processes a output per cell (I think) and the word weights are based on
# the image features (i.e. per this specific set of image features, what's the probability the features
# correspond to a specific word)
NUM_LSTM_UNITS = NUM_CNN_OUTPUTS 
DIM_EMBEDDING = NUM_CNN_OUTPUTS # Need to set this for embedding_matrix

LEARNING_RATE = .001
NUM_LSTM_EPOCHS = 200
USE_PRETRAINED_MODEL = False

SUMMARY_DIRECTORY = "tensorboard_summeries"   # Where to write the graphs of the model running
USE_PRETRAINED_MODEL = False
MODEL_PATH = "pretrained_models/model-1.ckpt"




