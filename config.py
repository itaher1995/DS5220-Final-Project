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
NUM_TOKENS = 3003
START_TOKEN_IDX = NUM_TOKENS + 1
STOP_TOKEN_IDX = NUM_TOKENS + 2
MAX_CAP_LEN = 35
CHANGE_VOCAB = False
CHANGE_IMAGES = False
BATCH_SIZE = 5  # Of images and sentances sent into model
NUM_BATCHES = 5

NUM_CNN_OUTPUTS = NUM_TOKENS	# Arbitrarily set for testing right now

# Both the lstm cell size and the dimisions for word-weight embedding correspond to the cnn output
# because the lstm cell processes a output per cell (I think) and the word weights are based on
# the image features (i.e. per this specific set of image features, what's the probability the features
# correspond to a specific word)
NUM_LSTM_UNITS = NUM_CNN_OUTPUTS 
DIM_EMBEDDING = NUM_CNN_OUTPUTS # Need to set this for embedding_matrix

LEARNING_RATE = .1
NUM_LSTM_EPOCHS = 1
USE_PRETRAINED_MODEL = False

SUMMARY_DIRECTORY = "pretrained_models"   # Where to write the graphs of the model running
MODEL_PATH = "pretrained_models/model-1.ckpt"




