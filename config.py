#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle


# Image height and width for resizing in data_handling.py and inputs in model.py

IMG_HEIGHT = 28

IMG_WIDTH = 28




# For model.py

with open('tokens.pkl', 'rb') as f:
    
    NUM_TOKENS = len(pickle.load(f))
    
DIM_EMBEDDING = 2   # Need to set this for 




