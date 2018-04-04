# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 17:29:04 2018

@author: ibiyt
"""

from ImageDecoder import ImageDecoder
import tensorflow as tf
import config
from nltk.translate.bleu_score import sentence_bleu
import pandas as pd
import os
from time import time
import numpy as np
import helperFunctions as hf


def train():
    '''
    NOTE: will eventually read in data.
    '''
    tf.reset_default_graph()
    
    image_data, caption_data = hf.getImageBatchFromPickle("train_data-1.pkl", "train2014_normalized")
        
    print(caption_data)
    print("Image batch size", len(image_data))

    # Allows us to save training sessions
    if not os.path.exists(config.SUMMARY_DIRECTORY):
            os.mkdir(config.SUMMARY_DIRECTORY)


    with tf.Session() as sess:
        model = ImageDecoder()

        summ_writer = tf.summary.FileWriter(config.SUMMARY_DIRECTORY,
                                                sess.graph)
        #saver = tf.train.Saver(max_to_keep=50)
        loss, images, captions = model.buildModel()

        print(1)
        train_op = tf.train.AdamOptimizer(config.LEARNING_RATE).minimize(loss)
        print(3)
        # This is where the number of epochs for the LSTM are controlled
        for epoch in range(config.NUM_LSTM_EPOCHS):
            
            sess.run(tf.global_variables_initializer())
            
            feed_dict = {images: image_data,
                         captions: caption_data}
                         #m.initial_state = initial_state}
            print(4)
            summary, loss_result = sess.run([train_op, loss], feed_dict = feed_dict)
            
            # each result is a result per image
            print(loss_result)
            summ_writer.add_summary(summary, epoch)
            #saver.save(sess, os.path.join(confg.MODEL_PATH, 'model'), global_step=epoch)
    
    summ_writer.close()
    
def test(X):
    return "hey"
    
def meanBLEUScore(candidate3DArray,groundTruth2DArray):
    '''
    Calculates the mean BLEU Score over all images. Will eventually take in an
    array that contains a 2D array of predicted words that correspond to a 
    sentence S and another array of arrays with the truth captions.
    '''
    
    return sum([sentence_bleu(candidate3DArray[i],groundTruth2DArray[i]) for i in range(len(groundTruth2DArray))])/len(groundTruth2DArray)
