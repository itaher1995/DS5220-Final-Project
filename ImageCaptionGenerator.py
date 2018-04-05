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

    # Allows us to save training sessions
    if not os.path.exists(config.SUMMARY_DIRECTORY):
            os.mkdir(config.SUMMARY_DIRECTORY)

    if not os.path.exists('pretrained_models'):
            os.mkdir('pretrained_models')


    with tf.Session() as sess:
        model = ImageDecoder()

        loss, summary, images, captions = model.buildModel()

        saver = tf.train.Saver(max_to_keep=50)

        print(1)
        train_op = tf.train.AdamOptimizer(config.LEARNING_RATE).minimize(loss)
        print(3)
        # This is where the number of epochs for the LSTM are controlled
        sess.run(tf.global_variables_initializer())

        if config.USE_PRETRAINED_MODEL == True:
            print("Restoring pretrained model")
            saver.restore(sess, config.MODEL_PATH)

        summ_writer = tf.summary.FileWriter(config.SUMMARY_DIRECTORY,
                                                sess.graph)
        
        for batch in range(config.NUM_BATCHES):
            for epoch in range(config.NUM_LSTM_EPOCHS):
                image_data, caption_data = hf.getImageBatchFromPickle("train_data-1.pkl", "train2014_normalized")
                print(caption_data)
                print("Image batch size", len(image_data))
                
                feed_dict = {images: image_data,
                             captions: caption_data}
                             #m.initial_state = initial_state}

                _, results = sess.run([train_op, summary], feed_dict = feed_dict)
                
                # each result is a result per image
                
                summ_writer.add_summary(results, epoch)
                saver.save(sess, config.MODEL_PATH, global_step=epoch)
    
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
