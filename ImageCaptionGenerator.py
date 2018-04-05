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
from skimage import data

def idx_to_word_translate(idx_matrix, images):
    idx_to_word = pd.read_pickle("idx_to_word-1.pkl")
    new_caps = [[idx_to_word[idx] for idx in idx_cap] for idx_cap in idx_matrix]

def getImageBatchFromPickle(pkl, data_directory):
    '''
    Gets the image batch and the corresponding captions. Takes a pickle file
    and then randomly select batchSize indices. Then we locate the those
    indices in the dataframe and output a dataframe with just the file_name
    and the idx_captions.
    '''
    df = pd.read_pickle(pkl)
    imageBatchIndex = np.random.choice(df.index,size=config.BATCH_SIZE)
    imageBatch=df.iloc[imageBatchIndex][['file_name','mapped_captions']]

    while (len(imageBatch['file_name'].unique()) < config.BATCH_SIZE):
        imageBatchIndex = np.random.choice(df.index,size=config.BATCH_SIZE)
        imageBatch=df.iloc[imageBatchIndex][['file_name','mapped_captions']]

    # Just gets a couple images and captions for testing right now
    image_filenames = list(imageBatch['file_name'])
    #print(image_filenames)
    
    images = []
    captions = []
    
    for f in image_filenames:
        filepath = os.path.join(data_directory, f)
        images.append(data.imread(filepath))
        cap_row = imageBatch[imageBatch['file_name'] == f]
        captions.append(list(cap_row['mapped_captions'].item()))
    
    return images, captions

def is_nonzero(num):
    if num > 0:
        return 1
    else:
        return 0

def train(filterSize,numFilters,strides,k,eta):
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

        loss, summary, pdm, images, captions, masks = model.buildModel(filterSize,numFilters,strides,k)

        saver = tf.train.Saver(max_to_keep=50)

        train_op = tf.train.AdamOptimizer(eta).minimize(loss)

        # This is where the number of epochs for the LSTM are controlled
        sess.run(tf.global_variables_initializer())

        if config.USE_PRETRAINED_MODEL == True:
            print("Restoring pretrained model")
            saver.restore(sess, config.MODEL_PATH)

        summ_writer = tf.summary.FileWriter(config.SUMMARY_DIRECTORY,
                                                sess.graph)
        start = time()

        prior_loss = 0
        for epoch in range(config.NUM_LSTM_EPOCHS):
            image_data, caption_data = getImageBatchFromPickle("train_data-1.pkl", "train2014_normalized")
            
            mask_matrix = [[is_nonzero(idx) for idx in idx_cap] for idx_cap in caption_data]

            feed_dict = {images: image_data,
                         captions: caption_data,
                         masks: mask_matrix}
                         #m.initial_state = initial_state}

            _, results, loss_result, pred_caps = sess.run([train_op, summary, loss, pdm], feed_dict = feed_dict)
            
            # each result is a result per image
            
            summ_writer.add_summary(results, epoch)
            saver.save(sess, config.MODEL_PATH, global_step=epoch)

            print("Num epochs", epoch,"   time", round(time() - start))
            print("Loss", loss_result, "   Prior loss", prior_loss,"   difference", prior_loss - loss_result)
            print()
            prior_loss = loss_result
            
    summ_writer.close()

    idx_to_word_translate(pred_caps, image_data)

    return loss_result



def test(X):
    return "hey"

def gridSearch(filterSize,numFilters,strides,k,eta):
    '''
    Conducts a grid search on all set hyperparameters.
    '''
    bestLoss = 100000
    for fs in filterSize:
        for nf in numFilters:
            for s in strides:
                for poolsize in k:
                    for e in eta:
                        loss = train(fs,nf,s,poolsize,e)
                        if loss<bestLoss:
                            bestEta = e
                            bestK = poolsize
                            bestS = s
                            bestNF = nf
                            bestFS = fs
    return bestEta, bestK, bestS, bestNF, bestFS
                            
                        
                        
                        
 
def meanBLEUScore(candidate3DArray,groundTruth2DArray):
    '''
    Calculates the mean BLEU Score over all images. Will eventually take in an
    array that contains a 2D array of predicted words that correspond to a 
    sentence S and another array of arrays with the truth captions.
    '''
    
    return sum([sentence_bleu(candidate3DArray[i],groundTruth2DArray[i]) for i in range(len(groundTruth2DArray))])/len(groundTruth2DArray)
