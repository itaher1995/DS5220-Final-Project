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
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file



def idx_to_word_translate(idx_matrix, images):
    print(idx_matrix)
    idx_to_word = pd.read_pickle("idx_to_word-1.pkl")
    #print(type(idx_to_word))
    new_caps = np.array([np.array([idx_to_word[idx] for idx in idx_cap]) for idx_cap in idx_matrix]).T
    return new_caps

def image_captions(pkl, filenames):
    df = pd.read_pickle(pkl)

    indicies = []
    for f in filenames:
        temp_df = df[df['file_name'] == f]
        indicies = indicies + list(temp_df.index)
    
    return df.iloc[indicies][['file_name', 'caption', 'mapped_captions']]


def getImageBatchFromPickle(pkl, data_directory):
    '''
    Gets the image batch and the corresponding captions. Takes a pickle file
    and then randomly select batchSize indices. Then we locate the those
    indices in the dataframe and output a dataframe with just the file_name
    and the idx_captions.
    '''
    df = pd.read_pickle(pkl)

    while True:
        imageBatchIndex = np.random.choice(df.index,size=config.BATCH_SIZE)
        imageBatch=df.iloc[imageBatchIndex][['file_name','mapped_captions']]


        while (len(imageBatch['file_name'].unique()) < config.BATCH_SIZE):
            imageBatchIndex = np.random.choice(df.index,size=config.BATCH_SIZE)
            imageBatch=df.iloc[imageBatchIndex][['file_name','mapped_captions']]

        # Just gets a couple images and captions for testing right now
        image_filenames = list(imageBatch['file_name'])

        if set(image_filenames).intersection(os.listdir(data_directory))==set(image_filenames):
            break

    #print(image_filenames)
    
    images = []
    captions = []
    
    for f in image_filenames:
        filepath = os.path.join(data_directory, f)
        images.append(data.imread(filepath))
        cap_row = imageBatch[imageBatch['file_name'] == f]
        captions.append(list(cap_row['mapped_captions'].item()))
    
    return images, captions, image_filenames

def is_nonzero(num):
    if num == config.PAD_TOKEN_IDX:
        return 0
    else:
        return 1

def train(filterSize_1,
        numFilters_1,
        filterSize_2,
        numFilters_2,
        filterSize_34,
        numFilters_34,
        filterSize_5,
        numFilters_5,
        strides,
        k,
        eta):
    '''
    NOTE: will eventually read in data.
    '''
    tf.reset_default_graph()
    # Allows us to save training sessions
    if not os.path.exists(config.SUMMARY_DIRECTORY):
            os.mkdir(config.SUMMARY_DIRECTORY)

    if not os.path.exists('pretrained_models_TAHER'):
            os.mkdir('pretrained_models_TAHER')

    # Creates path for new model directory according to the parameters being searched
    hyperparameters = [filterSize_1,numFilters_1,filterSize_2,numFilters_2,filterSize_34,numFilters_34,
        filterSize_5,numFilters_5,strides,k,eta]
    hyperparameters = [str(hp) for hp in hyperparameters]
    model_name = "-".join(hyperparameters)
    model_dir = "pretrained_models_TAHER" +"/"+ model_name
    if not os.path.exists(model_dir):
            os.mkdir(model_dir)
    model_path = model_dir + "/" + model_name

    with tf.Session() as sess:
        #model = ImageDecoder(config.IMG_SIZE, config.DIM_EMBEDDING, config.DIM_EMBEDDING, config.BATCH_SIZE, config.MAX_CAP_LEN +2, config.NUM_TOKENS, bias_init_vector=None)
        model = ImageDecoder()
        loss, summary, pdm, images, captions, masks = model.buildModel(filterSize_1,
                                                                        numFilters_1,
                                                                        filterSize_2,
                                                                        numFilters_2,
                                                                        filterSize_34,
                                                                        numFilters_34,
                                                                        filterSize_5,
                                                                        numFilters_5,
                                                                        strides,
                                                                        k)
        #loss, images, captions, masks = model.buildModel(filterSize,numFilters,strides,k)

        saver = tf.train.Saver(max_to_keep=5)

        train_op = tf.train.AdamOptimizer(eta).minimize(loss)
        
        # This is where the number of epochs for the LSTM are controlled
        sess.run(tf.global_variables_initializer())
        
        if config.USE_PRETRAINED_MODEL == True:
            print("Restoring pretrained model")
            saver.restore(sess, model_path)

        summ_writer = tf.summary.FileWriter(config.SUMMARY_DIRECTORY,
                                                sess.graph)
        start = time()

        prior_loss = 0
        for epoch in range(config.NUM_LSTM_EPOCHS):
            image_data, caption_data, _ = getImageBatchFromPickle("train_data-1.pkl", "train2014_normalized")
            
            mask_matrix = [[is_nonzero(idx) for idx in idx_cap] for idx_cap in caption_data]

            feed_dict = {images: image_data,
                         captions: caption_data,
                         masks: mask_matrix}
                         #m.initial_state = initial_state}

            _, results, loss_result, pred_caps = sess.run([train_op, summary, loss, pdm], feed_dict = feed_dict)
            #_, loss_result = sess.run([train_op, loss], feed_dict = feed_dict)
            # each result is a result per image
            
            summ_writer.add_summary(results, epoch)
            if not epoch % 10:
                saver.save(sess, model_path)#, global_step=epoch)#, write_meta_graph=False)

            print(model_name)
            print("Num epochs", epoch,"   time", round(time() - start))
            print("Loss", loss_result, "   Prior loss", prior_loss,"   difference", prior_loss - loss_result)
            print()
            prior_loss = loss_result
    
        saver.save(sess, model_path)
        summ_writer.close()

        #idx_to_word_translate(pred_caps, image_data)

    return {'final_loss':loss_result,'model_filename':model_name}



def test(filterSize_1,
        numFilters_1,
        filterSize_2,
        numFilters_2,
        filterSize_34,
        numFilters_34,
        filterSize_5,
        numFilters_5,
        strides,
        k,
        eta):

    tf.reset_default_graph()

    hyperparameters = [filterSize_1,numFilters_1,filterSize_2,numFilters_2,filterSize_34,numFilters_34,
        filterSize_5,numFilters_5,strides,k,eta]
    hyperparameters = [str(hp) for hp in hyperparameters]
    model_name = "-".join(hyperparameters)
    #model_name = "model"
    model_path = "pretrained_models_TAHER" +"/"+ model_name 
    
    #print_tensors_in_checkpoint_file(file_name=tf.train.latest_checkpoint(model_path + "/"), tensor_name='', all_tensors = '',all_tensor_names = True)
    
    with tf.Session() as sess:
        model = ImageDecoder()
        pdm, images = model.test(filterSize_1,numFilters_1, filterSize_2,numFilters_2,
            filterSize_34,numFilters_34,filterSize_5,numFilters_5,strides,k)

        
        print(model_path + ".meta")
        #saver = tf.train.import_meta_graph(model_path + "/" + model_name + ".meta")
        saver = tf.train.Saver()
        #saver.restore(sess,model_path)
        saver.restore(sess,tf.train.latest_checkpoint(model_path + "/"))
        #print("var" % embed_word_W.eval())

        # Will need to change for validation
        image_data, caption_data, image_files = getImageBatchFromPickle("train_data-1.pkl", "train2014_normalized")
        
        
        # Returns DataFrame with the filenames, english captions, and indexed captions of the image files for the loaded data
        BLEU_captions = image_captions("train_data-1.pkl", image_files)

        feed_dict = {images: image_data}
        pred_caps = sess.run([pdm], feed_dict = feed_dict)
        
        
        captions=idx_to_word_translate(pred_caps[0], image_data)
        imageCapDic = {image_files[i]:' '.join(list(captions[i])) for i in range(len(captions))}
        matchedCaps = [(list(BLEU_captions.loc[BLEU_captions.file_name==k]['caption'].apply(lambda x: ' '.join(x))),imageCapDic[k]) for k in imageCapDic]
        reference = [x[0] for x in matchedCaps]
        candidate = [x[1] for x in matchedCaps]
        
        avgBLEU = meanBLEUScore(candidate,reference)
        
        summary={'filterSize_1':hyperparameters[0],
        'numFilters_1':hyperparameters[1],
        'filterSize_2':hyperparameters[2],
        'numFilters_2':hyperparameters[3],
        'filterSize_34':hyperparameters[4],
        'numFilters_34':hyperparameters[5],
        'filterSize_5':hyperparameters[6],
        'numFilters_5':hyperparameters[7],
        'strides':hyperparameters[8],
        'k':hyperparameters[9],
        'eta':hyperparameters[10], 'BLEU_Score':avgBLEU}
        return summary

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
