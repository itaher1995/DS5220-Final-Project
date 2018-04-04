# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 01:51:44 2018

@author: ibiyt
"""

import pandas as pd
import numpy as np
from time import time
from skimage import data, transform
import tensorflow as tf
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from string import punctuation
from collections import Counter
import config
import pickle
import os
from nltk.corpus import stopwords

#import nltk
#nltk.download('punkt')
#nltk.download('stopwords')

def cleanString(s):
    '''
    Removes all punctuation and makes all words lowercase.
    '''
    table=s.maketrans({key: None for key in punctuation})
    s=s.translate(table).lower()
    return s



def buildVocab(filePrefix):
    '''
    Gets top n number of tokens in the corpus of the training set. Takes in the
    prefix to a file with the suffix _annotations.csv. Creates a dataframe and
    uses PANDAS operations to get the n most frequent tokens in the corpus. N
    is based the config file
    '''
    filename = f'{filePrefix}_annotations.csv'
    captionDataFrame = pd.read_csv(filename)
    
    #get unique tokens
    captionDataFrame['caption']=captionDataFrame['caption'].apply(cleanString)
    captionAgg = captionDataFrame.caption.str.cat(sep=' ')
    frequentTokens = pd.Series(word_tokenize(captionAgg)).value_counts().iloc[:config.NUM_TOKENS]
    #frequentTokens = frequentTokens[~(frequentTokens.index.isin(stopwords.words('english')))]
    
    return frequentTokens

def tokenMap(tokens):
    '''
    Maps a token to a number. In this token map is also a __START__ and __STOP__
    token. The token map keys are indices and the token map values are tokens.
    '''
    tokenList = tokens.index
    tokenMap = {i+1:tokenList[i] for i in range(len(tokens))}
    tokenMap[config.START_TOKEN_IDX]="__START__"
    tokenMap[config.STOP_TOKEN_IDX]="__STOP__"
    
    return tokenMap

def intersectionEquals(arr, arr1):
    '''
    Check if all words in the caption are in our corpus.
    '''
    if set(arr).intersection(arr1)==set(arr):
        return True
    return False

def subsetCaptions(filePrefix,tokenMap):
    '''
    Takes a set of captions and gets a subset such that a word in the caption
    exists in our token corpus and the length of the caption is less than our
    max length.
    '''
    captionDataFrame=pd.read_csv(f'{filePrefix}_annotations.csv')
    captionDataFrame['caption']=captionDataFrame['caption'].apply(cleanString).str.split(' ')
    captionIndices = [i for i in range(len(captionDataFrame)) if intersectionEquals(captionDataFrame['caption'].iloc[i],tokenMap.values()) and len(captionDataFrame.iloc[i])<=config.MAX_CAP_LEN]
    return captionDataFrame.loc[captionIndices]

def subsetImages(images,captions):
    '''
    Returns subset of images that have ids in the subset of captions.
    '''
    return images.loc[images.image_id.isin(captions['image_id'])]

def mergeImageCaptionData(captions,filePrefix,tokenMap):
    '''
    Merges the image data and the caption data together in one dataframe.
    The merge is an inner merge but it creates a one to many join.
    '''
    imageData = pd.read_csv(f"{filePrefix}_image_data.csv").rename(columns = {'id':'image_id'})
    captions = captions.rename(columns = {'id':'caption_id'})
    imgSubset = subsetImages(imageData,captions)
    wordToIndex = {tokenMap[k]:k for k in tokenMap}
    captions['mapped_captions']=[[wordToIndex[token] for token in caption] for caption in captions['caption']]
    imgCaptionMerge=pd.merge(imgSubset,captions,how='inner',on='image_id')
    outfile=f"{filePrefix}_data-1.pkl"
    imgCaptionMerge.to_pickle(outfile)
    return imgCaptionMerge

# Resizes images based on the image data we've filtered --IBI DID NOT CHANGE THiS FUNCTION.
def resize_images(filetype):
    
    start = time()
    
    datapath = filetype + "_data.pkl"

    data = pd.read_pickle(datapath)
    
    data_directory = filetype + "2014"
    
    output_directory = data_directory + "_normalized"
    
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Only prints first 5 here for testing purposes
    image_filenames = list(data['file_name'][0:5])
    
    print(image_filenames)
    
    for f in image_filenames:
        
        filepath = os.path.join(data_directory, f)
        outpath = os.path.join(output_directory, f)
        
        #image = data.imread(filepath)
        
        #print(type(image))
        
        plt.imshow(image)
        
        result = transform.resize(image, (config.IMG_HEIGHT, config.IMG_WIDTH))
        
        #print(result.ndim)
        
        #plt.imshow(result)
        
        plt.imsave(outpath, result)
        
        #plt.imshow(data.imread(outpath))
    
    print("Standardization DONE IN", round(time() - start), "SEC")

def main():
    start = time()
    #do this for train
    tokens = buildVocab('train')
    tMap= tokenMap(tokens)
    trainSubset = subsetCaptions('train',tMap)
    mergedImageCaptionDataFrame = mergeImageCaptionData(trainSubset,'train',tMap)
    return f"This took {time()-start} seconds to finish" #188 seconds on my computer
    #didn't call resize images in here. Wasn't sure what you wanted to do with it.
    
    


