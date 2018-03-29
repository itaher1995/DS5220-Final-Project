#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd

import numpy as np

from time import time

from skimage import data, transform

import tensorflow as tf

import matplotlib.pyplot as plt

import config

import pickle

import fnmatch

import os
       



def image_data_subset(filetype, perc = .1):
    
    filename = "/Users/forresthooton/Documents/Masters Classes/Supervised Machine Learning/Class Project/" + filetype + "_image_data.csv"
    
    data = pd.read_csv(filename)
    
    # Sets the number of images we want to train based off a percent of the total number
    
    numImages = round(data.shape[0] * perc)
    
    # Gets random DataFrame subset where the number of rows is numImages
    
    data = data.loc[np.random.choice(data.index,numImages,False)].reset_index(drop=True).copy()
    
    return data
    



def attach_annotations(imgData, annData):
    
    start = time()
    
    condencedData = pd.DataFrame()
    
    # Cycles through each image in the image data table
    
    
    
    for img in imgData['image_id']:
        
        # Returns the subset of annotations with a particular image_id
        
        annSub = annData[annData['image_id'] == img]
        
        annDict = {}
        
        # Goes through each row of annotation subset for a specific image id and makes a single dictionary
        # where the key is the caption_id and the value is the caption
        
        
        
        for idx, row in annSub.iterrows():
            
            annDict[row['caption_id']] = row['caption']
        
        # Makes a pd.Series to updata a new df with the image_id and associated annotations
        
        entry = pd.Series()
        
        entry['image_id'] = img
        
        entry['annotations'] = annDict
        
        # Adds column into df
        
        condencedData = pd.concat([condencedData, entry], 1)
    
    
    
    condencedData = condencedData.T
    
    # Merges the new df of image_id's and annotations into image data table
    
    imgData = imgData.merge(condencedData, on = 'image_id', how = 'left')
    
    print("DONE IN", round(time() - start), "SEC")
    
    return imgData




# Attaches annotations as dict to image data
    
def reorganize_data_tables():
    
    # Returns DataFrame of randomly chosen rows
    
    trainSubset = image_data_subset("train")
    
    valSubset = image_data_subset("val")
    
    
    
    # Rename id for clarity throughout the project
    
    trainSubset = trainSubset.rename(columns = {'id':'image_id'})
    
    valSubset = valSubset.rename(columns = {'id':'image_id'})
    
    trainAnns = pd.read_csv("/Users/forresthooton/Documents/Masters Classes/Supervised Machine Learning/Class Project/train_annotations.csv")
    
    valAnns = pd.read_csv("/Users/forresthooton/Documents/Masters Classes/Supervised Machine Learning/Class Project/val_annotations.csv")
    
    
    
    # Rename id for clarity throughout the project
    
    trainAnns = trainAnns.rename(columns = {'id':'caption_id'})
    
    valAnns = valAnns.rename(columns = {'id':'caption_id'})
    
    
    
    # Assign the subset of annotations in which the image_id is in the image data subset to a new var
    
    trainAnnSubset = trainAnns[trainAnns['image_id'].isin(trainSubset['image_id'])]
    
    valAnnSubset = valAnns[valAnns['image_id'].isin(valSubset['image_id'])]
    
    
    
    # attach_annotations() returns a DataFrame that is the image data subset with an annotations column attached
    # The values in the annotations column are dictionaries, where the key is the 'caption_id'
    # and the value is the 'caption'. This means that each row of the new table contains all the
    # annotations for an image
    
    trainSubset = attach_annotations(trainSubset.copy(), trainAnnSubset.copy())
    
    valSubset = attach_annotations(valSubset.copy(), valAnnSubset.copy())
    
    trainSubset.to_pickle("train_data.pkl")
    
    valSubset.to_pickle("val_data.pkl")




def resize_images():
    
    start = time()
    
    
    
    trainData = pd.read_pickle("train_data.pkl")
    
    data_directory = "train2014"
    
    output_directory = "train2014_normalized"
    
    image_filenames = list(trainData['file_name'][0:5])
    
    print(image_filenames)
    
    
    
    for f in image_filenames:
        
        filepath = os.path.join(data_directory, f)
        
        outpath = os.path.join(output_directory, f)
        
        image = data.imread(filepath)
        
        print(type(image))
        
        plt.imshow(image)
        
        result = transform.resize(image, (config.IMG_HEIGHT, config.IMG_WIDTH))
        
        plt.imshow(result)
        
        plt.imsave(outpath, result)
        
        plt.imshow(data.imread(outpath))
    
    
    
    print("Standardization DONE IN", round(time() - start), "SEC")




def tokenize_captions():
    
    data = pd.read_pickle("train_data.pkl")
    
    captions = []
    
    
    
    # Append all captions to one list
    
    for idx, row in data.iterrows():
        
        for key, value in row['annotations'].items():
            
            captions.append(value)
    
    
    
    joined_cap = " ".join(captions) # Join all captions into one string
    
    tokenized_cap = joined_cap.lower().rstrip('.').split()  # Split words individually
    
    unique_tokens = list(set(tokenized_cap))    # Removes duplicates leaving tokens
    
    
    
    with open('tokens.pkl', 'wb') as f:
        pickle.dump(unique_tokens, f)
    
    
    
    
def main():
    
    #reorganize_data_tables()
    
    #resize_images()
    
    tokenize_captions()
    
    
    
    
if __name__ == "__main__":
    main()



# Scrap code for comparing length of files
'''
    annotations_train = pd.read_csv("/Users/forresthooton/Documents/Masters Classes/Supervised Machine Learning/Class Project/train_annotations.csv")
    annotations_val = pd.read_csv("/Users/forresthooton/Documents/Masters Classes/Supervised Machine Learning/Class Project/val_annotations.csv")
    
    z = len(annotations_train)
    p = len(annotations_val)
    print(z+p)
    print(annotations_train.head())

    
    dirpath = "/Users/forresthooton/Documents/Masters Classes/Supervised Machine Learning/Class Project/train2014/"
    x = len(fnmatch.filter(os.listdir(dirpath), '*.jpg'))
    print(x)
    
    dirpath = "/Users/forresthooton/Documents/Masters Classes/Supervised Machine Learning/Class Project/val2014/"
    y = len(fnmatch.filter(os.listdir(dirpath), '*.jpg'))
    print(y)
    
    dirpath = "/Users/forresthooton/Documents/Masters Classes/Supervised Machine Learning/Class Project/test2014"
    w = len(fnmatch.filter(os.listdir(dirpath), '*.jpg'))
    print(w)
    
    print(x + y)
    '''
