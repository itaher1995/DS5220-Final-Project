# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 18:36:03 2018

@author: ibiyt
"""

import pandas as pd
import numpy as np
from skimage import data
import os
import config

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
    
    # Just gets a couple images and captions for testing right now
    image_filenames = list(imageBatch['file_name'])
    print(image_filenames)
    
    images = []
    captions = []
    
    for f in image_filenames:
        filepath = os.path.join(data_directory, f)
        images.append(data.imread(filepath))
        cap_row = imageBatch[imageBatch['file_name'] == f]
        captions.append(list(cap_row['mapped_captions'].item()))
    
    return images, captions

    



    