# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 18:36:03 2018

@author: ibiyt
"""

import pandas as pd
import numpy as np

def getImageBatchFromPickle(batchSize,pkl):
    '''
    Gets the image batch and the corresponding captions. Takes a pickle file
    and then randomly select batchSize indices. Then we locate the those
    indices in the dataframe and output a dataframe with just the file_name
    and the idx_captions.
    '''
    df = pd.read_pickle(pkl)
    imageBatchIndex = np.random.choice(df.index,size=batchSize)
    imageBatch=df.iloc[imageBatchIndex][['file_name','idx_caption_matrix']]
    
    
    return imageBatch

    



    