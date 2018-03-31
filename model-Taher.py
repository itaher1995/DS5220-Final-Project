# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 13:11:31 2018

@author: ibiyt
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import config
from skimage import data
from time import time

class ImageCaptionGenerator():
    '''
    Implementation of NIC Algorithm from Show and Tell: A Neural Image Caption
    Generator. The caption generator is a neural network where our encoder is
    a convolutional neural network with chunked layers of the convolution step,
    max pooling and normalization step followed by a recurrent neural network
    with LSTM.
    
    The CNN was built with the help of the following tutorial: 
        https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/
        master/02_Convolutional_Neural_Network.ipynb
    
    The RNN was built with the help f the following tutorial:
        
    The link to the inspiration for this implementation:
        https://www.cv-foundation.org/openaccess/content_cvpr_2015/app/
        2A_101.pdf
    
    This algorithm will implement truncated back propogation.
    '''
    def __init__(self):
        self.imgSize = config.IMG_SIZE
        self.flattenDim = config.IMG_SIZE * config.IMG_SIZE
        self.imageShape = (config.IMG_SIZE, config.IMG_SIZE)
        self.numChannels = 3
    
    def __INITIALIZEWEIGHTS__(self,shape):
        '''
        Helper function to be called in create_conv_layer. This initializes
        random weights for the beginning of our image-caption generation.
        
        INPUT: shape (a 4D tensor that is made up of the filter dimensions,
        number of input channels and number of filters)
        
        OUTPUT: Random Weights
        '''
        return tf.Variable(tf.truncated_normal(shape, stddev=0.05))
    
    def __INITIALIZEBIAS__(self,length):
        '''
        Helper function to be called in create_conv_layer. This initializes
        random biases for the beginning of image-caption generation.
        
        INPUT: length (a tensor that is the length of the number of filters)
        '''
        tf.Variable(tf.constant(0.05, shape=[length]))
    
    def __CREATECNNCHUNK__(self,inputLayer,
                          numInputChannels,
                          filterSize,
                          numFilters):
        '''
        Takes in inputs regarding the previous layer, the number of channels,
        the size and number of the filters for convolution and creates a chunk
        that goes as so:
            
            CONVOLUTION -> MAX_POOLING -> NORMALIZATION W/ReLU
        
        This is done in order to reduce the dimensions of the data, capturing 
        only the most relevant information for each image, which in turn will 
        help the accuracy of the Image Caption Decoder (the RNN w/ LSTM).
        '''
        #shape list structure determined by TensorFlow API
        shape = [filterSize,filterSize,numInputChannels,numFilters]
        weights = self.__INITIALIZEWEIGHTS__(shape) #creates weights given the shape
        biases = self.__INITIALIZEBIAS__(length=numFilters) #creates new biases one for each filter
        
        #convolution layer
        #This convolution step has its input being a previous layer, inputLayer,
        #the filter is the weights determined with the help of the shape.
        #It's stride is moving one pixed across the x and y axis.
        #padding= 'SAME' essentially means that the layer with pad a layer
        #of 0s such that it will be equal in dimensions (though I think
        #we handle this already.)
        convolution = tf.nn.conv2d(input=self.inputLayer,
                             filter=weights,
                             strides=[1, 1, 1, 1],
                             padding='SAME')
        
        #add biases to the result of the convolution
        convolution += biases
        
        #max pooling
        #We compute max pooling so that we can find the most "relevant" features
        #of our images.
        maxPooling = tf.nn.max_pool(value=convolution,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')
        
        #normalization with ReLU, this is to remove any negative values
        normalized = tf.nn.relu(maxPooling)
        
        return normalized, weights
    
    def __FLATTEN__(self,layer):
        '''
        Flatten layer from 4-D tensor to 2-D tensor that way we can create
        a fully connected layer at the end of our CNN.
        '''
        # Get the shape of the input layer.
        layer_shape = layer.get_shape()
    
        # The shape of the input layer is assumed to be:
        # layer_shape == [num_images, img_height, img_width, num_channels]
    
        # The number of features is: img_height * img_width * num_channels
        # We can use a function from TensorFlow to calculate this.
        num_features = layer_shape[1:4].num_elements()
        
        # Reshape the layer to [num_images, num_features].
        # Note that we just set the size of the second dimension
        # to num_features and the size of the first dimension to -1
        # which means the size in that dimension is calculated
        # so the total size of the tensor is unchanged from the reshaping.
        layer_flat = tf.reshape(layer, [-1, num_features])
    
        # The shape of the flattened layer is now:
        # [num_images, img_height * img_width * num_channels]
    
        # Return both the flattened layer and the number of features.
        return layer_flat, num_features
    
    def __FULLYCONNECTED__(self,inputLayer,numInputs,numOutputs):
        '''
        Creates a fully connected layer. Theoretically, this will be the image
        encoder's last step before we begin the decoding process.
        '''
        
        #initialize new weights and biases
        weights = self.__INITIALIZEWEIGHTS__(shape=[numInputs, numOutputs])
        biases = self.__INITIALIZEBIAS__(length=numOutputs)
        
        fullyConnected = tf.matmul(inputLayer,weights) + biases
        
        #normalize fully connected layer
        
        normFullConnected = tf.nn.relu(fullyConnected)
        
        return normFullConnected
    
    def __IMAGEENCODER__(self,X_train,):
        '''
        Function to build the Image Encoding (CNN) step of the image-caption
        generation. It essentially will use an input set of images and develop
        a functional path between the input layer and the final fully connected
        layer. This fully connected layer will then be the input to our image
        decoder (RNN).
        '''
        return "This section is not completed yet - IT."
    
    
        

    
        
    
    


        