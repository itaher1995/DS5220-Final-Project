# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 16:34:15 2018

@author: ibiyt
"""

import tensorflow as tf

class ImageEncoder():
    '''
    Class that represents our convolution neural network. The neural network
    will go through M "CNN chunks" which we define as a convolution->pooling->
    ReLU set of steps. 
    
    These steps will then be piped into a fully connected layer that is of the
    size of our corpus of words N.
    
    INPUT: Image Data
    OUTPUT: FULLY CONNECTED COMPONENTS OF SIZE NX1
    '''
    
    def __init__(self):
        return None
    
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
        return tf.Variable(tf.constant(0.05, shape=[length]))
    
    def createCNNChunk(self,inputLayer,
                          numInputChannels,
                          filterSize,
                          numFilters,
                          strides,
                          k):
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
        
        with tf.name_scope("CONV.WeightsAndBiases"):
            weights = self.__INITIALIZEWEIGHTS__(shape) #creates weights given the shape
            biases = self.__INITIALIZEBIAS__(length=numFilters) #creates new biases one for each filter
        
        #convolution layer
        #This convolution step has its input being a previous layer, inputLayer,
        #the filter is the weights determined with the help of the shape.
        #It's stride is moving one pixed across the x and y axis.
        #padding= 'SAME' essentially means that the layer with pad a layer
        #of 0s such that it will be equal in dimensions (though I think
        #we handle this already.)
        with tf.name_scope("CNN_Chunk"):
            convolution = tf.nn.conv2d(input=inputLayer,
                                 filter=weights,
                                 strides=[1, strides, strides, 1],
                                 padding='SAME',name="Convolution")
            #add biases to the result of the convolution
            convolution = tf.nn.bias_add(convolution, biases)
            
            convolutionWReLU = tf.nn.relu(convolution)
            
            #max pooling
            #We compute max pooling so that we can find the most "relevant" features
            #of our images.
            maxPooling = tf.nn.max_pool(value=convolutionWReLU,
                                   ksize=[1, k, k, 1],
                                   strides=[1, k, k, 1],
                                   padding='SAME',name="Max_Pooling")
            
            #normalization with ReLU, this is to remove any negative values
            normalized = tf.nn.relu(maxPooling,name="Chunk_Normalization")
            
            return normalized, weights
    
    def flatten(self,layer):
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
        layer_flat = tf.reshape(layer, [-1, num_features],name="Flatten_Layer")
    
        # The shape of the flattened layer is now:
        # [num_images, img_height * img_width * num_channels]
    
        # Return both the flattened layer and the number of features.
        return layer_flat, num_features
    
    def fullyConnectedComponent(self,inputLayer,numInputs,numOutputs):
        '''
        Creates a fully connected layer. Theoretically, this will be the image
        encoder's last step before we begin the decoding process.
        '''
        
        #initialize new weights and biases
        with tf.name_scope("FULLYCONNECTED.WeightsAndBiases"):
            weights = self.__INITIALIZEWEIGHTS__(shape=[numInputs, numOutputs])
            biases = self.__INITIALIZEBIAS__(length=numOutputs)
        with tf.name_scope("Fully_Connected_Chunk"):
            fullyConnected = tf.matmul(inputLayer,weights, name="Fully_Connected_Layer") + biases
            
            #normalize fully connected layer
            normFullConnected = tf.nn.relu(fullyConnected,name="Normalization_Fully_Connected_Layer")
        
        return normFullConnected