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
import pandas as pd
import pickle
import os
from PIL import Image

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
        self.imgSize = config.IMG_SIZE #size of the image (28)
        self.flattenDim = config.IMG_SIZE * config.IMG_SIZE #area of the image 28*28
        self.imageShape = (config.IMG_SIZE, config.IMG_SIZE) #dimensions of the image
        self.numChannels = config.NUM_CHANNELS #number of input channels
        self.eta = config.LEARNING_RATE #eta is the learning rate
        self.maxCapLength = config.MAX_CAP_LEN #max length of any sentence for padding
        self.numLSTMUnits = config.NUM_LSTM_UNITS #number of hidden layers
        self.batchSize = config.BATCH_SIZE #size of batches to be used for training
        self.numTokens = config.NUM_TOKENS #number of tokens in our corpus
        self.dimEmbeddling = config.DIM_EMBEDDING #dimensions of the embedding matrix for P(S_t|S_(t-1),...S_1)
        self.numLSTMEpochs = config.NUM_LSTM_EPOCHS #the number epochs we will be training over
        self.hidden_state = tf.zeros([config.BATCH_SIZE, config.NUM_LSTM_UNITS], name = "global_hidden_state")

        with tf.device("/cpu:0"):
            self.embedding_matrix = tf.Variable(tf.random_uniform([config.NUM_TOKENS, config.DIM_EMBEDDING], -1.0, 1.0), name='embedding_weights')


    
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
    
    def __STRINGPADDER__(self,captionMatrix,maxCapLength):
        '''
        Takes a a matrix of captions and pads them s.t. all of our captions have the same length.
        '''
        padded_matrix = []
        for i in range(len(captionMatrix)):
            caption_array = captionMatrix[i]
            
            # Want to pad up to max length
            num_pad = maxCapLength - len(caption_array)
            
            # Pads strings with zero's. Accounted for this when we mapped the word idx's starting at 1
            padded_caption = np.pad(caption_array, (0,num_pad), 'constant', constant_values = (0,0))
            padded_matrix.append(padded_caption)

        return padded_matrix
    
    def __CREATECNNCHUNK__(self,inputLayer,
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
        weights = self.__INITIALIZEWEIGHTS__(shape) #creates weights given the shape
        biases = self.__INITIALIZEBIAS__(length=numFilters) #creates new biases one for each filter
        
        #convolution layer
        #This convolution step has its input being a previous layer, inputLayer,
        #the filter is the weights determined with the help of the shape.
        #It's stride is moving one pixed across the x and y axis.
        #padding= 'SAME' essentially means that the layer with pad a layer
        #of 0s such that it will be equal in dimensions (though I think
        #we handle this already.)
        convolution = tf.nn.conv2d(input=inputLayer,
                             filter=weights,
                             strides=[1, strides, strides, 1],
                             padding='SAME')
        #add biases to the result of the convolution
        convolution = tf.nn.bias_add(convolution, biases)
        
        convolutionWReLU = tf.nn.relu(convolution)
        
        #max pooling
        #We compute max pooling so that we can find the most "relevant" features
        #of our images.
        maxPooling = tf.nn.max_pool(value=convolutionWReLU,
                               ksize=[1, k, k, 1],
                               strides=[1, k, k, 1],
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
    
    def __IMAGEENCODER__(self,X_train,numInputChannels,filterSize,numFilters,
                         fcSize,strides,k):
        '''
        Function to build the Image Encoding (CNN) step of the image-caption
        generation. It essentially will use an input set of images and develop
        a functional path between the input layer and the final fully connected
        layer. This fully connected layer will then be the input to our image
        decoder (RNN).
        '''
        
        #convolution layer 
        layer1, weights = self.__CREATECNNCHUNK__(X_train,
                                                  numInputChannels,
                                                  filterSize,
                                                  numFilters,
                                                  strides,
                                                  k)
        
        #flatten layer
        layer2, numFeatures = self.__FLATTEN__(layer1)
        
        #fully connected layer
        outputEncoded = self.__FULLYCONNECTED__(layer2,numFeatures,fcSize)
        
        return outputEncoded
    
    def __IMAGEDECODER__(self,inputLayer, Y_train, batchSize,numLSTMUnits,maxCapLen,numTokens):
        with tf.name_scope("lstm"):
            with tf.variable_scope(tf.get_variable_scope()) as scope:

                with tf.variable_scope("initialize"):
                
                    # Initialize lstm cell
                    lstm = tf.contrib.rnn.BasicLSTMCell(numLSTMUnits)

                    # BATCH_SIZE x _
                    prior_word = tf.zeros([batchSize], tf.int32)
                    print("Prior word:", prior_word.shape)
                    
                    # Initialize input, BATCH_SIZE x NUM_LSTM_UNITS
                    current_input = inputLayer
                    print("Current_input", current_input.shape)
                    
                    # The hidden state corresponds the the cnn inputs, both are BATCH_SIZE x NUM_LSTM_UNITS vectors
                    initial_hidden_state = self.hidden_state
                    initial_current_state = tf.zeros([batchSize,numLSTMUnits])

                    # Needed to start model, tuple of vectors
                    prior_state = initial_hidden_state, initial_current_state
                    #prior_state = m.initial_state.eval()
                
                predicted_caption = []
                loss = 0
                
                #with tf.variable_scope(tf.get_variable_scope()) as scope:
                # For training, need to loop through all the of possible positions
                for i in range(maxCapLen):
                    
                    # Create onehot vector, or vector of entire dictionary where the word in sentance is labeled 1
                    labels = Y_train[:,i]
                    # BATCH_SIZE x NUM_TOKENS matrix
                    onehot_labels = tf.one_hot(labels, numTokens,
                                               on_value = 1, off_value = 0,
                                               name = "onehot_labels")
                    #print("onehot:", onehot_labels.shape)
                    
                    if i != 0:
                        with tf.variable_scope("word_embedding"):
                            # Can't be run on a gpu for some reason
                            with tf.device("/cpu:0"):
                                # Accounts for the one_hot vector
                                # BATCH_SIZE x NUM_TOKENS matrix
                                prior_word_probs = tf.nn.embedding_lookup(self.embedding_matrix, prior_word)
                            current_input = tf.multiply(prior_word_probs, tf.cast(onehot_labels, tf.float32))
                    
                    with tf.variable_scope("lstm_function"):
                        
                        # This line executes the actual gates of lstm to update values, output is BATCH_SIZE x NUM_LSTM_UNITS
                        print(current_input==inputLayer)
                        output, state = lstm(current_input, prior_state)
    
                        _, current_state = state
                    
                    with tf.variable_scope("lstm_output"):
                        # BATCH_SIZE x NUM_LSTM_UNITS
                        m_t = tf.multiply(output, current_state)
    
                        #logits = 
    
                        # BATCH_SIZE x NUM_LSTM_UNITS
                        p_t = tf.nn.softmax(m_t, name = "word_probabilities")
    
                    # Calculates the loss for the training, performs it in a slightly different manner than paper
                    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = p_t, labels = Y_train[:,i])
                    current_loss = tf.reduce_sum(cross_entropy)
                    loss = loss + current_loss
                    #print("Loop", i, "Loss", loss)
    
                    predicted_word = tf.argmax(p_t, 1)
    
                    predicted_caption.append(predicted_word)
    
                    prior_word = Y_train[:, i-1]
                    prior_state = state
                    prior_output = output
                    
                    # Needs to come after everything in the loop and evaluation process so that the variables can be run with the next input
                    tf.get_variable_scope().reuse_variables()
        
        hidden_state, _ = prior_state
        self.hidden_state = hidden_state
        print(2)
        return loss, inputLayer, Y_train

    
    def train(self,filterSize,numFilters,fcSize,strides,k,batchSize,numLSTMUnits,maxCapLen,numTokens):
        '''
        Method to train the image-caption generator
        
        Until Convergence:
            1. Call __IMAGEENCODER__ which generates a fully connected component
            2. That component is fed into the __IMAGEDECODER__ which will output phrases
            3. Calculate J(Theta) and update weights and biases via truncated backpropogation
        '''
        #placeholder for image
        tf.reset_default_graph()
        x = tf.placeholder(tf.float32, shape=[None, self.flattenDim], name='x')
        x = tf.reshape(x, shape=[-1, config.IMG_SIZE, config.IMG_SIZE, 4])
        #placeholder for caption will go here
        y = tf.placeholder(dtype = tf.int32, shape = [config.BATCH_SIZE, config.MAX_CAP_LEN], name = "y")
        
        
        
        cnnOutput = self.__IMAGEENCODER__(x,self.numChannels,filterSize,numFilters,fcSize,strides,k)
        loss, xDecode, yDecode = self.__IMAGEDECODER__(cnnOutput, y, batchSize,numLSTMUnits,maxCapLen,numTokens) 
        
        print(1)
        train_op = tf.train.AdamOptimizer(config.LEARNING_RATE).minimize(loss)
        print(3)
        # This is where the number of epochs for the LSTM are controlled
        with tf.Session() as sess:
            for epoch in range(config.NUM_LSTM_EPOCHS):
                
                sess.run(tf.global_variables_initializer())
                
                # Need to pad captions
                y = self.__STRINGPADDER__(y)
                
                feed_dict = {xDecode: x,
                             yDecode: y}
                             #m.initial_state = initial_state}
                print(4)
                _, loss_result = sess.run([train_op, loss], feed_dict = feed_dict)
                
                # each result is a result per image
                print(loss_result)
                #saver.save(sess, os.path.join(confg.MODEL_PATH, 'model'), global_step=epoch)
            
        return cnnOutput, x, y, loss

def main():
        # reads in necessary image data
    img_data = pd.read_pickle("train_data.pkl")
    
    # Just gets a couple images and captions for testing right now
    image_filenames = list(img_data['file_name'][0:config.BATCH_SIZE])
    print(image_filenames)
    
    data_directory = "train2014_normalized"
    
    image_data = []
    caption_data = []
    
    for f in image_filenames:
        
        filepath = os.path.join(data_directory, f)
        
        image_data.append(data.imread(filepath))
        
        cap_row = img_data[img_data['file_name'] == f].copy()
        
        # Note that annotations is a pd.Series()
        idx_captions = cap_row['idx_caption_matrix'].item()        
        
        # really only want one annotation per image for testing
        for sentance in idx_captions:
            #caption_data.append(sentance)
            caption_data.append(list(range(len(sentance))))
            break
   
    model = ImageCaptionGenerator()
    cnnOutput, x, y, loss = model.train(config.FILTER_SIZE,config.NUM_FILTERS,config.FULLY_CON_LAYER_SIZE,
                               config.STRIDES,config.POOL_SIZE, config.BATCH_SIZE, config.NUM_LSTM_UNITS,
                               config.MAX_CAP_LEN, config.NUM_TOKENS)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        feed_dict = {x: image_data, y: caption_data}
        result = sess.run(cnnOutput, feed_dict = feed_dict)
        print(result)
    

 
        

if __name__ == "__main__":
    main()
    
    


        