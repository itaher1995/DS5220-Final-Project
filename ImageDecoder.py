# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 16:44:59 2018

@author: ibiyt
"""

import tensorflow as tf
import config
import math
import numpy as np
#from ImageEncoder import ImageEncoder

class ImageDecoder():
    '''
    Implementation of a caption generator, which is a neural network where our
    encoder is a convolutional neural network with chunked layers of the 
    convolution step, max pooling and normalization step followed by a
    recurrent neural network with LSTM.
    
    The functionality is as such. We will call this model, which will in turn
    build an LSTM neural network, which will then call our Image Encoder.
    
    Model -> LSTM -> CNN
    
    INPUT: Captions and Images
    OUTPUT: Captions
    '''
    
    def __init__(self):
        with tf.device("/cpu:0"):
            self.hidden_state = self.init_weight(config.BATCH_SIZE, config.NUM_LSTM_UNITS, name = "global_hidden_state")
            self.embedding_matrix = tf.Variable(tf.random_uniform([config.NUM_TOKENS, config.DIM_EMBEDDING], -1.0, 1.0), name='embedding_weights')
    
    def init_weight(self, dim_in, dim_out, name=None, stddev=1.0):
        return tf.Variable(tf.truncated_normal([dim_in, dim_out], name=name))
    
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
    
    def buildModel(self,filterSize,
                          numFilters,
                          strides,
                          k):
        '''
        Builds the LSTM and CNN and links them together.
        '''
        # Tensor to return and demonstrate program works
        #tester = tf.constant("it works")
        
        # Note that these placeholders take in an entire batch of inputs, i.e. 80 images and captions
        images = tf.placeholder(dtype = tf.float32, shape = [None, config.IMG_HEIGHT, config.IMG_WIDTH, 4], name = "image_input")
        captions = tf.placeholder(dtype = tf.int32, shape = [config.BATCH_SIZE, config.MAX_CAP_LEN + 2], name = "input_captions")
        
        # To include later if we want to help training
        # mask = tf.placeholder(dtype = tf.int32, shape = [config.BATCH_SIZE, config.MAX_CAP_LEN])
        
        # Build CNN
        with tf.name_scope("Image_Encoder"):
            chunk, weights = self.createCNNChunk(images,config.NUM_CHANNELS,
                                                 filterSize, numFilters,
                                                 strides, k)
            
            flattenLayer, numFeatures = self.flatten(chunk)
            cnnOutput = self.fullyConnectedComponent(flattenLayer, numFeatures,
                                                             config.NUM_CNN_OUTPUTS)
        
        #Build RNN
        #with tf.name_scope("Image_Decoder"):
        with tf.variable_scope(tf.get_variable_scope()) as scope:

            with tf.variable_scope("LSTM"):
            
                # Initialize lstm cell
                lstm = tf.contrib.rnn.BasicLSTMCell(config.NUM_LSTM_UNITS)

                # BATCH_SIZE x _
                prior_word = tf.zeros([config.BATCH_SIZE], tf.int32)
                print("Prior word:", prior_word.shape)

                # Initialize input, BATCH_SIZE x NUM_LSTM_UNITS
                current_input = cnnOutput
                print("Current_input", current_input.shape)

                # The hidden state corresponds the the cnn inputs, both are BATCH_SIZE x NUM_LSTM_UNITS vectors
                initial_hidden_state = self.hidden_state
                initial_current_state = tf.zeros([config.BATCH_SIZE, config.NUM_LSTM_UNITS])

                # Needed to start model, tuple of vectors
                prior_state = initial_hidden_state, initial_current_state
                #prior_state = m.initial_state.eval()
            
            predicted_caption = []
            loss = 0

            with tf.variable_scope("loss_loop"):
                # For training, need to loop through all the of possible positions
                for i in range(config.MAX_CAP_LEN + 2):
                    
                    # Create onehot vector, or vector of entire dictionary where the word in sentance is labeled 1
                    labels = captions[:,i]
                    # BATCH_SIZE x NUM_TOKENS matrix
                    onehot_labels = tf.one_hot(labels, config.NUM_TOKENS,
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
                        output, state = lstm(current_input, prior_state)

                        _, current_state = state
                    
                    with tf.variable_scope("lstm_output"):
                        # BATCH_SIZE x NUM_LSTM_UNITS
                        m_t = tf.multiply(output, current_state)

                        #logits = 

                        # BATCH_SIZE x NUM_LSTM_UNITS
                        p_t = tf.nn.softmax(m_t, name = "word_probabilities")

                    # Calculates the loss for the training, performs it in a slightly different manner than paper
                    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = p_t, labels = captions[:,i])
                    current_loss = tf.reduce_mean(cross_entropy)
                    loss = loss + current_loss
                    #print("Loop", i, "Loss", loss)

                    predicted_word = tf.argmax(p_t, 1)

                    predicted_caption.append(predicted_word)

                    prior_word = captions[:, i-1]
                    prior_state = state
                    #prior_output = output
                    
                    # Needs to come after everything in the loop and evaluation process so that the variables can be run with the next input
                    tf.get_variable_scope().reuse_variables()
    
        hidden_state, _ = prior_state
        cross_entropies = tf.stack(cross_entropy)
        # Got rid of the masks being divided by
        cross_entropy_loss = tf.reduce_mean(cross_entropies)

        self.loss = loss
        self.cross_entropy_loss = cross_entropy_loss
        self.hidden_state = hidden_state
        print(2)
        summary = self.build_summary()
        return loss,summary, predicted_caption, images, captions
        
    def test(self):
        return "Incomplete"



    def build_summary(self):
        """ Build the summary (for TensorBoard visualization). """
        """
        with tf.name_scope("variables"):
            for var in tf.trainable_variables():
                with tf.name_scope(var.name[:var.name.find(":")]):
                    self.variable_summary(var)
        """
        with tf.name_scope("metrics"):
            tf.summary.scalar("cross_entropy_loss", self.cross_entropy_loss)
            tf.summary.scalar("reg_loss", self.loss)

        self.summary = tf.summary.merge_all()
        return tf.summary.merge_all()

    def variable_summary(self, var):
        """ Build the summary for a variable. """
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)
        
    