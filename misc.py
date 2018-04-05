import tensorflow as tf
import tensorflow.contrib.layers as layers

def conv2d(inputs,
           filters,
           kernel_size = (3, 3),
           strides = (1, 1),
           activation = tf.nn.relu,
           use_bias = True,
           name = None):
    """ 2D Convolution layer. """
    return tf.layers.conv2d(
        inputs = inputs,
        filters = filters,
        kernel_size = kernel_size,
        strides = strides,
        padding='same',
        activation = activation,
        use_bias = use_bias,
        name = name)

def max_pool2d(inputs,
               pool_size = (2, 2),
               strides = (2, 2),
               name = None):
    """ 2D Max Pooling layer. """
    return tf.layers.max_pooling2d(
        inputs = inputs,
        pool_size = pool_size,
        strides = strides,
        padding='same',
        name = name)

def dense(inputs,
          units,
          activation = tf.tanh,
          use_bias = True,
          name = None):
    
    return tf.layers.dense(
        inputs = inputs,
        units = units,
        activation = activation,
        use_bias = use_bias,
        name = name)


def vgg16(self,images):
        """ Build the VGG16 net. """
        conv1_1_feats = misc.conv2d(images, 64, name = 'conv1_1')
        conv1_2_feats = misc.conv2d(conv1_1_feats, 64, name = 'conv1_2')
        pool1_feats = misc.max_pool2d(conv1_2_feats, name = 'pool1')

        conv2_1_feats = misc.conv2d(pool1_feats, 128, name = 'conv2_1')
        conv2_2_feats = misc.conv2d(conv2_1_feats, 128, name = 'conv2_2')
        pool2_feats = misc.max_pool2d(conv2_2_feats, name = 'pool2')

        conv3_1_feats = misc.conv2d(pool2_feats, 256, name = 'conv3_1')
        conv3_2_feats = misc.conv2d(conv3_1_feats, 256, name = 'conv3_2')
        conv3_3_feats = misc.conv2d(conv3_2_feats, 256, name = 'conv3_3')
        pool3_feats = misc.max_pool2d(conv3_3_feats, name = 'pool3')

        conv4_1_feats = misc.conv2d(pool3_feats, 512, name = 'conv4_1')
        conv4_2_feats = misc.conv2d(conv4_1_feats, 512, name = 'conv4_2')
        conv4_3_feats = misc.conv2d(conv4_2_feats, 512, name = 'conv4_3')
        pool4_feats = misc.max_pool2d(conv4_3_feats, name = 'pool4')

        conv5_1_feats = misc.conv2d(pool4_feats, 512, name = 'conv5_1')
        conv5_2_feats = misc.conv2d(conv5_1_feats, 512, name = 'conv5_2')
        conv5_3_feats = misc.conv2d(conv5_2_feats, 512, name = 'conv5_3')

        reshaped_conv5_3_feats = tf.reshape(conv5_3_feats,
                                            [config.BATCH_SIZE, 196, 512])

        return reshaped_conv5_3_feats