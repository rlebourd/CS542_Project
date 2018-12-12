# CS542 Machine Learning Fall 2018
# Project SpaceXYZ Group 22 
# BuildNetVgg16.py
# Build FCN with pretrained VGG16 model

# Reference: Github https://github.com/shekkizh/FCN.tensorflow
# Reference: Github https://github.com/sagieppel/Fully-convolutional-neural-network-FCN-for-semantic-segmentation-Tensorflow-implementation

import os
import inspect
import numpy as np
import tensorflow as tf
import TensorflowUtils as utils

# Mean value of pixels in R G and B channels
VGG_MEAN = [245.23, 245.23, 245.23]
#VGG_MEAN = [9,77, 9.77, 9.77]

class BUILD_NET_VGG16:
    def __init__(self, vgg16_npy_path=None):
        self.data_dict = np.load(vgg16_npy_path, encoding='latin1').item()
        print("vgg16_npy loaded")

    def build(self, rgb, NUM_CLASSES, keep_prob):
        self.SumWeights = tf.constant(0.0, name="SumFiltersWeights")

        print("build model started")

        # BGR and substract pixels mean
        blue, green, red = tf.split(axis=3, num_or_size_splits=3, value=rgb)

        bgr = tf.concat(axis=3, values=[
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ])

       #Layer 1
        self.conv1_1 = self.conv_layer(bgr, "conv1_1")
        self.conv1_2 = self.conv_layer(self.conv1_1, "conv1_2")
        self.pool1 = self.max_pool(self.conv1_2, 'pool1')
       # Layer 2
        self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, "conv2_2")
        self.pool2 = self.max_pool(self.conv2_2, 'pool2')
        # Layer 3
        self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, "conv3_2")
        self.conv3_3 = self.conv_layer(self.conv3_2, "conv3_3")
        self.pool3 = self.max_pool(self.conv3_3, 'pool3')
        # Layer 4
        self.conv4_1 = self.conv_layer(self.pool3, "conv4_1")
        self.conv4_2 = self.conv_layer(self.conv4_1, "conv4_2")
        self.conv4_3 = self.conv_layer(self.conv4_2, "conv4_3")
        self.pool4 = self.max_pool(self.conv4_3, 'pool4')
        # Layer 5
        self.conv5_1 = self.conv_layer(self.pool4, "conv5_1")
        self.conv5_2 = self.conv_layer(self.conv5_1, "conv5_2")
        self.conv5_3 = self.conv_layer(self.conv5_2, "conv5_3")
        self.pool5 = self.max_pool(self.conv5_3, 'pool5')
        # Layer 6
        W6 = utils.weight_variable([7, 7, 512, 4096],name="W6")
        b6 = utils.bias_variable([4096], name="b6")
        self.conv6 = utils.conv2d_basic(self.pool5 , W6, b6)
        self.relu6 = tf.nn.relu(self.conv6, name="relu6")
        self.relu_dropout6 = tf.nn.dropout(self.relu6,keep_prob=keep_prob)
        # Layer 7
        W7 = utils.weight_variable([1, 1, 4096, 4096], name="W7")
        b7 = utils.bias_variable([4096], name="b7")
        self.conv7 = utils.conv2d_basic(self.relu_dropout6, W7, b7)
        self.relu7 = tf.nn.relu(self.conv7, name="relu7")
        self.relu_dropout7 = tf.nn.dropout(self.relu7, keep_prob=keep_prob)
        # Layer 8
        W8 = utils.weight_variable([1, 1, 4096, NUM_CLASSES],name="W8")
        b8 = utils.bias_variable([NUM_CLASSES], name="b8")
        self.conv8 = utils.conv2d_basic(self.relu_dropout7, W8, b8)

        # Upscale to actual image size
        deconv_shape1 = self.pool4.get_shape()
        W_t1 = utils.weight_variable([4, 4, deconv_shape1[3].value, NUM_CLASSES],name="W_t1") 
        b_t1 = utils.bias_variable([deconv_shape1[3].value], name="b_t1")
        self.conv_t1 = utils.conv2d_transpose_strided(self.conv8, W_t1, b_t1, output_shape=tf.shape(self.pool4))
        self.fuse_1 = tf.add(self.conv_t1, self.pool4, name="fuse_1")

        deconv_shape2 = self.pool3.get_shape()
        W_t2 = utils.weight_variable([4, 4, deconv_shape2[3].value, deconv_shape1[3].value], name="W_t2")
        b_t2 = utils.bias_variable([deconv_shape2[3].value], name="b_t2")
        self.conv_t2 = utils.conv2d_transpose_strided(self.fuse_1, W_t2, b_t2, output_shape=tf.shape(self.pool3))
        self.fuse_2 = tf.add(self.conv_t2, self.pool3, name="fuse_2")

        shape = tf.shape(rgb)
        W_t3 = utils.weight_variable([16, 16, NUM_CLASSES, deconv_shape2[3].value], name="W_t3")
        b_t3 = utils.bias_variable([NUM_CLASSES], name="b_t3")

        self.Prob = utils.conv2d_transpose_strided(self.fuse_2, W_t3, b_t3, output_shape=[shape[0], shape[1], shape[2], NUM_CLASSES], stride=8)
        self.Pred = tf.argmax(self.Prob, dimension=3, name="Pred")
        print("FCN model built")

    
    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, 
                              ksize   = [1, 2, 2, 1], 
                              strides = [1, 2, 2, 1], 
                              padding = 'SAME', name=name)


    def conv_layer(self, bottom, name):
        with tf.variable_scope(name):
            filt = self.get_conv_filter(name)
            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)
            relu = tf.nn.relu(bias)
            return relu


    def fc_layer(self, bottom, name):
        with tf.variable_scope(name):
            shape = bottom.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *= d
            x = tf.reshape(bottom, [-1, dim])
            weights = self.get_fc_weight(name)
            biases = self.get_bias(name)
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)
            return fc


    def get_conv_filter(self, name):
        var=tf.Variable(self.data_dict[name][0], name="filter_" + name)
        self.SumWeights+=tf.nn.l2_loss(var)
        return var


    def get_bias(self, name):
        return tf.Variable(self.data_dict[name][1], name="biases_"+name)


    def get_fc_weight(self, name):
        return tf.Variable(self.data_dict[name][0], name="weights_"+name)

