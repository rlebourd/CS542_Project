# CS542 Machine Learning Fall 2018
# Project SpaceXYZ Group 22 
# Train.py
# Train FCN with pretrained VGG16 model

# Reference: Github https://github.com/shekkizh/FCN.tensorflow
# Reference: Github https://github.com/sagieppel/Fully-convolutional-neural-network-FCN-for-semantic-segmentation-Tensorflow-implementation

# Train fully convolutional neural net for sematic segmentation
# Instructions:
# 1) Set folder of train images in Train_Image_Dir
# 2) Set folder for ground truth labels in Train_Label_Dir
#    The Label Maps should be saved as png image with same name as the corresponding image and png ending
# 3) Download pretrained vgg16 model and put in model_path: ftp://mi.eng.cam.ac.uk/pub/mttt2/models/vgg16.npy
##########################################################################################################################################################################

import os
import numpy as np
import tensorflow as tf
import scipy.misc as misc

import Data_Reader
import BuildNetVgg16

model_path = "Model_Zoo/vgg16.npy"

Train_Image_Dir = "Data_Zoo/ori_bgr_inv"
Train_Label_Dir = "Data_Zoo/GT_Binary"

logs_dir = "logs"
if not os.path.exists(logs_dir): os.makedirs(logs_dir)
TrainLossTxtFile = logs_dir + "/TrainLoss.txt"


learning_rate = 1e-5      # Learning rate for Adam Optimizer
Weight_Loss_Rate = 5e-4   # Weight for the weight decay loss function
Batch_Size = 2

MAX_ITERATION = int(501)  # 5 epochs
NUM_CLASSES = 2


def train(loss_val, var_list):
    with tf.device('/gpu:0'):
        # AdamOPtimizer or MomentumOptimizer
        optimizer = tf.train.AdamOptimizer(learning_rate)
        #optimizer = tf.train.MomentumOptimizer(learning_rate, Weight_Loss_Rate)
        grads = optimizer.compute_gradients(loss_val, var_list=var_list)
        return optimizer.apply_gradients(grads)


def main():
    tf.reset_default_graph()
    keep_prob = tf.placeholder(tf.float32, name="keep_probabilty")                       # Dropout probability

    image = tf.placeholder(tf.float32, shape=[None, None, None, 3], name="input_image")
    GTLabel = tf.placeholder(tf.int32, shape=[None, None, None, 1], name="GTLabel")
 
    Net =  BuildNetVgg16.BUILD_NET_VGG16(vgg16_npy_path=model_path)
    Net.build(image, NUM_CLASSES, keep_prob)

    # Cross-entropy Loss
    Loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.squeeze(GTLabel, squeeze_dims=[3]), logits=Net.Prob, name="Loss")))

    trainable_var = tf.trainable_variables()
    train_op = train(Loss, trainable_var)

    TrainReader = Data_Reader.Data_Reader(ImageDir=Train_Image_Dir,  GTLabelDir=Train_Label_Dir, BatchSize=Batch_Size)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
 
        saver = tf.train.Saver()
        
        ckpt = tf.train.get_checkpoint_state(logs_dir)
        # If train model exists, then we restore it
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("Model restored...")
        # Record Training Loss
        f = open(TrainLossTxtFile, "w")
        f.write("Iteration\tloss\t Learning Rate=" + str(learning_rate))
        f.close()
        
        for itr in range(MAX_ITERATION):
            Images,  GTLabels = TrainReader.ReadAndAugmentNextBatch()
            feed_dict = {image: Images,GTLabel:GTLabels, keep_prob: 0.5}
            sess.run(train_op, feed_dict=feed_dict)
            # Save the model every 100 iters
            if itr % 100 == 0 and itr>0:
                print("Saving Model to file in " + logs_dir)
                saver.save(sess, logs_dir + "model.ckpt", itr)
            # Record train loss every 10 iters
            if itr % 10==0:
                feed_dict = {image: Images, GTLabel: GTLabels, keep_prob: 1}
                TLoss=sess.run(Loss, feed_dict=feed_dict)
                print("Step "+str(itr)+" Train Loss=" + str(TLoss))
                with open(TrainLossTxtFile, "a") as f:
                    f.write("\n"+str(itr)+"\t"+str(TLoss))
                    f.close()

main()
print("Finished")
