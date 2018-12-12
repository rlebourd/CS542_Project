# CS542 Machine Learning Fall 2018
# Project SpaceXYZ Group 22 
# Inference.py
# Predict with trained FCN model

# Reference: Github https://github.com/shekkizh/FCN.tensorflow
# Reference: Github https://github.com/sagieppel/Fully-convolutional-neural-network-FCN-for-semantic-segmentation-Tensorflow-implementation

# Run prediction and genertae pixelwise annotation for every pixels in the image using fully coonvolutional neural net
# Output saved as label images, and label image overlay on the original image
# 1) Make sure you you have trained model in logs_dir
# 2) Set the Image_Dir to the folder where the input image for prediction are located
# 3) Set number of classes number in NUM_CLASSES
# 4) Set Pred_Dir the folder where you want the output annotated images to be save

import os
import sys
import numpy as np
import tensorflow as tf
import scipy.misc as misc

import BuildNetVgg16
import TensorflowUtils
import Data_Reader
import OverrlayLabelOnImage as Overlay

model_path="Model_Zoo/vgg16.npy"
logs_dir= "logs_dir/"

Image_Dir="Data_Zoo/ori_bgr_inv/"
Pred_Dir="Output_Prediction/"
NameEnd=""

NUM_CLASSES = 2
w=0.6


def main(argv=None):
    keep_prob = tf.placeholder(tf.float32, name="keep_probabilty")  # Dropout probability
    image = tf.placeholder(tf.float32, shape=[None, None, None, 3], name="input_image")

    Net = BuildNetVgg16.BUILD_NET_VGG16(vgg16_npy_path=model_path)
    Net.build(image, NUM_CLASSES, keep_prob)
 
    ValidReader = Data_Reader.Data_Reader(Image_Dir,  BatchSize=1)

    sess = tf.Session()
    print("Setting up Saver...")
    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state(logs_dir)
    # If train model exists, then we restore it
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored Successfully...")
    else:
        print("ERROR! NO Trained Model in: " + ckpt.model_checkpoint_path)
        sys.exit()

    if not os.path.exists(Pred_Dir): os.makedirs(Pred_Dir)
    if not os.path.exists(Pred_Dir + "/Label"): os.makedirs(Pred_Dir + "/Label")
    if not os.path.exists(Pred_Dir+"/OverLay"): os.makedirs(Pred_Dir+"/OverLay")

    print("Running Predictions:")
    print("Saving output to:" + Pred_Dir)

    fim = 0
    print("Start Predicting " + str(ValidReader.NumFiles) + " images")
    while (ValidReader.itr < ValidReader.NumFiles):
        print(str(fim * 100.0 / ValidReader.NumFiles) + "%")
        fim += 1
  
        FileName=ValidReader.OrderedFiles[ValidReader.itr]
        Images = ValidReader.ReadNextBatchClean()

        # Predict
        LabelPred = sess.run(Net.Pred, feed_dict={image: Images, keep_prob: 1.0})
        # Save the prediction
        misc.imsave(Pred_Dir + "/OverLay/"+ FileName+NameEnd  , Overlay.OverLayLabelOnImage(Images[0],LabelPred[0], w)) #Overlay label on image
        misc.imsave(Pred_Dir + "/Label/" + FileName[:-4] + ".png" + NameEnd, LabelPred[0].astype(np.uint8))
        ##################################################################################################################################################

main()
print("Finished")