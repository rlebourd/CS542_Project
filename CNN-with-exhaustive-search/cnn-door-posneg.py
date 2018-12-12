import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import os

tf.logging.set_verbosity(tf.logging.ERROR)

DOOR_WIDTH = 80
DOOR_HEIGHT = 80

class DoorLocalizer:
	def __init__(self, X, y, InputWidth, InputHeight, NumClasses):
		self.filename = './door_localizer_model.ckpt'
		
		self.X = X
		self.y = y
	
		# Construction Phase		
		with tf.name_scope('reshape'):
			self.x = tf.reshape(self.X, [-1, InputWidth, InputHeight, 1])
		
#		with tf.name_scope('convrelu1'):
#			# tensorflow's tf.layers.conv2d function creates a convolutional layer with
#			# the weights initialized to random values drawn from a normal distribution,
#			# and with the biases initialized to zero.
#			self.conv1 = tf.layers.conv2d(inputs=self.x, filters=32, kernel_size=(5,5), strides=(2,2), padding='SAME')
#			self.convrelu1 = tf.nn.relu(features=self.conv1)
#			print('conv1 has shape', self.convrelu1.shape)
#
#		with tf.name_scope('pool1'):
#			# downsample by a factor of 2
#			self.pool1 = tf.nn.max_pool(value=self.convrelu1, ksize=(1,2,2,1), strides=(1,2,2,1), padding='VALID')
#			print('pool1 has shape', self.pool1.shape)
#
#		with tf.name_scope('convrelu2'):
#			self.conv2 = tf.layers.conv2d(inputs=self.pool1, filters=64, kernel_size=(5, 5), strides=(1, 1), padding='SAME')
#			self.convrelu2 = tf.nn.relu(features=self.conv2)
#			print('conv2 has shape', self.convrelu2.shape)
#
#		with tf.name_scope('pool2'):
#			# downsample by a factor of 2
#			self.pool2 = tf.nn.max_pool(value=self.convrelu2, ksize=(1,2,2,1), strides=(1,2,2,1), padding='VALID')
#			print('pool2 has shape', self.pool2.shape)

		with tf.name_scope('convrelu1'):
			# tensorflow's tf.layers.conv2d function creates a convolutional layer with
			# the weights initialized to random values drawn from a normal distribution,
			# and with the biases initialized to zero.
			self.conv1 = tf.layers.conv2d(inputs=self.x, filters=32, kernel_size=(3,3), strides=(2,2), padding='SAME')
			self.convrelu1 = tf.nn.relu(features=self.conv1)
			print('conv1 has shape', self.convrelu1.shape)

		with tf.name_scope('pool1'):
			# downsample by a factor of 2
			self.pool1 = tf.nn.max_pool(value=self.convrelu1, ksize=(1,2,2,1), strides=(1,2,2,1), padding='VALID')
			print('pool1 has shape', self.pool1.shape)

		with tf.name_scope('convrelu2'):
			self.conv2 = tf.layers.conv2d(inputs=self.pool1, filters=64, kernel_size=(3, 3), strides=(1, 1), padding='SAME')
			self.convrelu2 = tf.nn.relu(features=self.conv2)
			print('conv2 has shape', self.convrelu2.shape)

		with tf.name_scope('pool2'):
			# downsample by a factor of 2
			self.pool2 = tf.nn.max_pool(value=self.convrelu2, ksize=(1,2,2,1), strides=(1,2,2,1), padding='VALID')
			print('pool2 has shape', self.pool2.shape)

		with tf.name_scope('convrelu3'):
			self.conv3 = tf.layers.conv2d(inputs=self.pool2, filters=128, kernel_size=(3, 3), strides=(1, 1), padding='SAME')
			self.convrelu3 = tf.nn.relu(features=self.conv3)
			print('conv3 has shape', self.convrelu3.shape)

		with tf.name_scope('pool3'):
			# downsample by a factor of 2
			self.pool3 = tf.nn.max_pool(value=self.convrelu3, ksize=(1,2,2,1), strides=(1,2,2,1), padding='VALID')
			print('pool3 has shape', self.pool3.shape)

		self.last_pool = self.pool3

		with tf.name_scope('fc1'):
			input_size = self.last_pool.shape[1]*self.last_pool.shape[2]*self.last_pool.shape[3]
			self.last_pool_flat = tf.reshape(self.last_pool, [-1, int(input_size)])
			self.fc1 = tf.layers.dense(inputs=self.last_pool_flat,
										units=1024,
										activation=tf.nn.relu,
										use_bias=True,
										kernel_initializer=None, # weights are initialized using the default initializer used by tf.get_variable
										bias_initializer=None,
										trainable=True,
										name='fc1')
			print('fc1 has shape', self.fc1.shape)

		with tf.name_scope('fc2'):
			self.fc2 = tf.layers.dense(inputs=self.fc1, 
										units=NumClasses,
										activation=None,
										use_bias=True,
										kernel_initializer=None, # weights are initialized using the default initializer used by tf.get_variable
										bias_initializer=None,
										trainable=True,
										name='fc2')	
			print('fc2 has shape', self.fc2.shape)
			
		with tf.name_scope('loss'):
			self.cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=tf.argmax(self.y, axis=1), logits=self.fc2)
			self.cross_entropy = tf.reduce_mean(self.cross_entropy)
		
		with tf.name_scope('adam_optimizer'):
			self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.cross_entropy)
		
		with tf.name_scope('accuracy'):
			self.correct_prediction = tf.equal(tf.argmax(self.fc2, axis=1), tf.argmax(self.y, axis=1))
			self.correct_prediction = tf.cast(self.correct_prediction, tf.float32)
			self.accuracy = tf.reduce_mean(self.correct_prediction)
	
		self.saver = tf.train.Saver()

	def Train(self, X, y):
		# Execution Phase
		inputs = []
		outputs = []
		for line in open('doorex.txt'):
			input = [float(numstr) for numstr in line.split(",")]
			output = [0, 0]
			output[int(input[-1])] = 1
			del input[-1]
			inputs.append(input)
			outputs.append(output)

		NUM_EXAMPLES = len(inputs)
		inputs_train = inputs[0:int(NUM_EXAMPLES*0.8)]
		outputs_train = outputs[0:int(NUM_EXAMPLES*0.8)]

		inputs_validation = inputs[int(NUM_EXAMPLES*0.8):]
		outputs_validation = outputs[int(NUM_EXAMPLES*0.8):]
		
		
		with tf.Session() as sess:
			summary_writer = tf.summary.FileWriter(os.getcwd() + '/tensorboardlog',
																						 sess.graph)

			sess.run(tf.global_variables_initializer())
			#self.saver.restore(sess, self.filename)
			for i in range(50):
				BATCH_SIZE = 50
				for batch_index in range(int(NUM_EXAMPLES*0.8/50)):
					batch0 = inputs_train[batch_index*BATCH_SIZE:(batch_index+1)*BATCH_SIZE]
					batch1 = outputs_train[batch_index*BATCH_SIZE:(batch_index+1)*BATCH_SIZE]
					if i % 10 == 0 and batch_index == 0:
						train_accuracy = self.accuracy.eval(feed_dict={X: batch0, y: batch1})
						print('%d\t%g\t%g' % (i, train_accuracy, self.accuracy.eval(feed_dict={X: inputs_validation, y: outputs_validation})))
					self.train_step.run(feed_dict={X: batch0, y: batch1})
					self.saver.save(sess, self.filename)

	def FindDoors(self, FloorPlanImage, Restore=False):
		Output = np.zeros(FloorPlanImage.shape)
		with tf.Session() as sess:
			if Restore:
				self.saver.restore(sess, self.filename)
			NUM_ROWS, NUM_COLS = FloorPlanImage.shape
			DISP_Y = 10
			DISP_X = 10
			for i in range(0, int((NUM_ROWS-DOOR_HEIGHT)/DISP_Y)): # iterate over the rows
				for j in range(0, int((NUM_COLS-DOOR_WIDTH)/DISP_X)): # iterate over the columns
					row = int(i*DISP_Y)
					col = int(j*DISP_X)
					SubImageToClassify = np.reshape(FloorPlanImage[row:int(row+DOOR_HEIGHT), col:int(col+DOOR_WIDTH)], [1, -1])
					if DOOR_HEIGHT*DOOR_WIDTH == SubImageToClassify.shape[1]:
						Result = np.argmax(self.fc2.eval(feed_dict={X: SubImageToClassify}), axis=1)
						Output[row:row+DOOR_HEIGHT, col:col+DOOR_WIDTH] += 10*Result;
		return Output

InputWidth = DOOR_WIDTH
InputHeight = DOOR_HEIGHT
NumClasses = 2
X = tf.placeholder(dtype=tf.float32, shape=(None, InputWidth*InputHeight), name='X') # input to a CNN is a tensor of shape (batch size, img width, img height, img depth)
y = tf.placeholder(dtype=tf.int64,   shape=(None, NumClasses), 	  name='y') # output from a CNN is a tensor of shape (batch size, num classes)
			
algorithm = DoorLocalizer(X, y, InputWidth, InputHeight, NumClasses)
algorithm.Train(X, y)

"""
datafile = open('1-test.txt') # positive examples (i.e. images of doors)
input = []

for line in datafile:
	input.append([float(numstr) for numstr in line.split(",")])

Output = algorithm.FindDoors(FloorPlanImage=np.array(input), Restore=True)
Output = 255*(Output > np.max(Output)*0.8)
plt.imshow(Output, cmap='gray')
plt.show()
"""
