import numpy as np
from matplotlib import pyplot as plt

## Finds and fills a contiguous region of zeros with a given FillValue in a 2D Input matrix
#
# This function finds a contiguous region of zeros within the Input matrix. It then fills
# that region of the input matrix with the given FillValue.
def FloodFill(Input, FillValue):
	m = Input.shape[0]
	n = Input.shape[1]
	
	# find the index of an entry equal to 0
	for i in range(m):
		for j in range(n):
			if Input[i, j] == 0:
				break
		if Input[i, j] == 0:
			break

	Output = Input.copy()

	# return the output since it does not contain any zeros
	if Input[i, j] != 0:
		return Output
	
	# fill the contiguous region of zeros with the given FillValue via breadth-first algorithm
	InQueue = [(i, j)]
	OutQueue = []
	while len(InQueue) > 0:
		for index in InQueue:
			for di in range(-1, 2):
				for dj in range(-1, 2):
					i = index[0]+di
					j = index[1]+dj
					if i < Output.shape[0] and j < Output.shape[1] and Output[i, j] == 0:
						Output[i, j] = FillValue
						OutQueue.append((i, j))
	
		InQueue = OutQueue
		OutQueue = []
		
	return Output

## Partitions the 2D Input matrix
def Partition(Input):
	pass

## FloodFill unit test
def UnitTestFloodFill():
	Input = np.array([[1, 1, 0, 0], [1, 0, 0, 1], [1, 1, 0, 1]])
	print('Input is', Input)
	Output = FloodFill(Input, 5)
	print('Output is', Output)
	
	plt.figure()
	plt.imshow(Input)
	
	plt.figure()
	plt.imshow(Output)
	plt.show()
	
UnitTestFloodFill()