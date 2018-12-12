
import sys
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

print(sys.argv[1])
img = cv.imread(sys.argv[1], 0)
plt.imshow(img, 'gray')
plt.show()
