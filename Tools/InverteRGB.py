
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

NUM_of_IMGS = 100

for i in range(1, NUM_of_IMGS+1):
    img = cv.imread('./ori/' + str(i) + '.png', 1)
    img_inv = cv.bitwise_not(img)
    cv.imwrite('./ori_rgb_inv/' + str(i) + '.png', img_inv)
