
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

NUM_of_IMGS = 100

for i in range(1, NUM_of_IMGS+1):
    img = cv.imread('./ori/' + str(i) + '.png', 0)
    #_, th1 = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
    cv.imwrite('./ori_gray/' + str(i) + '.png', img)

