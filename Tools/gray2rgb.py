
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

NUM_of_IMGS = 100

for i in range(1, NUM_of_IMGS+1):
    img = cv.imread('./ori_th20/' + str(i) + '.png', 1)
    #_, th1 = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
    cv.imwrite('./ori_bgr_th20/' + str(i) + '.png', img)
