
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

NUM_of_IMGS = 100
b_avg_sum, g_avg_sum, r_avg_sum = 0, 0, 0

for i in range(1, NUM_of_IMGS+1):
    b_sum, g_sum, r_sum = 0, 0, 0
    img = cv.imread('./ori_bgr_inv/' + str(i) + '.png', 1) # BGR
    width, height, _ = img.shape
    num_of_pixles = width * height
    for w in range(width):
        for h in range(height):
            b_sum += img[w][h][0] 
            g_sum += img[w][h][1] 
            r_sum += img[w][h][2]
    b_avg_sum += b_sum / num_of_pixles
    g_avg_sum += g_sum / num_of_pixles
    r_avg_sum += r_sum / num_of_pixles
    print('img ' + str(i) + ' done')
b_avg = b_avg_sum / NUM_of_IMGS
g_avg = g_avg_sum / NUM_of_IMGS
r_avg = r_avg_sum / NUM_of_IMGS

print('Mean of b, g, r:', b_avg, g_avg, r_avg)
