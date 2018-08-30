import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d
from scipy.signal import argrelextrema as extrema
from os import listdir

def segment(dir):
    read_dir = dir
    save_dir = dir
    err_dir = dir
    file_list = listdir(read_dir)
    print("total original image number " + str(len(file_list)))
    file_list.sort()
    # print file_list
    count = 0
    for filename in file_list:
        print("processing image No." + str(count))
        # print(filename)
        img = cv2.imread(read_dir + '/' + filename)
        origin = img.copy()
        gray1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray1)

        width, height = gray1.shape

        groups = [range(0, width - 1), \
                  range(int(np.rint(width * 0.0)), int(np.rint(width * 0.25))), \
                  range(int(np.rint(width * 0.25)), int(np.rint(width * 0.5))), \
                  range(int(np.rint(width * 0.5)), int(np.rint(width * 0.75))), \
                  range(int(np.rint(width * 0.75)), width - 1)]

        left = 0
        right = height - 1
        for index_range in groups:
            found_solution = False
            average = np.mean(gray1[index_range, :], axis=0)
            sigma = 20
            sig2 = 10
            points = []
            last = 0
            while (len(points) != 2):
                filtered = gaussian_filter1d(average, sigma, truncate=2 * sigma)
                filtered = gaussian_filter1d(filtered, sig2, truncate=2 * sig2)
                minima = extrema(filtered, np.less)
                points = minima[0]
                if len(points) < 2:
                    sigma = max(sigma - 1, 1)
                    sig2 = max(sig2 - 1, 1)
                elif len(points) > 2:
                    sigma = min(sigma + 1, 30)
                    sig2 = min(sig2 + 1, 30)

                if len(points) < 2:
                    sigma = max(sigma - 1, 1)
                    sig2 = max(sig2 - 1, 1)
                    if last == 1:
                        break
                    last = -1
                elif len(points) > 2:
                    sigma = min(sigma + 1, 30)
                    sig2 = min(sig2 + 1, 30)
                    if last == -1:
                        break
                    last = 1
                else:
                    found_solution = True

                if sigma == 1 and sig2 == 1:
                    # print("low limit reached")
                    break

                if sigma == 30 and sig2 == 30:
                    # print("high limit reached")
                    break

            if found_solution:
                if 120 < abs(points[0] - points[1]) < 300:
                    if points[0] > points[1]:
                        left = points[1]
                        right = points[0]
                    else:
                        left = points[0]
                        right = points[1]
                    break

        if not found_solution or (abs(points[0] - points[1]) > 300 or abs(points[0] - points[1]) < 100):
            print("did not find boundary!")
            out = img
            outfile_ = err_dir + '/' + filename
            # cv2.imwrite(outfile_, out)
        else:
            out = img[:, left:right, :]
            outfile_ = save_dir + '/' + str(count) + '.png'
            # cv2.imwrite(outfile_, out)

        count = count + 1
        return (out, left, right)

