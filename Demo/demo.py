import demo_large_scale_predict as lcp
import demo_segment as seg
import demo_small_scale_predict as scp
import numpy as np
import os
import cv2 as cv
import matplotlib.pyplot as plt
from time import sleep
from scipy import ndimage

IMAGE_SIZE = 80
DAMAGE_INDEX = '3'

def img_resize_gray(img):
    img_shape = img.shape
    small_size = min(img_shape[0], img_shape[1])

    img = img[0:small_size, 0:small_size]
    img = cv.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    out = img
    outfile_ = 'scripts/demo_image/temp.png'
    cv.imwrite(outfile_, out)
    img = cv.imread(outfile_)

    if os.path.exists(outfile_):
        os.remove(outfile_)

    return img

def damage_locate(img, left, right, detect_size):
    origin = img
    img = img[:,left:right]
    # cv.imshow('i',img)
    # cv.waitKey(0)
    img_shape = img.shape
    h = img_shape[0]
    w = img_shape[1]

    list_i = []
    list_j = []
    # for i in range (0, h, IMAGE_SIZE):
        # for j in range (0, w, IMAGE_SIZE):
    for i in range (0, h, detect_size):
        for j in range (0, w, detect_size):
            temp_img = img[i:i+detect_size, j:j+detect_size]

            temp_img_blur = ndimage.gaussian_filter(temp_img, 3)
            temp_img = 2*temp_img - temp_img_blur


            out = temp_img
            outfile_ = 'scripts/demo_image/temp.png'
            cv.imwrite(outfile_, out)
            temp_img = cv.imread(outfile_)

            if os.path.exists(outfile_):
                os.remove(outfile_)
            pred = scp.small_scale_predict(temp_img)
            print(pred)

            if pred == DAMAGE_INDEX:
                if i<h-detect_size and j<w-detect_size:
                    list_i.append(j+left)
                    list_j.append(i)
                    print(i)
                    print(j)
                    print(list_i)
                    print(list_j)
                    # cv.imshow('w', temp_img)
                    # cv.waitKey(0)





    return (list_i, list_j)




if __name__ == '__main__':

    file_dir = "scripts/demo_image"
    image_name = "1.jpg"
    demo_img_origin = cv.imread(file_dir+'/'+image_name)

    demo_img_origin_gray = cv.cvtColor(demo_img_origin, cv.COLOR_BGR2GRAY)
    seg_out = seg.segment(file_dir)
    demo_img = seg_out[0]
    demo_left = seg_out[1]
    demo_right = seg_out[2]
    demo_img_gray = cv.cvtColor(demo_img, cv.COLOR_BGR2GRAY)

    demo_img_gray_resized = img_resize_gray(demo_img_gray)
    pred = lcp.large_scale_predict(demo_img_gray_resized)
    print(pred)
    if pred != DAMAGE_INDEX:
        print('NORMAL')

    else:
        print('DAMAGED')
        detect_size_small = 70
        list = damage_locate(demo_img_origin_gray, demo_left, demo_right, detect_size_small)
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)  # 锐化
        demo_img_gray = cv.filter2D(demo_img_gray, -1, kernel=kernel)
        # cv.imshow('t', demo_img_gray)
        # cv.waitKey(1111)
        list_i = list[0]
        list_j = list[1]
        target_length = len(list_i)
        if target_length > 0:
            for target_index in range(target_length):
                i = list_i[target_index]
                j = list_j[target_index]
                cv.rectangle(demo_img_origin, (i, j), (i+detect_size_small, j+detect_size_small), (255, 0, 0), 1)
        #
        # else:
        #     print('------========-=-=-')
        #     detect_size_large = 80
        #     list = damage_locate(demo_img_origin_gray, demo_left, demo_right, detect_size_large)
        #     list_i = list[0]
        #     list_j = list[1]
        #     target_length = len(list_i)
        #     for target_index in range(target_length):
        #         i = list_i[target_index]
        #         j = list_j[target_index]
        #         cv.rectangle(demo_img_origin, (i, j), (i + detect_size_large, j + detect_size_large), (255, 0, 0), 1)
        cv.imshow("result", demo_img_origin)
        cv.waitKey(0)

