from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import time
import os
import imtools as its
from sklearn.metrics import confusion_matrix

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib import cm
from scipy.ndimage import imread
from scipy.ndimage import gaussian_filter


os.environ["CUDA_VISIBLE_DEVICES"]="-1"


def load_graph(model_file):
    graph = tf.Graph()
    graph_def = tf.GraphDef()

    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)

    return graph


def read_tensor_from_image_file(file_name, input_height=299, input_width=299,
                                input_mean=0, input_std=255, occ_x=40, occ_y=40, range=10):
    input_name = "file_reader"
    output_name = "normalized"
    file_reader = tf.read_file(file_name, input_name)
    if file_name.endswith(".png"):
        image_reader = tf.image.decode_png(file_reader, channels=3,
                                           name='png_reader')
    elif file_name.endswith(".gif"):
        image_reader = tf.squeeze(tf.image.decode_gif(file_reader,
                                                      name='gif_reader'))
    elif file_name.endswith(".bmp"):
        image_reader = tf.image.decode_bmp(file_reader, name='bmp_reader')
    else:
        image_reader = tf.image.decode_jpeg(file_reader, channels=3,
                                            name='jpeg_reader')

    # too occlude some part of the image and see the effect
    # on predicted class probability
    mask = np.ones([80, 80], dtype=np.uint8)
    mask[ max(occ_x - range, 0):min(occ_x + range, 80), max(occ_y - range, 0):min(occ_y + range, 80) ] = 0
    mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)

    # img_mask = imread('/media/jiajuns/LENOVO/mobileHDbackup/From Asus laptop/Underwater_pipe_crack_detection/tfmobile/tf_files/iampng.png', mode='RGB')
    # mask2 = np.zeros([80, 80, 3], dtype=np.uint8)
    # print(mask2.shape)
    # print(img_mask.shape)
    # mask2[max(occ_x - range, 0):min(occ_x + range, 80), max(occ_y - range, 0):min(occ_y + range, 80), :] = \
    #     img_mask[max(occ_x - range, 0):min(occ_x + range, 80), max(occ_y - range, 0):min(occ_y + range, 80), :]

    sess = tf.Session()
    with sess.as_default():
        cut = image_reader.eval()
        cut = cut[max(occ_x - range, 0):min(occ_x + range, 80), max(occ_y - range, 0):min(occ_y + range, 80), :]
        cut = gaussian_filter(cut, sigma=2)
    mask2 = np.zeros([80, 80, 3], dtype=np.uint8)
    mask2[max(occ_x - range, 0):min(occ_x + range, 80), max(occ_y - range, 0):min(occ_y + range, 80), :] = cut

    image_reader_occ = tf.multiply(image_reader, tf.cast(mask, tf.uint8))
    image_reader_occ = tf.add(image_reader_occ, tf.cast(mask2, tf.uint8))

    float_caster = tf.cast(image_reader_occ, tf.float32)
    dims_expander = tf.expand_dims(float_caster, 0)
    resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
    normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
    # sess = tf.Session()
    # if occ_y == 50 and occ_x == 35:
    #     with sess.as_default():
    #         #print(resized.eval()) # this line worked
    #         resized = np.asarray(resized.eval(), dtype=np.uint8)
    #         resized = np.squeeze(resized)
    #         #print(resized.shape)
    #     plt.imshow(resized, cmap=cm.Greys)
    #     plt.show()
    #     print("------------------  plot -------------------")

    sess = tf.Session()
    result = sess.run(normalized)

    return result


def load_labels(label_file):
    label = []
    proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
    for l in proto_as_ascii_lines:
        label.append(l.rstrip())
    return label


def get_imlist(path):
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.png')]


if __name__ == "__main__":

    model_file = "tf_files/retrained_graph.pb"
    label_file = "tf_files/retrained_labels.txt"
    graph = load_graph(model_file)
    labels = load_labels(label_file)
    input_height = 128
    input_width = 128
    input_mean = 128
    input_std = 128
    input_layer = "input"
    output_layer = "final_result"
    input_name = "import/" + input_layer
    output_name = "import/" + output_layer
    input_operation = graph.get_operation_by_name(input_name);
    output_operation = graph.get_operation_by_name(output_name);
    y_true = []
    y_pred = []

    # the image
    img_hard_h = 80
    img_hard_w = 80
    x_count = 0
    y_count = 0
    stride = 5
    prob_map = np.zeros([img_hard_h // stride, img_hard_w // stride])
    class_map = np.zeros([img_hard_h // stride, img_hard_w // stride], dtype=np.uint8)

    img_num = 1
    for cx in np.arange(0, img_hard_h-1, stride):
        for cy in np.arange(0, img_hard_w, stride):
            # file_name = "tf_files/test_photos/0/1.png"
            file_index = 3
            file_name = "tf_files/test_photos/"+str(file_index)+"/"+str(img_num)+".png"

            # read in one image
            t = read_tensor_from_image_file(file_name,
                                            input_height=input_height,
                                            input_width=input_width,
                                            input_mean=input_mean,
                                            input_std=input_std,
                                            occ_x=cx,
                                            occ_y=cy,
                                            range=15)

            with tf.Session(graph=graph) as sess:
                results = sess.run(output_operation.outputs[0],
                                   {input_operation.outputs[0]: t})

            results = np.squeeze(results)
            # "results" here contain the value of prediction probability of each class
            top_k = results.argsort()[-5:][::-1]

            y_true.append(file_index)
            pred_index = top_k[0]
            y_pred.append(pred_index)
            print("<<<<<<<<<<<<<< result >>>>>>>>>>>>>>")
            print("pixel " + '[' + str(cx) + ', ' + str(cy) + ']')
            print(results)
            print("predicted label: " + str(pred_index))
            print("====================================")
            prob_map[x_count, y_count] = results[file_index]
            class_map[x_count, y_count] = pred_index
            # template = "{} (score={:0.5f})"
            # for i in top_k:
            #     print(template.format(labels[i], results[i]))
            #break
            y_count = y_count + 1
        #break
        y_count = 0
        x_count = x_count + 1
    x_count = 0

    #C = confusion_matrix(y_true, y_pred)
    #C_n = C.astype('float') / C.sum(axis=1)[:, np.newaxis]
    cmap = plt.cm.rainbow
    norm = BoundaryNorm(np.arange(-0.5, 4.5, 1), cmap.N)

    plt.subplot(1, 2, 1)
    plt.matshow(prob_map, fignum=False, cmap=plt.cm.gnuplot)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.subplot(1, 2, 2)
    plt.matshow(class_map, fignum=False, cmap=cmap,norm=norm)
    plt.colorbar(fraction=0.046, pad=0.04, ticks=np.linspace(0,3,4))
    plt.show()



    #print(C)
    #print(C_n)
