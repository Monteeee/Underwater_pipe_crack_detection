# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import time
import cv2 as cv
import numpy as np
import tensorflow as tf
import os

DEBUG_FLAG = True
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

# def open_camera():
#     cap = cv.VideoCapture()
#     if cap.isOpened():
#         print('Camera opened')
#     return cap


def close_camera(cap):
    print('Camera closed')
    cap.release()
    cv.destroyAllWindows()
    return 0


def load_graph(model_file):
    graph = tf.Graph()
    graph_def = tf.GraphDef()

    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)

    return graph


def read_tensor_from_image_file(file_name, input_height=299, input_width=299,
                                input_mean=0, input_std=255):
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

    float_caster = tf.cast(image_reader, tf.float32)
    dims_expander = tf.expand_dims(float_caster, 0);
    resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
    normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
    sess = tf.Session()
    result = sess.run(normalized)

    return result


def read_tensor_from_camera(image_reader, input_height=299, input_width=299,
                            input_mean=0, input_std=255):
    input_name = "file_reader"
    output_name = "normalized"

    float_caster = tf.cast(image_reader, tf.float32)
    dims_expander = tf.expand_dims(float_caster, 0);
    resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
    normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
    sess = tf.Session()
    result = sess.run(normalized)

    return result


def load_labels(label_file):
    label = []
    proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
    for l in proto_as_ascii_lines:
        label.append(l.rstrip())
    return label


def return_status():
    file_name = "tf_files/flower_photos/daisy/3475870145_685a19116d.jpg"
    model_file = "tf_files/retrained_graph.pb"
    label_file = "tf_files/retrained_labels.txt"
    input_height = 128
    input_width = 128
    input_mean = 128
    input_std = 128
    input_layer = "input"
    output_layer = "final_result"

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", help="image to be processed")
    parser.add_argument("--graph", help="graph/model to be executed")
    parser.add_argument("--labels", help="name of file containing labels")
    parser.add_argument("--input_height", type=int, help="input height")
    parser.add_argument("--input_width", type=int, help="input width")
    parser.add_argument("--input_mean", type=int, help="input mean")
    parser.add_argument("--input_std", type=int, help="input std")
    parser.add_argument("--input_layer", help="name of input layer")
    parser.add_argument("--output_layer", help="name of output layer")
    args = parser.parse_args()

    if args.graph:
        model_file = args.graph
    # print(type(model_file))
    # print(model_file)
    if args.image:
        file_name = args.image
    # print(file_name)
    if args.labels:
        label_file = args.labels
    if args.input_height:
        input_height = args.input_height
    if args.input_width:
        input_width = args.input_width
    if args.input_mean:
        input_mean = args.input_mean
    if args.input_std:
        input_std = args.input_std
    if args.input_layer:
        input_layer = args.input_layer
    if args.output_layer:
        output_layer = args.output_layer

    model_file = 'tf_files/retrained_graph.pb'
    file_name = 'tf_files/test_data/hand_none/none_1.jpg'

    graph = load_graph(model_file)

    camera = cv.VideoCapture(1)


    count = 0
    frame_nums = 0
    while True:
        ret, frame = camera.read()
        x = 40
        y = 120
        frame = frame[x:480 - x, y:640 - y]
        origin = frame
        image_reader = tf.image.convert_image_dtype(origin, dtype=tf.uint8, name='jpeg_reader')

        t = read_tensor_from_camera(image_reader,
                                    input_height=input_height,
                                    input_width=input_width,
                                    input_mean=input_mean,
                                    input_std=input_std)

        input_name = "import/" + input_layer
        output_name = "import/" + output_layer
        input_operation = graph.get_operation_by_name(input_name);
        output_operation = graph.get_operation_by_name(output_name);

        with tf.Session(graph=graph) as sess:
            start = time.time()
            results = sess.run(output_operation.outputs[0],
                               {input_operation.outputs[0]: t})
            end = time.time()
        results = np.squeeze(results)

        top_k = results.argsort()[-5:][::-1]
        labels = load_labels(label_file)

        #print('\nEvaluation time (1-image): {:.3f}s\n'.format(end - start))
        template = "{} (score={:0.5f})"
        # for i in top_k:
        #     print(template.format(labels[i], results[i]))

        figure = frame
        green = (0, 255, 0)
        index = top_k[0]

        if index == 0:
            count -= 1
        elif index == 1:
            pass
        else:
            count += 1
        frame_nums+=1



        #print(index)
        pos = results[2]
        neg = results[0]
        none = results[1]
        if DEBUG_FLAG:
            cv.rectangle(origin, (20, 0), (60, int(150 * pos)), (0, 255, 0), thickness=-1)
            cv.rectangle(origin, (60, 0), (100, int(150 * neg)), (0, 0, 255), thickness=-1)
            cv.rectangle(origin, (100, 0), (140, int(150 * none)), (255, 255, 255), thickness=-1)
            cv.putText(origin, template.format(labels[index], results[index]), (150, 40), cv.FONT_HERSHEY_COMPLEX,
                       0.8,
                       green,
                       1)
            cv.imshow('Windows', origin)

            keyflag = cv.waitKey(1)
            if keyflag & 0xff == ord('q') or keyflag & 0xff == ord('Q'):
                close_camera(camera)
                break
        if frame_nums==8:
            if count >= 4:
                camera.release()
                return 2
            elif count <= -4:
                camera.release()
                return 0
            elif -2 <= count <= 2:
                camera.release()
                return 1
            else:
                frame_nums=0
                count = 0
while(True):
    a = return_status()
    print(a)
