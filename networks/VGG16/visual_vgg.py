import time
import argparse
import os
import numpy as np
import glob
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.externals import joblib
from keras import backend as K
from matplotlib import pyplot as plt

import config
import util
from scipy.ndimage import gaussian_filter
from matplotlib.colors import BoundaryNorm


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--path', dest='path', help='Path to image', default=None, type=str)
    parser.add_argument(
        '--accuracy', action='store_true', help='To print accuracy score')
    parser.add_argument('--plot_confusion_matrix', action='store_true')
    parser.add_argument('--execution_time', action='store_true')
    parser.add_argument('--store_activations', action='store_true')
    parser.add_argument('--novelty_detection', action='store_true')
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Base model architecture',
        choices=[
            config.MODEL_RESNET50, config.MODEL_RESNET152,
            config.MODEL_INCEPTION_V3, config.MODEL_VGG16
        ])
    parser.add_argument('--data_dir', help='Path to data train directory')
    parser.add_argument(
        '--batch_size',
        default=50,
        type=int,
        help='How many files to predict on at once')
    args = parser.parse_args()
    return args


def get_files(path):
    if os.path.isdir(path):
        print(path)
        path1 = path
        path = path + '1/'
        print(path)
        files = glob.glob(path + '*.png')
        path = path1 + '0/'
        files1 = glob.glob(path + '*.png')
        files = files + files1
        path = path1 + '2/'
        files1 = glob.glob(path + '*.png')
        files = files + files1
        path = path1 + '3/'
        files1 = glob.glob(path + '*.png')
        files = files + files1
    elif path.find('*') > 0:
        files = glob.glob(path)
    else:
        files = [path]

    if not len(files):
        print('No images found by the given path')
        exit(1)

    return files


def get_inputs_and_trues(files):
    inputs = []
    y_true = []
    vgg_mean = np.array([103.939, 116.779, 123.68], dtype=np.float32)
    for i in files:
        x = model_module.load_img(i)
        # print("=== before ===")
        # print(x[50, 50, :])
        # x = np.subtract(x, vgg_mean)
        # x = x * 1./255
        # print("=== after ===")
        # print(x[50, 50, :])
        try:
            image_class = i.split(os.sep)[-2]
            keras_class = int(classes_in_keras_format[image_class])
            y_true.append(keras_class)
        except Exception:
            # print(os.path.split(i))
            y_true.append(os.path.split(i)[1])
        # plt.figure(3)
        # plt.imshow(x, cmap='gray')
        # print(x[:, :, 0])
        # print(x[:, :, 1])
        # print(x[:, :, 2])
        # plt.show()
        inputs.append(x)

    return y_true, inputs


def predict(path):
    files = get_files(path)
    n_files = len(files)
    print('Found {} files'.format(n_files))

    if args.novelty_detection:
        activation_function = util.get_activation_function(
            model, model_module.noveltyDetectionLayerName)
        novelty_detection_clf = joblib.load(
            config.get_novelty_detection_model_path())

    range = 15
    img_hard_h = 80
    img_hard_w = 80
    x_count = 0
    y_count = 0
    stride = 5
    prob_map = np.zeros([img_hard_h // stride, img_hard_w // stride])
    class_map = np.zeros([img_hard_h // stride, img_hard_w // stride], dtype=np.uint8)

    n = 0
    class_index = 3
    image_index = 1

    y_true, inputs = get_inputs_and_trues(["data/sorted/test/" + str(class_index) + \
                                          "/" + str(image_index) + ".png"])
    for cx in np.arange(0, img_hard_h-1, stride):
        for cy in np.arange(0, img_hard_w, stride):

            if not args.store_activations:
                # Warm up the model
                if n == 0:
                    print('Warming up the model')
                    start = time.clock()
                    model.predict(np.array([inputs[0]]))
                    end = time.clock()
                    print('Warming up took {} s'.format(end - start))
                    n = 1

                cut = inputs[0][max(cx - range, 0): min(cx + range, 80), max(cy - range, 0): min(cy + range, 80), :]
                cut = gaussian_filter(cut, sigma=3)
                inputs[0][max(cx - range, 0): min(cx + range, 80), max(cy - range, 0): min(cy + range, 80), :] = cut

                # Make predictions
                start = time.clock()
                out = model.predict(np.array(inputs))
                end = time.clock()
                pred_index = np.argmax(out, axis=1)
                print("<<<<<<<<<<<<<< result >>>>>>>>>>>>>>")
                print(out)
                print("predicted label is " + str(pred_index))
                print("====================================")
                prob_map[x_count, y_count] = out[0][class_index]
                class_map[x_count, y_count] = pred_index

                y_count = y_count + 1
        y_count = 0
        x_count = x_count + 1
    x_count = 0

    cmap = plt.cm.rainbow
    norm = BoundaryNorm(np.arange(-0.5, 4.5, 1), cmap.N)

    plt.subplot(1, 2, 1)
    plt.matshow(prob_map, fignum=False, cmap=plt.cm.gnuplot)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.subplot(1, 2, 2)
    plt.matshow(class_map, fignum=False, cmap=cmap,norm=norm)
    plt.colorbar(fraction=0.046, pad=0.04, ticks=np.linspace(0,3,4))
    plt.show()

if __name__ == '__main__':
    tic = time.clock()

    args = parse_args()
    print('=' * 50)
    print('Called with args:')
    print(args)

    if args.data_dir:
        config.data_dir = args.data_dir
        config.set_paths()
    if args.model:
        config.model = args.model

    util.set_img_format()
    model_module = util.get_model_class_instance()
    model = model_module.load()

    classes_in_keras_format = util.get_classes_in_keras_format()

    predict(args.path)

    if args.execution_time:
        toc = time.clock()
        print('Time: %s' % (toc - tic))
