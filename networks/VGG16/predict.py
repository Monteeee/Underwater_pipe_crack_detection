import argparse
import glob
import os
import pickle
import time
from itertools import cycle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

import config
import util


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", dest="path", help="Path to image",
        default='C:/Users/yipai.du/Downloads/large_Keras/sorted/test/',
        # default='C:/Users/yipai.du/Downloads/small/test/',
        type=str
    )
    parser.add_argument(
        "--accuracy", action="store_true", help="To print accuracy score"
    )
    parser.add_argument("--plot_confusion_matrix", action="store_true")
    parser.add_argument("--execution_time", action="store_true")
    parser.add_argument("--store_activations", action="store_true")
    parser.add_argument("--novelty_detection", action="store_true")
    parser.add_argument(
        "--model",
        type=str,
        help="Base model architecture",
        choices=[
            config.MODEL_RESNET50,
            config.MODEL_RESNET152,
            config.MODEL_INCEPTION_V3,
            config.MODEL_VGG16,
        ]
    )
    parser.add_argument("--data_dir", help="Path to data train directory")
    parser.add_argument(
        "--batch_size",
        default=500,
        type=int,
        help="How many files to predict on at once",
    )
    args = parser.parse_args()
    return args


def get_files(path):
    if os.path.isdir(path):
        print(path)
        path1 = path
        path = path + "1\\"
        print(path)
        files = glob.glob(path + "*.png")
        path = path1 + "0\\"
        files1 = glob.glob(path + "*.png")
        files = files + files1
        path = path1 + "2\\"
        files1 = glob.glob(path + "*.png")
        files = files + files1
        path = path1 + "3\\"
        files1 = glob.glob(path + "*.png")
        files = files + files1
    elif path.find("*") > 0:
        files = glob.glob(path)
    else:
        files = [path]

    if not len(files):
        print("No images found by the given path")
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
        x = x * 1.0 / 255
        # print("=== after ===")
        # print(x[50, 50, :])
        try:
            image_class = i.split(os.sep)[-2]
            keras_class = int(classes_in_keras_format[image_class])
            y_true.append(keras_class)
        except Exception:
            # print(os.path.split(i))
            y_true.append(os.path.split(i)[1])
        inputs.append(x)

    return y_true, inputs


def predict(path):
    n_classes = 4
    files = get_files(path)
    n_files = len(files)
    print("Found {} files".format(n_files))

    if args.novelty_detection:
        activation_function = util.get_activation_function(
            model, model_module.noveltyDetectionLayerName
        )
        novelty_detection_clf = joblib.load(config.get_novelty_detection_model_path())

    y_trues = []
    predictions = np.zeros(shape=(n_files,))
    scores = np.zeros(shape=(n_files, n_classes))
    nb_batch = int(np.ceil(n_files / float(args.batch_size)))
    for n in range(0, nb_batch):
        print("Batch {}".format(n))
        n_from = n * args.batch_size
        n_to = min(args.batch_size * (n + 1), n_files)

        y_true, inputs = get_inputs_and_trues(files[n_from:n_to])
        y_trues += y_true

        if args.store_activations:
            util.save_activations(
                model,
                inputs,
                files[n_from:n_to],
                model_module.noveltyDetectionLayerName,
                n,
            )

        if args.novelty_detection:
            activations = util.get_activations(activation_function, [inputs[0]])
            nd_preds = novelty_detection_clf.predict(activations)[0]
            print(novelty_detection_clf.__classes[nd_preds])

        if not args.store_activations:
            # Warm up the model
            if n == 0:
                print("Warming up the model")
                start = time.clock()
                model.predict(np.array([inputs[0]]))
                end = time.clock()
                print("Warming up took {} s".format(end - start))

            # Make predictions
            start = time.clock()
            out = model.predict(np.array(inputs))
            end = time.clock()
            predictions[n_from:n_to] = np.argmax(out, axis=1)
            scores[n_from:n_to, :] = out
            print("Prediction on batch {} took: {}".format(n, end - start))

    if not args.store_activations:
        t = np.zeros_like(predictions)
        for i in range(len(y_trues)):
            string = y_trues[i]
            t[i] = int(string[11])
        y_trues = t
        print("test Confusion Matrix \n")
        print(confusion_matrix(y_trues, predictions))
        target_names = ["clean", "annode", "connection", "damage"]
        print(
            "test classification report \n",
            classification_report(
                y_trues, predictions, target_names=target_names, digits=5
            ),
        )
        y_one_hot = label_binarize(y_trues, np.arange(n_classes))
        # print("test LogLoss", round(log_loss(y_true, y_pred), 4))
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_one_hot[:, i], scores[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        fpr["micro"], tpr["micro"], _ = roc_curve(
            y_one_hot.ravel(), scores.ravel()
        )
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        with open("./vgg_large_micro_roc.pkl", "wb") as fP:
            pickle.dump([fpr["micro"], tpr["micro"]], fP, pickle.HIGHEST_PROTOCOL)
            fP.close()

        lw = 2
        plt.figure()

        colors = cycle(["aqua", "darkorange", "cornflowerblue", "navy"])
        class_name = ["clean", "anode", "connection", "damage"]

        for i, color in zip(range(n_classes), colors):
            plt.plot(
                fpr[i],
                tpr[i],
                color=color,
                lw=lw,
                label="ROC curve of class {0} (area = {1:0.5f})".format(
                    class_name[i], roc_auc[i]
                ),
            )

        plt.plot(
            fpr["micro"],
            tpr["micro"],
            label="micro-average ROC curve (area = {0:0.5f})"
            "".format(roc_auc["micro"]),
            color="navy",
            linestyle=":",
            linewidth=2,
        )

        plt.plot([0, 1], [0, 1], "k--", lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic to classes")
        plt.legend(loc="lower right")
        plt.show()
        plt.savefig('roc.png')
        if args.accuracy:
            print(
                "Accuracy {}".format(accuracy_score(y_true=y_trues, y_pred=predictions))
            )


if __name__ == "__main__":
    tic = time.clock()

    args = parse_args()
    print("=" * 50)
    print("Called with args:")
    print(args)

    if args.data_dir:
        config.data_dir = args.data_dir
        config.set_paths()
    config.model = config.MODEL_VGG16

    util.set_img_format()
    model_module = util.get_model_class_instance()
    model = model_module.load()

    classes_in_keras_format = util.get_classes_in_keras_format()

    predict(args.path)

    if args.execution_time:
        toc = time.clock()
        print("Time: %s" % (toc - tic))
