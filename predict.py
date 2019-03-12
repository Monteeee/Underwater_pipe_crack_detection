#!/usr/bin/python
# -*- coding: UTF-8 -*-

import utils.model as model
import numpy as np
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss, roc_auc_score, f1_score, roc_curve
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix
from sklearn.metrics import auc
from sklearn.preprocessing import label_binarize
from sklearn.metrics import classification_report

from itertools import cycle

if __name__ == "__main__":
    # load parameters
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--savePath',
        type=str,
        default='C:/Users/wyxkd4/Desktop/Underwater_pipe_crack_detection/model',
        help='The path to save the model and log')
    parser.add_argument(
        '--dataPath',
        type=str,
        default='C:/Users/wyxkd4/Desktop/Underwater_pipe_crack_detection/data/data_DAM_ANO_CON/sorted',
        help='The path to data')
    args = parser.parse_args()

    savePath = args.savePath
    dataPath = args.dataPath
    model_weight_path = savePath + '/mobile/model_weight.h5'
    # testModel = model.NetVGG16FC(80, 4)
    # testModel.summary()
    n_classes = 3
    testModel = model.NetMobileFC(
        save_path=savePath,
        image_size=128,
        class_num=n_classes,
        alpha=0.5)

    testModel.model.summary()
    testModel.model.load_weights(model_weight_path)

    print("=>=>=>=>=> testing =>=>=>=>=>")
    y_pred, y_true, y_score = testModel.test_model(data_path=dataPath, batch_size=16)

    print("test Acc", round(accuracy_score(y_true, y_pred), 4))
    y_one_hot = label_binarize(y_true, np.arange(n_classes))
    # print("test LogLoss", round(log_loss(y_true, y_pred), 4))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_one_hot[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # fpr, tpr, thresholds = roc_curve(y_one_hot.ravel(), y_score.ravel())
    lw = 2
    plt.figure()

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw, label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.show()

    print("test Confusion Matrix \n", confusion_matrix(y_true, y_pred))
    target_names = ['class 0', 'class 1', 'class 2']
    print("test classification report \n"classification_report(y_true, y_pred, target_names=target_names))

