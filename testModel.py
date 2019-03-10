#!/usr/bin/python
# -*- coding: UTF-8 -*-

import utils.model as model
import numpy as np
import argparse
from sklearn.metrics import log_loss, roc_auc_score, f1_score
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix

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
        default='C:/Users/wyxkd4/Desktop/Underwater_pipe_crack_detection/data/data_new_dam_pipe_all/sorted',
        help='The path to data')
    args = parser.parse_args()

    savePath = args.savePath
    dataPath = args.dataPath

    # testModel = model.NetVGG16FC(80, 4)
    # testModel.summary()
    testModel = model.NetMobileFC(
        save_path=savePath,
        image_size=128,
        class_num=2,
        alpha=0.5)

    testModel.model.summary()
    testModel.train_model(data_path=dataPath, epochs=15, batch_size=16)

    print("=>=>=>=>=> testing =>=>=>=>=>")
    y_pred, y_true = testModel.test_model(data_path=dataPath, batch_size=16)

    print("test Acc", round(accuracy_score(y_true, y_pred), 4))
    print("test LogLoss", round(log_loss(y_true, y_pred), 4))
    print("test AUC", round(roc_auc_score(y_true, y_pred), 4))
    print("test Recall", round(recall_score(y_true, y_pred), 4))
    print("test F1 Score", round(f1_score(y_true, y_pred), 4))
    print("test Confusion Matrix \n", confusion_matrix(y_true, y_pred))
