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
        default='C:/Users/wyxkd4/Desktop/Underwater_pipe_crack_detection/data/data_DAM_ANO_CON/sorted',
        help='The path to data')
    args = parser.parse_args()

    savePath = args.savePath
    dataPath = args.dataPath

    # testModel = model.NetVGG16FC(80, 4)
    # testModel.summary()
    testModel = model.NetMobileFC(
        save_path=savePath,
        image_size=128,
        class_num=3,
        alpha=0.5)

    testModel.model.summary()
    testModel.train_model(data_path=dataPath, epochs=100, batch_size=32)

    print("=>=>=>=>=> testing =>=>=>=>=>")
    y_pred, y_true = testModel.test_model(data_path=dataPath, batch_size=16)
