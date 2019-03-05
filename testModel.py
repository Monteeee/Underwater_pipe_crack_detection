#!/usr/bin/python
# -*- coding: UTF-8 -*-

import utils.model as model
import numpy as np
import argparse

if __name__ == "__main__":
    # load parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--savePath', type=str, default='/home/jiajuns/Documents/Some Github Repo/UPCD_for_ICRA/Underwater_pipe_crack_detection/model', \
        help='The path to save the model')
    parser.add_argument('--dataPath', type=str, default='/home/jiajuns/Documents/Some Github Repo/UPCD_for_ICRA/Underwater_pipe_crack_detection/data/data_ALL', \
        help='The path to data')
    args = parser.parse_args()

    savePath = args.savePath
    dataPath = args.dataPath

    # testModel = model.NetVGG16FC(80, 4)
    # testModel.summary()
    testModel = model.NetMobileFC(save_path=savePath, image_size=128, class_num=4, alpha=0.5)
    testModel.model.summary()
    testModel.train_model(data_path=dataPath, epochs=10, batchsize=50)