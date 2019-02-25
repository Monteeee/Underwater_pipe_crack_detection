#!/usr/bin/python
# -*- coding: UTF-8 -*-

import utils.model as model
import numpy as np
import argparse

if __name__ == "__main__":
    # load parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--parameter', type=int, default=1, help='The Example of input parameter')
    args = parser.parse_args()

    parameter = args.parameter

    testModel = model.NetVGG16FC(80, 4)
    # testModel.summary()
    testModel.model.summary()