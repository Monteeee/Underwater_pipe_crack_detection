#!/usr/bin/python
# -*- coding: UTF-8 -*-

'''
@Description:  The util for keras model
@Author: Wenjie Yin
@Date: 2019-02-18 11:14:26
@LastEditTime: 2019-02-25 19:57:53
@LastEditors: Wenjie Yin
'''

import numpy as np

from abc import ABC, abstractmethod
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dense, GlobalAveragePooling2D, Dropout, Flatten, BatchNormalization
from keras.layers.normalization import BatchNormalization
from keras.applications.mobilenet import MobileNet
from keras.applications.vgg16 import VGG16
from keras.optimizers import Adam
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau

class NetModel(ABC):
    '''
    @description: 
        The class of neural network model
    '''
    def __init__(self, image_size:int, class_num:int):
        # the input image size
        self._image_size = image_size
        # The number of classes
        self._class_num = class_num
        self._model = None

        # set the channel axis and input shape
        if K.image_dim_ordering() == "th":
            self._channel_axis = 1
            self._input_shape = (3, image_size, image_size)
        elif K.image_dim_ordering() == "tf":
            self._channel_axis = -1
            self._input_shape = (image_size, image_size, 3)

    @abstractmethod
    def init_model(self):
        '''
        @description: 
            initialize the model 
        @return: 
            the model
        '''
        ...

    @property
    def image_size(self):
        return self._image_size

    @property
    def class_num(self):
        return self._class_num
     
    @property
    def input_shape(self):
        return self._input_shape  

    @property
    def model(self):
        return self._model    
    
class NetMobileFC(NetModel):
    '''
    @description: 
        The class of mobilenet
    @param {
        image_size: the size of image
        alpha: the parameter of mobilenet
    } 
    @return: 
        the model
    '''
    def __init__(self, image_size:int, class_num:int, alpha:float):
        super().__init__(image_size, class_num)
        self._alpha = alpha
        self._model = self.init_model()

    @property
    def alpha(self):
        return self._alpha
    
    def init_model(self):
        # Create Model
        input_tensor = Input(shape=self.input_shape)   
        batch_normalization_layer_1 = BatchNormalization()(input_tensor)
        base_mobilenet_model = MobileNet(input_shape=self.input_shape, \
            alpha=0.5, \
            include_top=False, \
            weights='imagenet')(batch_normalization_layer_1)

        batch_normalization_layer_2 = BatchNormalization()(base_mobilenet_model)
        global_average_pooling_2D_1 = GlobalAveragePooling2D()(batch_normalization_layer_2)
        dropout_1 = Dropout(0.2)(global_average_pooling_2D_1)
        dense_1 = Dense(128, activation='relu')(dropout_1)
        output_layer_1 = Dense(self.class_num)(dense_1)

        model = Model(inputs=input_tensor, outputs=output_layer_1)

        model.compile(optimizer='adam',
                loss='mse',
                metrics=['accuracy'])

        return model

class NetVGG16FC(NetModel):
    '''
    @description: 
        The class of VGG16
    @param {
        image_size: the size of image
    } 
    @return: 
        the model
    '''   
    def __init__(self, image_size:int, class_num:int):
        super().__init__(image_size, class_num)
        self._model = self.init_model()      

    def init_model(self):
        # Create Model
        input_tensor = Input(shape=self.input_shape)   
        batch_normalization_layer_1 = BatchNormalization()(input_tensor)
        base_vgg16_model = VGG16(include_top=False, \
            weights='imagenet', \
            input_shape=self.input_shape)(batch_normalization_layer_1)
        batch_normalization_layer_2 = BatchNormalization()(base_vgg16_model)
        global_average_pooling_2D_1 = GlobalAveragePooling2D()(batch_normalization_layer_2)
        dropout_1 = Dropout(0.2)(global_average_pooling_2D_1)
        dense_1 = Dense(128, activation='relu')(dropout_1)
        output_layer_1 = Dense(self.class_num)(dense_1)

        model = Model(inputs=input_tensor, outputs=output_layer_1)

        model.compile(optimizer='adam',
                loss='mse',
                metrics=['accuracy'])
        
        return model