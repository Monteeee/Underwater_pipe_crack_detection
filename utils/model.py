#!/usr/bin/python
# -*- coding: UTF-8 -*-

'''
@Description:  The util for keras model
@Author: Wenjie Yin
@Date: 2019-02-18 11:14:26
@LastEditTime: 2019-03-03 15:38:29
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
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau, TensorBoard
from keras.preprocessing.image import ImageDataGenerator

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
    
    @abstractmethod
    def init_model(self):
        '''
        @description: 
            initialize the model 
        @return: 
            the model
        '''
        ...
    
    def get_train_datagen(self, path:str, batch_size:int=20):
        train_datagen = ImageDataGenerator(rescale=1./255, \
            rotation_range=30., \
            shear_range=0.2, \
            zoom_range=0.2, \
            horizontal_flip=True)
        train_generator = train_datagen.flow_from_directory(
            path,
            target_size = (self.image_size, self.image_size),
            batch_size = batch_size,
            class_mode = 'categorical',)
            # classes = )

        return train_generator
    
    def get_validation_datagen(self, path:str, batch_size:int=128):
        train_datagen = ImageDataGenerator(rescale=1./255)
        train_generator = train_datagen.flow_from_directory(
            path,
            target_size = (self.image_size, self.image_size),
            batch_size = batch_size,
            class_mode = 'categorical',)
            # classes = )

        return train_generator
        
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
    def __init__(self, save_path:str, image_size:int, class_num:int, alpha:float):
        super().__init__(image_size, class_num)
        self._save_path = save_path
        self._alpha = alpha
        self._model = self.init_model()

    @property
    def alpha(self):
        return self._alpha
    
    @property
    def save_path(self):
        return self._save_path

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
                loss='categorical_crossentropy',
                metrics=['accuracy'])

        return model
        
    def callbacks(self):
        # Save the model after every epoch.
        self.checkpoint = ModelCheckpoint(self.save_path + "/mobile/model_weight.h5", monitor='val_loss', verbose=1, \
            save_best_only=True, mode='min', save_weights_only = True)

        # Reduce learning rate when a metric has stopped improving.
        self.reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=10, verbose=1, \
            mode='auto', epsilon=0.0001, cooldown=5, min_lr=0.0001)
        
        # Stop training when a monitored quantity has stopped improving.
        self.early = EarlyStopping(monitor="val_loss", mode="min", patience=30) 
        
        # TensorBoard basic visualizations.
        self.tbCallBack = TensorBoard(log_dir=self.save_path + '/mobile/logs', histogram_freq=0, write_graph=True, \
            write_grads=True, write_images=True, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
        
        callbacks_list = [self.checkpoint, self.early, self.reduceLROnPlat, self.tbCallBack]
        
        return callbacks_list    

    def train_model(self, data_path:str, epochs:int=100, batchsize=100):
        train_data = self.get_train_datagen(data_path+'/train', batch_size=batchsize)
        validation_data = self.get_validation_datagen(data_path+'/valid', )
        self.callbacks_list = self.callbacks()

        self.model.fit_generator(train_data, steps_per_epoch=2000, epochs=epochs, validation_data=validation_data, validation_steps=2000, callbacks=self.callbacks_list)


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
                loss='categorical_crossentropy',
                metrics=['accuracy'])
        
        return model