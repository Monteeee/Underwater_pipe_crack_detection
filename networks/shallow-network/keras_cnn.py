from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import numpy as np

np.random.seed(1337)  # for reproducibility

img_width, img_height = 80, 80

train_data_dir = "C:/Users/yipai.du/Downloads/large data-20190313T201506Z-001 - Keras_format/sorted/train"
validation_data_dir = "C:/Users/yipai.du/Downloads/large data-20190313T201506Z-001 - Keras_format/sorted/valid"

steps_per_epoch = 50
epochs = 100
batch_size = 16

if K.image_data_format() == "channels_first":
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

model = Sequential()
model.add(Conv2D(64, (3, 3), input_shape=input_shape))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(256))
model.add(Activation("relu"))
model.add(Dropout(0.1))
model.add(Dense(4))
model.add(Activation("softmax"))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255, horizontal_flip=True, vertical_flip=True
)

valid_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir, target_size=(img_width, img_height), batch_size=batch_size
)

validation_generator = valid_datagen.flow_from_directory(
    validation_data_dir, target_size=(img_width, img_height), batch_size=batch_size
)

model.fit_generator(
    train_generator,
    epochs=epochs,
    steps_per_epoch=steps_per_epoch,
    validation_steps=1,
    validation_data=validation_generator,
)

model.save("my.h5")
