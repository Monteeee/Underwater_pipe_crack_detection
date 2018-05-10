from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model

img_width, img_height = 80, 80

test0_data_dir = 'data/test0'
test1_data_dir = 'data/test1'

batch_size = 32
test_datagen = ImageDataGenerator(rescale=1. / 255)

test0_generator = test_datagen.flow_from_directory(
    test0_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

test1_generator = test_datagen.flow_from_directory(
    test1_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

model = load_model('my.h5')

result = model.evaluate_generator(test0_generator)
print(model.predict_generator(test0_generator, verbose=1))
#print(result)
result = model.evaluate_generator(test1_generator)
print(model.predict_generator(test1_generator, verbose=1))
#print(result)
