from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

img_width, img_height = 80, 80

test_data_dir = 'image/test'

batch_size = 64
test_datagen = ImageDataGenerator(rescale=1. / 255)

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    shuffle=False)

model = load_model('my.h5')

#Confution Matrix and Classification Report
Y_pred = model.predict_generator(test_generator)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
C = confusion_matrix(test_generator.classes, y_pred)
C_n = C.astype('float') / C.sum(axis=1)[:, np.newaxis]
print(C_n)
# print('Classification Report')
# target_names = ['0', '1', '2', '3']
# print(
#     classification_report(
#         test_generator.classes, y_pred, target_names=target_names))
