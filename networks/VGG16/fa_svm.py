from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.models import Model
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
from sklearn.svm import SVC
from sklearn import linear_model
from sklearn.kernel_approximation import RBFSampler, AdditiveChi2Sampler
import pickle


def get_feature_and_label(model, generator, batch_number):
    b = batch_number
    features = model.predict(generator[b][0])
    f_shape = features[0].shape
    total_len = np.prod(f_shape)
    vect_feature = list()
    for feature_i in features:
        vect_feature.append(np.reshape(feature_i, total_len))

    vect_feature = np.array(vect_feature)

    label = list()
    for i in range(len(generator[b][1])):
        label.append(generator[b][1][i])

    label = np.array(label)

    return vect_feature, label


if __name__ == '__main__':
    
    model = VGG16(weights='imagenet', include_top=False)
    #base_model = VGG19(weights='imagenet')
    #model = Model(inputs=base_model.input, outputs=base_model.get_layer('block4_pool').output)

    cwd = os.getcwd()

    img_width, img_height = 80, 80
    batch_size = 100

    train_datagen = ImageDataGenerator(
        #rescale=1. / 255,
        #rotation_range=45,
        #shear_range=0.2,
        #zoom_range=0.2,
        width_shift_range=0.1,
        height_shift_range=0.1,
        #vertical_flip=True,
        #horizontal_flip=True
        )

    #valid_datagen = ImageDataGenerator(rescale=1./255)
    valid_datagen = ImageDataGenerator()

    train_data_dir = cwd + '\data\sorted/train'
    valid_data_dir = cwd + '\data\sorted/valid'
    train_data_n = 3700
    valid_data_n = 1000
    
    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary')

    valid_generator = valid_datagen.flow_from_directory(
        valid_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary')

    clf = linear_model.SGDClassifier(class_weight={0:0.4, 1:0.6})

    classes_ = np.array([0, 1])

    #rbf_feature = RBFSampler(gamma=1, random_state=1)
    chi_feature = AdditiveChi2Sampler()
    
    for i in range(train_data_n // batch_size):
        print("======= trainning! =======")
        print("batch No." + str(i) )
        feature_train, label_train = get_feature_and_label(model, train_generator, i)
        #feature_train = rbf_feature.fit_transform(feature_train_o)
        feature_train = chi_feature.fit_transform(feature_train, label_train)

        clf.partial_fit(feature_train, label_train, classes_)

        if np.remainder(i, 5) == 0:
            valid_score = []
            right_0 = []
            right_1 = []
            #right_all = []
            number_0 = []
            number_1 = []
            print("======= predicting! =======")
            for j in range(valid_data_n // batch_size):
                feature_valid, label_valid = get_feature_and_label(model, valid_generator, j)
                #feature_valid = rbf_feature.fit_transform(feature_valid_o)
                feature_valid = chi_feature.fit_transform(feature_valid, label_valid)
                
                valid_score.append(clf.score(feature_valid, label_valid))
                pred_label = clf.predict(feature_valid)
                #print(pred_label)
                #print(label_valid)
                n0 = np.sum(np.where(label_valid==0, 1, 0) )
                n1 = np.sum(np.where(label_valid==1, 1, 0) )
                number_0.append(n0)
                number_1.append(n1)
                right_0.append( (len(pred_label) - np.sum( np.where(label_valid == 0, pred_label, 1) ) )  )
                right_1.append( np.sum( np.where(label_valid == 1, pred_label, 0) ) )
                #right_all.append( np.sum( np.where(label_valid==pred_label, 1, 0) ) )
                
                
            print(valid_score)
            print(right_0)
            print(right_1)
            print( "class 0 acc: " + str( np.sum(right_0) /  np.sum(number_0) ) + "; class 1 acc: " + str( np.sum(right_1) /  np.sum(number_1))  )
            print( "mean_acc: " + str(np.mean(valid_score)))

    filename = 'svm_model.sav'
    pickle.dump(clf, open(filename, 'wb'))
