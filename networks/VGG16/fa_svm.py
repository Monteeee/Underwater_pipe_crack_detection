from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.models import Model
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
from sklearn.svm import SVC
from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn.kernel_approximation import RBFSampler, AdditiveChi2Sampler, Nystroem, SkewedChi2Sampler
from sklearn.preprocessing import normalize
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
import pickle
import glob
from copy import deepcopy


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
    batch_size = 200

    train_datagen = ImageDataGenerator(
        #rescale=1. / 255clear,
        #rotation_range=45,
        #shear_range=0.1,
        #zoom_range=0.3,
        width_shift_range=0.2,
        height_shift_range=0.2,
        vertical_flip=True,
        horizontal_flip=True
        )

    #valid_datagen = ImageDataGenerator(rescale=1./255)
    valid_datagen = ImageDataGenerator()

    train_data_dir = cwd + '/data/sorted/train'
    valid_data_dir = cwd + '/data/sorted/valid'
    test_data_dir = cwd + '/data/sorted/test'
    
    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='sparse'
        )

    train_data_n = len(os.listdir(train_data_dir + '/1')) + len(os.listdir(train_data_dir + '/0')) + len(os.listdir(train_data_dir + '/2'))

    chi_feature = AdditiveChi2Sampler()

    clf = SGDClassifier(class_weight={0:1.0, 1:1.2, 2:1.0})

    classes_ = np.array([0, 1, 2])

    rbf_feature = RBFSampler(gamma=4.0, n_components=3000)
    #rbf_feature = Nystroem(n_components=100, gamma=1.0, random_state=1)

    """
    feature_train_stack = np.zeros((100, 2048)) - 1
    label_train_stack = np.zeros((100, 1)) - 1
    for i in range(train_data_n // batch_size):
    #for i in range(2):
        print("======= data reading! =======")
        print("batch No." + str(i) )
        feature_train, label_train = get_feature_and_label(model, train_generator, i)
        feature_train  = normalize(feature_train)
        #print(feature_train.shape)
        
        #feature_train = rbf_feature.fit_transform(feature_train)
        #feature_train = chi_feature.fit_transform(feature_train, label_train)
        print(feature_train.shape)
        print(label_train.shape)

        if i == 0:
            feature_train_stack = feature_train
            label_train_stack = label_train
        else:
            feature_train_stack = np.vstack((feature_train_stack, feature_train))
            label_train_stack = np.hstack((label_train_stack, label_train))

        print(feature_train_stack.shape)
        print(label_train_stack.shape)
        #clf.partial_fit(feature_train, label_train, classes_)
        '''
        if False and (np.remainder(i, 5) == 0 or i == train_data_n // batch_size - 1):
            valid_score = []
            right_0 = []
            right_1 = []
            #right_all = []
            number_0 = []
            number_1 = []
            print("======= // predicting! // =======")
            for j in range(valid_data_n // batch_size):
                feature_valid, label_valid = get_feature_and_label(model, valid_generator, j)
                feature_valid  = normalize(feature_valid)
                
                #feature_valid = rbf_feature.fit_transform(feature_valid)
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
            '''
    
    print("saving data!")
    filename = 'svm_data5.pkl'
    pickle.dump([feature_train_stack, label_train_stack], open(filename, 'wb'))
    print("data saved!")
    """
    
    pkl_file = open('svm_data5.pkl', 'rb')
    feature_train_stack, label_train_stack = pickle.load(pkl_file)
    #feature_train_stack = chi_feature.fit_transform(feature_train_stack, label_train_stack)
    feature_train_stack = rbf_feature.fit_transform(feature_train_stack)
    
    print("====== start training ! =======")
    
    #clf.partial_fit(feature_train_stack, label_train_stack, classes_)

    ada_clf = AdaBoostClassifier(SVC(kernel='linear', probability=True,
                                 class_weight={0:1.0, 1:1.0, 2:1.0}),
                                 algorithm="SAMME.R",
                                 n_estimators=1)
    
    c2_index = np.where(label_train_stack==2)
    c1_index = np.where(label_train_stack==1)
    c0_index = np.where(label_train_stack==0)

    c2_feature = np.squeeze(feature_train_stack[c2_index, :])
    c1_feature = np.squeeze(feature_train_stack[c1_index, :])
    c0_feature = np.squeeze(feature_train_stack[c0_index, :])

    oneclass_svm1 = svm.OneClassSVM(nu=0.1, kernel='linear')
    oneclass_svm2 = svm.OneClassSVM(nu=0.1, kernel='linear')
    oneclass_svm3 = svm.OneClassSVM(nu=0.05, kernel='linear')

    oneclass_svm1.fit(c0_feature)
    
    y_pred_train0 = oneclass_svm1.predict(c0_feature)
    y_pred_test1 = oneclass_svm1.predict(c1_feature)
    y_pred_test2 = oneclass_svm1.predict(c2_feature)

    r_error_train = y_pred_train0[y_pred_train0 == -1].size / y_pred_train0.size
    r_error_test1 = y_pred_test1[y_pred_test1 == 1].size / y_pred_test1.size
    r_error_test2 = y_pred_test2[y_pred_test2 == 1].size / y_pred_test2.size

    print("train error rate: " + str(r_error_train) )
    print("test1 error rate: " + str(r_error_test1) )
    print("test2 error rate: " + str(r_error_test2) )

    # ====================================================
    oneclass_svm2.fit(c1_feature)

    y_pred_train = oneclass_svm2.predict(c1_feature)
    y_pred_test10 = oneclass_svm2.predict(c0_feature)
    y_pred_test2 = oneclass_svm2.predict(c2_feature)

    r_error_train = y_pred_train[y_pred_train == -1].size / y_pred_train.size
    r_error_test1 = y_pred_test10[y_pred_test10 == 1].size / y_pred_test10.size
    r_error_test2 = y_pred_test2[y_pred_test2 == 1].size / y_pred_test2.size

    print("train error rate: " + str(r_error_train))
    print("test1 error rate: " + str(r_error_test1))
    print("test2 error rate: " + str(r_error_test2))

    # ====================================================
    oneclass_svm3.fit(c2_feature)

    y_pred_train = oneclass_svm3.predict(c2_feature)
    y_pred_test100 = oneclass_svm3.predict(c0_feature)
    y_pred_test2 = oneclass_svm3.predict(c1_feature)

    r_error_train = y_pred_train[y_pred_train == -1].size / y_pred_train.size
    r_error_test1 = y_pred_test100[y_pred_test100 == 1].size / y_pred_test100.size
    r_error_test2 = y_pred_test2[y_pred_test2 == 1].size / y_pred_test2.size

    print("train error rate: " + str(r_error_train))
    print("test1 error rate: " + str(r_error_test1))
    print("test2 error rate: " + str(r_error_test2))

    y_pred_test0 = np.bitwise_and((y_pred_test10 == -1), (y_pred_test100 == -1))
    print( "rate: " + str(np.sum(y_pred_test0) / y_pred_test0.size) )

    #print(c0_feature.shape)

    class_0_n = np.size(c0_feature, 0)
    class_1_n = np.size(c1_feature, 0)
    class_2_n = np.size(c2_feature, 0)

    clf_list = list()
    print(class_0_n, class_1_n, class_2_n)

    """
    # this is for a large class and a small class classification.
    for i in range(class_0_n // class_1_n):
        print("training classifier " + str(i) + " / " + str(class_0_n // class_1_n))
        ada_clf = SVC(kernel='linear', probability=True,
                      class_weight={0:1.0, 1:1.0})
        labels = np.hstack((np.zeros((class_1_n)), np.ones(class_1_n)))
        print(labels.shape)
        features = np.vstack((np.squeeze(c0_feature[i*class_1_n:(i+1)*class_1_n,:]), c1_feature))
        print(features.shape)
        ada_clf.fit(features, labels)
        clf_list.append(deepcopy(ada_clf))
        """

    #pkl_file = open('svm_model.sav', 'rb')
    #clf = pickle.load(pkl_file)

    #print("cross-validation-check")
    #scores = cross_val_score(ada_clf, feature_train_stack, label_train_stack, cv=5)
    #print(scores)
        
    #filename = 'svm_models2.pkl'
    #pickle.dump(clf_list, open(filename, 'wb'))

    #ada_clf.fit(feature_train_stack, label_train_stack)
    #clf = ada_clf
    
    valid_score = []
    right_0 = []
    right_1 = []
    right_2 = []
    #right_all = []
    number_0 = []
    number_1 = []
    number_2 = []

    
    print("======= predicting! =======")

    for input_dir in [valid_data_dir, test_data_dir]:
        valid_generator = valid_datagen.flow_from_directory(
                                                    input_dir,
                                                    target_size=(img_width, img_height),
                                                    batch_size=batch_size,
                                                    class_mode='sparse'
                                                    )

        valid_data_n = len(os.listdir(input_dir + '/1')) + len(os.listdir(input_dir + '/0')) + len(os.listdir(input_dir + '/2'))
        
        for j in range(valid_data_n // batch_size + 1):
            print("validation batch " + str(j))
            feature_valid, label_valid = get_feature_and_label(model, valid_generator, j)
            feature_valid  = normalize(feature_valid)

            feature_valid = rbf_feature.fit_transform(feature_valid)
            #feature_valid = chi_feature.fit_transform(feature_valid, label_valid)

            if j == 0:
                feature_valid_stack = feature_valid
                label_valid_stack = label_valid
            else:
                feature_valid_stack = np.vstack((feature_valid_stack, feature_valid))
                label_valid_stack = np.hstack((label_valid_stack, label_valid))
            
            #valid_score.append(clf.score(feature_valid, label_valid))
            #pred_label = clf.predict(feature_valid)
            #print(pred_label)
            #print(label_valid)

            #n0 = np.sum(np.where(label_valid==0, 1, 0) )
            #n1 = np.sum(np.where(label_valid==1, 1, 0) )
            #n2 = np.sum(np.where(label_valid==2, 1, 0) )
            
            #number_0.append(n0)
            #number_1.append(n1)
            #number_2.append(n2)

            #right_0.append( np.sum(np.multiply(np.where(label_valid==0, 1, 0), np.where(pred_label==0, 1, 0)) ))
            #right_1.append( np.sum(np.multiply(np.where(label_valid==1, 1, 0), np.where(pred_label==1, 1, 0)) ))
            #right_2.append( np.sum(np.multiply(np.where(label_valid==2, 1, 0), np.where(pred_label==2, 1, 0)) ))

            #right_0.append( (len(pred_label) - np.sum( np.where(label_valid == 0, pred_label, 1) ) )  )
            #right_1.append( np.sum( np.where(label_valid == 1, pred_label, 0) ) )
            #right_all.append( np.sum( np.where(label_valid==pred_label, 1, 0) ) )
                
        #if input_dir == valid_data_dir:
            #pickle.dump([feature_valid_stack, label_valid_stack], open('svm_valid5.pkl', 'wb'))
            #print("saved!")
        #else:
            #pickle.dump([feature_valid_stack, label_valid_stack], open('svm_test5.pkl', 'wb'))
            #print("saved!")

        c2_index = np.where(label_valid_stack==2)
        c1_index = np.where(label_valid_stack==1)
        c0_index = np.where(label_valid_stack==0)

        c2_feature = np.squeeze(feature_valid_stack[c2_index, :])
        c1_feature = np.squeeze(feature_valid_stack[c1_index, :])
        c0_feature = np.squeeze(feature_valid_stack[c0_index, :])

        y_pred_c1 = oneclass_svm.predict(c1_feature)
        y_pred_c2 = oneclass_svm.predict(c2_feature)
        y_pred_c0 = oneclass_svm.predict(c0_feature)

        r_error_c0 = y_pred_c0[y_pred_c0 == 1].size / y_pred_c0.size
        r_error_c1 = y_pred_c1[y_pred_c1 == 1].size / y_pred_c1.size
        r_error_c2 = y_pred_c2[y_pred_c2 == 1].size / y_pred_c2.size

        print("c0 rate: " + str(r_error_c0) )
        print("c1 rate: " + str(r_error_c1) )
        print("c2 rate: " + str(r_error_c2) )

    """            
    print(valid_score)
    print(right_0, right_1, right_2)
    print( "class 0 acc: " + str( np.sum(right_0) /  np.sum(number_0) ) + "; class 1 acc: " + str( np.sum(right_1) /  np.sum(number_1))  \
           + "; class 2 acc: " + str( np.sum(right_2) /  np.sum(number_2)) )
    print( "mean_acc: " + str(np.mean(valid_score)))
    """
    

    """
    # this is for large class and small class classification (damage vs others)
    all_votes = np.zeros((class_0_n // class_1_n, valid_data_n))
    for i in range(class_0_n // class_1_n):
    #for i in range(2):
        clf = clf_list[i]
        for j in range(valid_data_n // batch_size + 1):
            feature_valid, label_valid = get_feature_and_label(model, valid_generator, j)
            feature_valid  = normalize(feature_valid)
            
            feature_valid = rbf_feature.fit_transform(feature_valid)
            #feature_valid = chi_feature.fit_transform(feature_valid)
            
            valid_score.append(clf.score(feature_valid, label_valid))
            pred_label = clf.predict(feature_valid)
            if i == 0:
                if j == 0:
                    pred_label_all = pred_label
                    label_valid_stack = label_valid
                else:
                    pred_label_all = np.hstack((pred_label_all, pred_label))
                    label_valid_stack = np.hstack((label_valid_stack, label_valid))
                #print(label_valid_stack.shape)
            #print(pred_label)
            #print(label_valid)
            n0 = np.sum(np.where(label_valid==0, 1, 0) )
            n1 = np.sum(np.where(label_valid==1, 1, 0) )
            number_0.append(n0)
            number_1.append(n1)
            right_0.append( (len(pred_label) - np.sum( np.where(label_valid == 0, pred_label, 1) ) )  )
            right_1.append( np.sum( np.where(label_valid == 1, pred_label, 0) ) )
            #right_all.append( np.sum( np.where(label_valid==pred_label, 1, 0) ) )
        #print(pred_label_all.shape)
        all_votes[i, :] = pred_label_all[:]
        
        print(valid_score)
        print(right_0)
        print(right_1)
        print( "class 0 acc: " + str( np.sum(right_0) /  np.sum(number_0) ) + "; class 1 acc: " + str( np.sum(right_1) /  np.sum(number_1))  )
        print( "mean_acc: " + str(np.mean(valid_score)))
    """
    """
    votes = np.sum(all_votes, axis=0)
    print((class_0_n // class_1_n) / 2)
    vote_pred = np.where(votes > (class_0_n // class_1_n) / 2, 1, 0)

    right_0_all = len(label_valid_stack) - np.sum( np.where(label_valid_stack == 0, vote_pred, 1) )
    right_1_all = np.sum( np.where(label_valid_stack == 1, vote_pred, 0) )
    
    print(right_0_all)
    print(right_1_all)
    """
    
