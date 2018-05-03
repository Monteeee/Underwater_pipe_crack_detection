import os
import numpy
import shutil

# path of the core folder
DATASET_TRAIN = 'data/sorted/train'

# destiny path of the valid dataset
DATASET_VALID = 'data/sorted/valid'

# destiny path of the test dataset
DATASET_TEST = 'data/sorted/test'

# labels inside of the core folder
labels = [folder for folder in sorted(os.listdir(DATASET_TRAIN))]

for label in labels:
    path_label = os.path.join(DATASET_TRAIN, label)
    n_img = []
    n_img += [i for i in os.listdir(path_label) if i.endswith('.png')]
    n_files = len(n_img)

    # ratio of split
    split_ratio0 = 0.7
    split_ratio1 = 0.2
    split_ratio2 = 0.1
    split_index0 = int(n_files * split_ratio0)
    split_index1 = int(n_files * (1 - split_ratio2))

    # shuffling the dataset
    numpy.random.shuffle(n_img)

    train = n_img[0:split_index0]
    valid = n_img[split_index0:split_index1]
    test = n_img[split_index1:]

    dest_valid = os.path.join(DATASET_VALID, label)

    if not os.path.exists(DATASET_VALID):
        os.makedirs(DATASET_VALID)

    if not os.path.exists(dest_valid):
        os.makedirs(dest_valid)

    for n_valid in valid:
        if not os.path.exists(os.path.join(dest_valid, n_valid)):
            shutil.move(os.path.join(path_label, n_valid), dest_valid)

    dest_test = os.path.join(DATASET_TEST, label)

    if not os.path.exists(DATASET_TEST):
        os.makedirs(DATASET_TEST)

    if not os.path.exists(dest_test):
        os.makedirs(dest_test)

    for n_test in test:
        if not os.path.exists(os.path.join(dest_test, n_test)):
            shutil.move(os.path.join(path_label, n_test), dest_test)