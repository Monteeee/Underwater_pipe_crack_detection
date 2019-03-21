from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss, roc_auc_score, f1_score, roc_curve
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix
from sklearn.metrics import auc
from sklearn.preprocessing import label_binarize
from sklearn.metrics import classification_report
from itertools import cycle
import pickle

img_width, img_height = 80, 80

test_data_dir = "C:/Users/yipai.du/Downloads/large data-20190313T201506Z-001 - Keras_format/sorted/test"
n_classes = 4
# test_data_dir = "C:/Users/yipai.du/Downloads/small/test"
# n_classes = 2

batch_size = 64
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    shuffle=False,
)


model = load_model("my.h5")
# Confution Matrix and Classification Report
y_score = model.predict_generator(
    test_generator, steps=int(test_generator.samples / batch_size + 0.5)
)
y_pred = np.argmax(y_score, axis=1)
y_true = test_generator.classes
if n_classes == 4:
    print("test Acc", round(accuracy_score(y_true, y_pred), 4))
    y_one_hot = label_binarize(y_true, np.arange(n_classes))
    # print("test LogLoss", round(log_loss(y_true, y_pred), 4))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_one_hot[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(y_one_hot.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    with open('./shallow_large_micro_roc.pkl', 'wb') as fP:
            pickle.dump(
                [fpr["micro"], tpr["micro"]],
                fP, pickle.HIGHEST_PROTOCOL)
            fP.close()
    lw = 2
    plt.figure()

    colors = cycle(["aqua", "darkorange", "cornflowerblue", "navy"])
    for i, color in zip(range(n_classes), colors):
        plt.plot(
            fpr[i],
            tpr[i],
            color=color,
            lw=lw,
            label="ROC curve of class {0} (area = {1:0.5f})".format(i, roc_auc[i]),
        )

    plt.plot([0, 1], [0, 1], "k--", lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic to classes")
    plt.legend(loc="lower right")
    plt.show()

    print("test Confusion Matrix \n", confusion_matrix(y_true, y_pred))
    target_names = ["clean", "anode", "connection", "damage"]
    print(
        "test classification report \n",
        classification_report(y_true, y_pred, target_names=target_names, digits=5),
    )
else:
    print("test Acc", round(accuracy_score(y_true, y_pred), 4))
    y_one_hot = label_binarize(y_true, np.arange(n_classes))
    # print("test LogLoss", round(log_loss(y_true, y_pred), 4))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # fpr, tpr, thresholds = roc_curve(y_pred, y_score)

    # roc_auc = auc(fpr, tpr)

    # # fpr, tpr, thresholds = roc_curve(y_one_hot.ravel(), y_score.ravel())
    # lw = 2
    # plt.figure()

    # # colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'navy'])

    # plt.plot(
    #     fpr, tpr, color="darkorange", lw=lw, label="ROC curve (area = %0.2f)" % roc_auc
    # )
    # plt.plot([0, 1], [0, 1], "k--", lw=lw)
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel("False Positive Rate")
    # plt.ylabel("True Positive Rate")
    # plt.title("Receiver Operating Characteristic to classes")
    # plt.legend(loc="lower right")
    # plt.show()

    print("test Confusion Matrix \n", confusion_matrix(y_true, y_pred))
    target_names = ["clean", "damage"]
    print(
        "test classification report \n",
        classification_report(y_true, y_pred, target_names=target_names, digits=5),
    )