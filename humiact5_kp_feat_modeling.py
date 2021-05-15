import os
from os import listdir
from os.path import join, isfile
import sys
from sys import platform
import cv2
import argparse
from collections import Counter

import pandas as pd
from pandas import DataFrame
import numpy as np
from matplotlib import pyplot
import seaborn as sn
from mpl_toolkits.axes_grid1 import ImageGrid

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import np_utils
from keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn import svm, metrics
from joblib import dump, load
import pickle

from humiact5_plot_learning_curve import plot_learning_curve
from humiact5_preprocessing import get_all_images_recursively
from humiact5_feature_extraction import extract_ROI_and_HOG_feature, \
     keypoints_sets_merging, \
     engineer_keypoint_based_feature_vector,\
     draw_combined_bounding_box,\
     draw_separated_bounding_boxes

#region train and test modules
def baseline_model(input_dim):
    #
    # baseline model for classifying
    #
    # manually tuning parameters
    # [visible-dropout, hidden-nerons, hidden-dropout]:[train-accuracy,val-accuracy]
    # good 01: [0.01, 5, 0.00]:[95.5, 91.3]
    # good 02:
    # good 03:
    model = Sequential()
    model.add(Dropout(0.2, input_shape=(input_dim,)))
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.1))
    model.add(Dense(6, activation="softmax"))
    opt = Adam(learning_rate=0.0001)

    # compile model
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    return model

# build 1 hidden NN classifier
def build_and_save_NN_model():
    # load dataset
    df_ds = pd.DataFrame()
    for cat in categories:
        df = pd.read_csv(join(train_dataset, cat, cat + "_keypoint_feature.csv"), index_col=0)
        labels = [cat for i in range(len(df.index))]
        df.insert(df.shape[1], "label", labels)
        df_ds = df_ds.append(df)

    # print(df_ds.shape)

    # separate features and labels
    X = df_ds.iloc[:, :-1]
    y = df_ds.iloc[:, -1]

    # apply scaler to each type of feature
    # trans = StandardScaler()
    # trans = MinMaxScaler()
    # X = trans.fit_transform(X)

    # apply one-hot coding for label
    encoder = LabelEncoder()
    encoder.fit(y)
    encoded_y = encoder.transform(y)

    # split train and test set
    X_train, X_val, y_train, y_val = train_test_split(X, encoded_y, test_size=0.33, shuffle=True)

    # convert integers to dummy variables (i.e. one hot encoded)
    # for y_train and y_val
    dummy_y_train = np_utils.to_categorical(y_train)
    dummy_y_val = np_utils.to_categorical(y_val)

    num_feats = X_train.shape[1]
    # create baseline model without PCA
    model = baseline_model(num_feats)

    # fit the model
    history = model.fit(X_train, dummy_y_train,
                        validation_data=(X_val, dummy_y_val),
                        epochs=100,
                        batch_size=16,
                        verbose=0
                        )
    # show confusion matrix for mis-classification on validation set
    y_pred_train = model.predict_classes(X_train)
    y_pred = model.predict_classes(X_val)

    show_confusion_matrix(y_pred_train,y_train,"NN network/ Training set")
    show_confusion_matrix(y_pred,y_val,"NN network/ Validation set")

    # print last epoch accuracy
    # for training set and validation set
    print("Last training accuracy: ")
    print(history.history['accuracy'][len(history.history['accuracy'])-1])
    print("Last validation accuracy: ")
    print(history.history['val_accuracy'][len(history.history['val_accuracy'])-1])

    # save the model
    model.save("humiact5_saved_models/NN-model-with-keypfea")

    # plot loss during training
    pyplot.title('Training / Validation Loss')
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='val')
    pyplot.legend()
    pyplot.show()

    # plot accuracy during training
    pyplot.title('Training / Validation Accuracy')
    pyplot.plot(history.history['accuracy'], label='train')
    pyplot.plot(history.history['val_accuracy'], label='val')
    pyplot.legend()
    pyplot.show()

# build SVM classifier
def build_and_save_SVM_Classifier():
    # load dataset
    df_ds = pd.DataFrame()
    for cat in categories:
        df = pd.read_csv(join(train_dataset, cat, cat + "_keypoint_feature.csv"), index_col=0)
        labels = [cat for i in range(len(df.index))]
        df.insert(df.shape[1], "label", labels)
        df_ds = df_ds.append(df)

    # print(df_ds.shape)

    # separate features and labels
    X = df_ds.iloc[:, :-1]
    y = df_ds.iloc[:, -1]

    # # apply scaling to entire dataset
    # trans = StandardScaler()
    # trans = MinMaxScaler()
    # X = trans.fit_transform(X)

    # apply one-hot coding for label
    encoder = LabelEncoder()
    encoder.fit(y)
    encoded_y = encoder.transform(y)

    # save encoded classes
    encoded_classes = list(encoder.classes_)
    dump(encoded_classes, 'humiact5_saved_models/encoded-classes.joblib')

    # train test split
    # split train and test set
    X_train, X_val, y_train, y_val = train_test_split(X, encoded_y, test_size=0.33, shuffle=True)

    # create SVM classifier
    clf = svm.SVC(kernel='rbf')

    # cross validation
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    # plot_learning_curve(clf, 'Learning curve for humiact model', X, y, ylim=(0.4, 1.01),
    #                     cv=cv, n_jobs=4)

    clf.fit(X_train,y_train)

    # dump classifier to file
    dump(clf, 'humiact5_saved_models/SVM-model-with-keypfea.joblib')

    # predict the response
    y_pred_train = clf.predict(X_train)
    y_pred = clf.predict(X_val)

    # show confusion matrix for mis-classification on training set
    show_confusion_matrix(y_pred_train, y_train, "SVM/ Training set")
    # show confusion matrix for mis-classification on validation set
    show_confusion_matrix(y_pred, y_val,"SVM/ Validation set")

    # Model Accuracy: how often is the classifier correct?
    print("Training accuracy:", metrics.accuracy_score(y_train, y_pred_train))
    print("Validation accuracy:", metrics.accuracy_score(y_val, y_pred))

def print_out_misclassified_samples(encoded_ground_truth, encoded_predicted, y_ground_truth, y_predicted):
    #
    # Print out samples which mistook "predicted" class for "ground_truth" class
    # Encoded classes: Boxing:0,Facing:1,HHold:2,HShake:3,XOXO:4
    # NOTE: This function is for diagnostic purpose.
    #
    # Input:
    #   -ground_truth -encoded ground truth class whose samples are considering
    #   -predicted -an encoded class predicted by classifier which differs from ground truth class
    #   -y_ground_truth -ground truth labels of the test set (no shuffle allowed)
    #   -y_predicted -predicted labels for the set
    # Output:
    #   -void() -print out misclassified samples
    #

    # misclassified indexes
    misclassified_indexes = np.where((y_ground_truth==encoded_ground_truth)&(y_predicted==encoded_predicted))

    # print out indexes of misclassified samples
    print ("Indexes of {} samples for which the classifier predicted as {}.".format(categories[encoded_ground_truth],categories[encoded_predicted]))

    for idx in misclassified_indexes:
        print(idx, end="")
    print("\n")

    pass

def evaluate_NN_Classifier_On_Test_Set(test_dataset):
    #
    # try to predict some input images with/without yoga poses
    #
    pass

def evaluate_SVM_Classifier_On_Test_Set():
    #
    # evaluate the accuracy of the model in test set
    #

    # load dataset
    df_ds = pd.DataFrame()
    for cat in categories:
        df = pd.read_csv(join(test_dataset, cat, cat + "_keypoint_feature.csv"), index_col=0)
        labels = [cat for i in range(len(df.index))]
        df.insert(df.shape[1], "label", labels)
        df_ds = df_ds.append(df)

    # separate features and labels
    X = df_ds.iloc[:, :-1]
    y = df_ds.iloc[:, -1]

    # # apply scaling to entire dataset
    # trans = StandardScaler()
    # trans = MinMaxScaler()
    # X = trans.fit_transform(X)

    # apply one-hot coding for label
    encoder = LabelEncoder()
    encoder.fit(y)
    encoded_y = encoder.transform(y)

    # load SVM classifier without PCA
    clf = load('humiact5_saved_models/SVM-model-with-keypfea.joblib')

    # predict
    y_pred = clf.predict(X)

    # diagnose misclassified samples
    # 0:Boxing, 1:Facing, 2:HHold, 3:HShake, 4:XOXO
    print_out_misclassified_samples(3, 1, encoded_y, y_pred)

    # show confusion matrix for mis-classification on validation set
    show_confusion_matrix(y_pred, encoded_y,"SVM/ Test set")

    # Model Accuracy: how often is the classifier correct?
    print("Test accuracy:", metrics.accuracy_score(encoded_y, y_pred))

def show_confusion_matrix(y_pred, y_actual,title):
    #
    # show confusion matrix for mis-classification on validation set
    #
    confusion = confusion_matrix(y_actual, y_pred, normalize='true');
    # print(confusion)

    df_cm = DataFrame(confusion, index=categories, columns=categories)

    fig, ax = pyplot.subplots(figsize=(7, 6))
    ax = sn.heatmap(df_cm, cmap='Oranges', annot=True, ax=ax)
    ax.set_ylabel('True')
    ax.set_xlabel('Predicted')
    ax.set_title(title)
    pyplot.show()

def build_confusion_matrix(isSVM=True):
    #
    # build confusion matrix for mis-classification in train-val dataset
    #

    # load dataset
    df_ds = pd.DataFrame()
    for cat in categories:
        df = pd.read_csv(join(train_dataset, cat, cat + "_keypoint_feature.csv"), index_col=0)
        labels = [cat for i in range(len(df.index))]
        df.insert(df.shape[1], "label", labels)
        df_ds = df_ds.append(df)

    # print(df_ds.shape)

    # separate features and labels
    X = df_ds.iloc[:, :-1]
    y = df_ds.iloc[:, -1]

    # apply scaling to entire dataset
    # trans = StandardScaler()
    # trans = MinMaxScaler()
    # X = trans.fit_transform(X)

    # apply one-hot coding for label
    encoder = LabelEncoder()
    encoder.fit(y)
    encoded_y = encoder.transform(y)

    y_preds = []
    if isSVM:
        # load SVM classifier without PCA
        clf = load('humiact5_saved_models/SVM-model-with-keypfea.joblib')
        # predict poses
        y_preds = clf.predict(X)
    else:
        # load the saved model
        model = keras.models.load_model("humiact5_saved_models/NN-model-with-keypfea")
        y_preds = model.predict_classes(X)

    # Model Accuracy: how often is the classifier correct?
    print("Accuracy on train dataset (included both train and validation samples):",
          metrics.accuracy_score(encoded_y, y_preds))

    confusion = confusion_matrix(encoded_y, y_preds, normalize='true');
    # print(confusion)

    df_cm = DataFrame(confusion, index=categories, columns=categories)

    ax = sn.heatmap(df_cm, cmap='Oranges', annot=True)
    ax.set_ylabel('True')
    ax.set_xlabel('Predicted')
    pyplot.show()

def visualize_SVM_Classifier(test_images_path):
    #
    # try to predict some input images with/without poses
    #

    if test_images_path == "":
        return

    #region init OpenPose
    # Import Openpose (Windows/Ubuntu/OSX)
    dir_path = os.path.dirname(os.path.abspath(__file__))
    try:
        # Windows Import
        if platform == "win32":
            # Change these variables to point to the correct folder (Release/x64 etc.)
            sys.path.append(dir_path + '/dependencies/openpose/libs');
            os.environ['PATH'] = os.environ['PATH'] + ';' + dir_path + '/dependencies/openpose/dlls;'
            import pyopenpose as op
        else:
            # Change these variables to point to the correct folder (Release/x64 etc.)
            sys.path.append('../../python');
            # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
            # sys.path.append('/usr/local/python')
            from openpose import pyopenpose as op
    except ImportError as e:
        print(
            'Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
        raise e

    # Custom Params (refer to include/openpose/flags.hpp for more parameters)
    params = dict()
    params["model_folder"] = "./dependencies/openpose/models/"
    params["net_resolution"] = "320x176"

    # Flags
    parser = argparse.ArgumentParser()
    args = parser.parse_known_args()
    # parse other params passed
    for i in range(0, len(args[1])):
        curr_item = args[1][i]
        if i != len(args[1]) - 1:
            next_item = args[1][i + 1]
        else:
            next_item = "1"
        if "--" in curr_item and "--" in next_item:
            key = curr_item.replace('-', '')
            if key not in params:  params[key] = "1"
        elif "--" in curr_item and "--" not in next_item:
            key = curr_item.replace('-', '')
            if key not in params: params[key] = next_item

    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()
    # endregion

    # get all images
    images = [join(test_images_path, f) for f in listdir(test_images_path) if isfile(join(test_images_path, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

    # hold keypoint features
    kp_record_details = []
    # hold image indexes of appropriate combined features
    img_idxs_of_records = []
    # hold test images data
    images_data = []
    for idx in range(len(images)):
        print("\nCurrent image: " + images[idx] + "\n")

        # get image full name (with file extension)
        _, image_full_name = os.path.split(images[idx])
        # image name without extension
        image_name = image_full_name[0:image_full_name.rindex('.')]

        # processed by OpenPose
        datum = op.Datum()
        imageToProcess = cv2.imread(images[idx])
        img_rgb = cv2.cvtColor(imageToProcess, cv2.COLOR_BGR2RGB)
        # add image data to list
        images_data.append(img_rgb)

        height, width, channels = imageToProcess.shape
        datum.cvInputData = imageToProcess
        opWrapper.emplaceAndPop([datum])

        # print("Body keypoints: \n" + str(datum.poseKeypoints))

        # check if exists pose key points data
        if not (datum.poseKeypoints.size < 2):
            # merging keypoints sets if applicable
            merged_poseKeypoints = keypoints_sets_merging(datum.poseKeypoints,
                                                          ratio_of_intersec_thresh=0.36,
                                                          ratio_of_distance_thresh=2)

            # assign merged_poseKeypoints to datum.poseKeypoints array
            datum.poseKeypoints = merged_poseKeypoints

            # draw bounding box on test image
            drawn_img = draw_separated_bounding_boxes(datum)
            drawn_img = cv2.cvtColor(drawn_img, cv2.COLOR_BGR2RGB)
            images_data[idx] = drawn_img

            # an array to save mixed features (keypoints and HOG) of an image
            keyPoint_feats_arrs = []
            if datum.poseKeypoints.size > 1 and datum.poseKeypoints.ndim == 3:
                # region extract keypoints features
                keypoints_coordinates = datum.poseKeypoints[:, :, :2]

                # after merging with new keypoints sets, these coordinates are translated to their center and scaled by their box size -> added in keypoints features
                # translate and scale keypoints by its center and box size,
                # and flattened it to 1D array
                keyPoint_feats_arrs = engineer_keypoint_based_feature_vector(keypoints_coordinates)

                # add to list of keypoint-only features
                kp_record_details.append((keyPoint_feats_arrs))

                # endregion

    if len(kp_record_details) > 0:

        # normalize feature vector
        X = pd.DataFrame(kp_record_details)

        # # apply scaling to entire dataset
        # trans = StandardScaler()
        # trans = MinMaxScaler()
        # X = trans.fit_transform(X)

        # load SVM classifier without PCA
        clf = load('humiact5_saved_models/SVM-model-with-keypfea.joblib')

        # predict poses
        pose_preds = clf.predict(X)

        # get encoded classes
        encoded_classes = load('humiact5_saved_models/encoded-classes.joblib')

        # show grid of predicted images
        nCols = 4
        nRows = 1
        if len(images_data)%nCols==0:
            nRows = (int)(len(images_data)/nCols)
        else:
            nRows = (int)(len(images_data)/nCols +1)

        size_w = 4*nCols
        size_h = 3.2*nRows

        pyplot.rc('figure', figsize=(size_w, size_h))
        fig = pyplot.figure()
        grid = ImageGrid(fig, 111,  # similar to subplot(111)
                         nrows_ncols=(nRows, nCols),  # creates mxn grid of axes
                         axes_pad=0.3,  # pad between axes in inch.
                         share_all=True
                         )
        fontdict = {'fontsize': 10,'color': 'red'}
        for ax, im, pred in zip(grid, images_data, pose_preds):
            # resize image
            im = cv2.resize(im, (400, 320))
            # Iterating over the grid returns the Axes.
            ax.imshow(im)
            ax.set_title(encoded_classes[pred], fontdict=fontdict, pad=2)

        pyplot.show()

#endregion

#region main function
if __name__ == "__main__":
    print("Starting __main__....")
    dir_path = os.path.dirname(os.path.abspath(__file__))

    categories = ["Boxing", "Facing", "HHold", "HShake", "XOXO"]
    train_dataset = join(dir_path, 'dataset-humiact5', 'train')
    test_dataset = join(dir_path, 'dataset-humiact5', 'test')

    # # build the NN model
    # build_and_save_NN_model()

    # build SVM classifier
    # build_and_save_SVM_Classifier()

    # evaluate accuracy in test set
    evaluate_SVM_Classifier_On_Test_Set()

    # # calculate confusion matrix
    # build_confusion_matrix(isSVM=False)

    # classify some input images
    # and print out predicted results in grid
    # visualize_SVM_Classifier("dataset-humiact5/test/HShake/")
#endregion
