import os
from os.path import join
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
from keras import regularizers
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn import svm, metrics
from joblib import dump, load
import pickle
from humiact5_preprocessing import get_all_images_recursively
from humiact5_feature_engineering import extract_ROI_and_HOG_feature, \
     keypoints_sets_merging, \
     engineer_keypoint_based_feature_vector,\
     draw_combined_bounding_box
from humiact5_plot_learning_curve import plot_learning_curve

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
    # model.add(Dropout(0.3, input_shape=(input_dim,)))
    model.add(Dense(256, activation="relu",
                    kernel_regularizer=regularizers.l2(0.1)))
    # model.add(Dropout(0.3))
    model.add(Dense(6, activation="softmax"))
    opt = Adam(learning_rate=0.0001)



    # compile model
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    return model

# build 1 hidden NN classifier
def build_and_save_NN_model(isPCAon= True):
    # load dataset
    df_ds = pd.DataFrame()
    for cat in categories:
        df = pd.read_csv(join(train_dataset, cat, cat + "_extra_feature_32x32.csv"), index_col=0)
        labels = [cat for i in range(len(df.index))]
        df.insert(df.shape[1], "label", labels)
        df_ds = df_ds.append(df)

    # print(df_ds.shape)

    # separate features and labels
    X = df_ds.iloc[:, :-1]
    y = df_ds.iloc[:, -1]

    # separate keypoint features and HOG features apart
    X1 = df_ds.iloc[:,:KP_LEN]
    X2 = df_ds.iloc[:,KP_LEN:-1]

    # apply scaler to each type of feature
    trans = StandardScaler()
    X1 = trans.fit_transform(X1)
    X2 = trans.fit_transform(X2)

    # merge features after applying standard scaler
    X = np.concatenate((X1,X2), axis=1)
    # apply scaling to entire dataset
    X = trans.fit_transform(X)

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

    if isPCAon:
        # initialize pca instance
        pca = PCA(0.95)

        # fit PCA on training set
        pca.fit(X_train)

        # apply mapping (transform) to both training and test set
        X_train = pca.transform(X_train)
        X_val = pca.transform(X_val)

        dump(pca, 'humiact5_saved_models/NN-PCA-transform.joblib')

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

    show_confusion_matrix(y_pred_train, y_train,"NN network/ Training set")
    show_confusion_matrix(y_pred, y_val,"NN network/ Validation set")

    # print last epoch accuracy
    # for training set and validation set
    print("Best training accuracy: ")
    print(max(history.history['accuracy']))
    print("Best validation accuracy: ")
    print(max(history.history['val_accuracy']))

    # save the model
    if isPCAon:
        model.save("humiact5_saved_models/NN-model-with-extrafea-PCA-on")
    else:
        model.save("humiact5_saved_models/NN-model-with-extrafea-PCA-off")

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
def build_and_save_SVM_Classifier(isPCAon= True):
    # load dataset
    df_ds = pd.DataFrame()
    for cat in categories:
        df = pd.read_csv(join(train_dataset, cat, cat + "_extra_feature_32x32.csv"), index_col=0)
        labels = [cat for i in range(len(df.index))]
        df.insert(df.shape[1], "label", labels)
        df_ds = df_ds.append(df)

    # print(df_ds.shape)

    # separate features and labels
    X = df_ds.iloc[:, :-1]
    y = df_ds.iloc[:, -1]

    # separate keypoint features and HOG features apart
    X1 = df_ds.iloc[:,:KP_LEN]
    X2 = df_ds.iloc[:,KP_LEN:-1]

    # apply scaler to each type of feature
    trans = StandardScaler()
    X1 = trans.fit_transform(X1)
    X2 = trans.fit_transform(X2)

    # merge features after applying standard scaler
    X = np.concatenate((X1,X2), axis=1)
    # apply scaling to entire dataset
    X = trans.fit_transform(X)

    # apply one-hot coding for label
    encoder = LabelEncoder()
    encoder.fit(y)
    encoded_y = encoder.transform(y)

    # save encoded classes
    encoded_classes = list(encoder.classes_)
    dump(encoded_classes, 'humiact5_saved_models/encoded-classes.joblib')

    # train test split
    # split train and test set
    X_train, X_val, y_train, y_val = train_test_split(X, encoded_y, test_size=0.10, shuffle=True)

    # apply PCA
    if isPCAon:
        # initialize pca instance
        pca = PCA(0.95)

        # fit PCA on training set
        pca.fit(X_train)

        # apply mapping (transform) to both training and test set
        X_train = pca.transform(X_train)
        X_val = pca.transform(X_val)

        dump(pca, 'humiact5_saved_models/SVM-PCA-transform.joblib')

    # create SVM classifier
    clf = svm.SVC(kernel='rbf')

    # cross validation
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    plot_learning_curve(clf, 'Learning curve for humiact model', X, y, ylim=(0.4, 1.01),
                        cv=cv, n_jobs=4)

    clf.fit(X_train,y_train)

    # dump classifier to file
    if isPCAon:
        dump(clf,'humiact5_saved_models/SVM-model-with-extrafea-PCA-on.joblib')
    else:
        dump(clf, 'humiact5_saved_models/SVM-model-with-extrafea-PCA-off.joblib')

    # predict the response
    y_pred_train = clf.predict(X_train)
    y_pred = clf.predict(X_val)

    # show confusion matrix for mis-classification on training set
    show_confusion_matrix(y_pred_train, y_train,"SVM/ Training set")
    # show confusion matrix for mis-classification on validation set
    show_confusion_matrix(y_pred, y_val,"SVM/ Validation set")

    # Model Accuracy: how often is the classifier correct?
    print("Training accuracy:", metrics.accuracy_score(y_train, y_pred_train))
    print("Validation accuracy:", metrics.accuracy_score(y_val, y_pred))

def evaluate_NN_Classifier_On_Test_Set(test_dataset, isPCAon=True):
    #
    # try to predict some input images with/without yoga poses
    #
    pass

def evaluate_SVM_Classifier_On_Test_Set(train_dataset, isPCAon=True):
    #
    # evaluate the accuracy of the model in test set
    #

    # load dataset
    df_ds = pd.DataFrame()
    for cat in categories:
        df = pd.read_csv(join(train_dataset, cat, cat + "_extra_feature_32x32.csv"), index_col=0)
        labels = [cat for i in range(len(df.index))]
        df.insert(df.shape[1], "label", labels)
        df_ds = df_ds.append(df)

    # separate features and labels
    X = df_ds.iloc[:, :-1]
    y = df_ds.iloc[:, -1]

    # separate keypoint features and HOG features apart
    X1 = df_ds.iloc[:, :KP_LEN]
    X2 = df_ds.iloc[:, KP_LEN:-1]

    # apply scaler to each type of feature
    trans = StandardScaler()
    X1 = trans.fit_transform(X1)
    X2 = trans.fit_transform(X2)

    # merge features after applying standard scaler
    X = np.concatenate((X1, X2), axis=1)
    # apply scaling to entire dataset
    X = trans.fit_transform(X)

    # apply one-hot coding for label
    encoder = LabelEncoder()
    encoder.fit(y)
    encoded_y = encoder.transform(y)

    y_pred = []
    if isPCAon:
        # load PCA model
        pca = load('humiact5_saved_models/SVM-PCA-transform.joblib')

        # apply mapping (transform) to both training and test set
        X = pca.transform(X)

        # load SVM classifier with PCA
        clf = load('humiact5_saved_models/SVM-model-with-extrafea-PCA-on.joblib')

        # predict
        y_pred = clf.predict(X)
    else:
        # load SVM classifier without PCA
        clf = load('humiact5_saved_models/SVM-model-with-extrafea-PCA-off.joblib')

        # predict
        y_pred = clf.predict(X)

    # show confusion matrix for mis-classification on validation set
    show_confusion_matrix(y_pred, encoded_y,"SVM/ Test set")

    # Model Accuracy: how often is the classifier correct?
    print("Test accuracy:", metrics.accuracy_score(encoded_y, y_pred))

def show_confusion_matrix(y_pred, y_actual,title):
    #
    # show confusion matrix for mis-classification on validation set
    #
    confusion = confusion_matrix(y_actual, y_pred, normalize='pred');
    # print(confusion)

    categories_short = ["Boxing", "Facing", "HHolding", "HShaking", "Hugging", "Kissing"]
    df_cm = DataFrame(confusion, index=categories_short, columns=categories_short)

    fig, ax = pyplot.subplots(figsize=(7, 6))
    ax = sn.heatmap(df_cm, cmap='Oranges', annot=True, ax=ax)
    ax.set_ylabel('True')
    ax.set_xlabel('Predicted')
    ax.set_title(title)
    pyplot.show()

def build_confusion_matrix(isSVM=True, isPCAon=True):
    #
    # build confusion matrix for mis-classification in train-val dataset
    #

    # load dataset
    df_ds = pd.DataFrame()
    for cat in categories:
        df = pd.read_csv(join(train_dataset, cat, cat + "_extra_feature_32x32.csv"), index_col=0)
        labels = [cat for i in range(len(df.index))]
        df.insert(df.shape[1], "label", labels)
        df_ds = df_ds.append(df)

    # separate features and labels
    X = df_ds.iloc[:, :-1]
    y = df_ds.iloc[:, -1]

    # separate keypoint features and HOG features apart
    X1 = df_ds.iloc[:, :KP_LEN]
    X2 = df_ds.iloc[:, KP_LEN:-1]

    # apply scaler to each type of feature
    trans = StandardScaler()
    X1 = trans.fit_transform(X1)
    X2 = trans.fit_transform(X2)

    # merge features after applying standard scaler
    X = np.concatenate((X1, X2), axis=1)
    # apply scaling to entire dataset
    X = trans.fit_transform(X)

    # apply one-hot coding for label
    encoder = LabelEncoder()
    encoder.fit(y)
    encoded_y = encoder.transform(y)

    y_preds = []
    if isPCAon:
        if isSVM:
            # load PCA model
            pca = load('humiact5_saved_models/SVM-PCA-transform.joblib')

            # apply mapping (transform) to both training and test set
            X = pca.transform(X)

            # load SVM classifier with PCA
            clf = load('humiact5_saved_models/SVM-model-with-extrafea-PCA-on.joblib')
            # predict poses
            y_preds = clf.predict(X)
        else:
            # load PCA model
            pca = load('humiact5_saved_models/NN-PCA-transform.joblib')

            # apply mapping (transform) to both training and test set
            X = pca.transform(X)

            # load the saved model
            model = keras.models.load_model("humiact5_saved_models/NN-model-with-extrafea-PCA-on")
            y_preds = model.predict_classes(X)
    else:
        if isSVM:
            # load SVM classifier without PCA
            clf = load('humiact5_saved_models/SVM-model-with-extrafea-PCA-off.joblib')
            # predict poses
            y_preds = clf.predict(X)
        else:
            # load the saved model
            model = keras.models.load_model("humiact5_saved_models/NN-model-with-extrafea-PCA-off")
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

def visualize_SVM_Classifier(test_images_path, isPCAon=True):
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

    # test images path
    test_images_path = 'yoga-pose-images-for-test'
    # get all images
    images = get_all_images_recursively(test_images_path)

    # hold combined features
    record_details = []
    # hold image indexes of appropriate combined features
    img_idxs_of_records = []
    # hold test images data
    images_data = []
    for idx in range(0,len(images)):
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
            # merging bounding boxes if applicable
            merged_poseKeypoints = keypoints_sets_merging(datum.poseKeypoints,
                                                          ratio_of_intersec_thresh=0.36,
                                                          ratio_of_distance_thresh=2)
            # assign merged_poseKeypoints to datum.poseKeypoints array
            datum.poseKeypoints = merged_poseKeypoints

            # draw bounding box on test image
            drawn_img = draw_combined_bounding_box(datum)
            drawn_img = cv2.cvtColor(drawn_img, cv2.COLOR_BGR2RGB)
            images_data[idx] = drawn_img

            # an array to save mixed features (keypoints and HOG) of an image
            keyPoint_HOG_feats_arrs = []
            if datum.poseKeypoints.size > 1 and datum.poseKeypoints.ndim == 3:
                # region extract keypoints features
                keypoints_coordinates = datum.poseKeypoints[:, :, :2]

                # translate and scale keypoints by its center and box size,
                # and flattened it to 1D array
                keyPoint_feats_arrs = engineer_keypoint_based_feature_vector(keypoints_coordinates)

                # check if len of the list is only 50 corresponding with only one keypoint set
                # append a zero array of size of 50 to this list
                ONE_KEYPOINTS_SET_FEATURE_LENGTH = 50
                if len(keyPoint_feats_arrs) == ONE_KEYPOINTS_SET_FEATURE_LENGTH:
                    keyPoint_feats_arrs += [0.0] * ONE_KEYPOINTS_SET_FEATURE_LENGTH
                # endregion

                extracted_hog_feats_arrs = \
                    extract_ROI_and_HOG_feature(datum,
                                                image_name,
                                                winSize= (32,32),
                                                isDisplayImageWithROIs=False,
                                                isExtractRoiHogBitmap=True)

                # append HOG feature to a combined feature array
                keyPoint_HOG_feats_arrs = np.append(keyPoint_feats_arrs,
                                                    extracted_hog_feats_arrs)

                # add features data to accumulate array
                record_details.append(keyPoint_HOG_feats_arrs)
                img_idxs_of_records.append(idx)

    if len(record_details) > 0:

        # normalize feature vector
        X = pd.DataFrame(record_details)

        # separate keypoint features and HOG features apart
        X1 = X.iloc[:, :KP_LEN]
        X2 = X.iloc[:, KP_LEN:]

        # apply scaler to each type of feature
        trans = StandardScaler()
        X1 = trans.fit_transform(X1)
        X2 = trans.fit_transform(X2)

        # merge features after applying standard scaler
        X = np.concatenate((X1, X2), axis=1)
        # apply scaling to entire dataset
        X = trans.fit_transform(X)

        pose_preds = []
        if isPCAon:
            # load PCA model
            pca = load('humiact5_saved_models/SVM-PCA-transform.joblib')

            # apply mapping (transform) to both training and test set
            X = pca.transform(X)

            # load SVM classifier with PCA
            clf = load('humiact5_saved_models/SVM-model-with-extrafea-PCA-on.joblib')

            # predict poses
            pose_preds = clf.predict(X)
        else:
            # load SVM classifier without PCA
            clf = load('humiact5_saved_models/SVM-model-with-extrafea-PCA-off.joblib')

            # predict poses
            pose_preds = clf.predict(X)

        # get encoded classes
        encoded_classes = load('humiact5_saved_models/encoded-classes.joblib')

        # build predict poses for each image
        # there might be more than one pose in one image
        image_poses_list = [()]*len(images_data)
        for pose_idx in range(0,len(pose_preds)):
            image_poses_list[img_idxs_of_records[pose_idx]] = image_poses_list[img_idxs_of_records[pose_idx]] + (pose_preds[pose_idx],)

        # count instances of poses in each image
        # and build label for each predicted image
        # hold labels for each predicted image
        predicted_img_lbls = [None]*len(images_data)
        for img_idx in range(0,len(image_poses_list)):
            c = Counter(image_poses_list[img_idx])
            lbl = ""
            for pose in c.keys():
                lbl = lbl + str(c[pose]) + " " + str(encoded_classes[pose]) + ", "
            if lbl == "": lbl = "No pose detected"
            else: lbl = lbl.rstrip().rstrip(',')

            # assign label to list
            predicted_img_lbls[img_idx] = lbl

        # show grid of predicted images
        nCols = 4
        nRows = 1
        if len(images_data)%nCols==0:
            nRows = (int)(len(images_data)/nCols)
        else:
            nRows = (int)(len(images_data)/nCols +1)

        pyplot.rc('figure', figsize=(20, 20))
        fig = pyplot.figure()
        grid = ImageGrid(fig, 111,  # similar to subplot(111)
                         nrows_ncols=(nRows, nCols),  # creates mxn grid of axes
                         axes_pad=0.3,  # pad between axes in inch.
                         share_all=True
                         )
        fontdict = {'fontsize': 10,'color': 'red'}
        for ax, im, lbl in zip(grid, images_data, predicted_img_lbls):
            # resize image
            im = cv2.resize(im, (400, 320))
            # Iterating over the grid returns the Axes.
            ax.imshow(im)
            ax.set_title(lbl, fontdict=fontdict, pad=2)

        pyplot.show()

#endregion

#region main function
if __name__ == "__main__":
    print("Starting __main__....")
    dir_path = os.path.dirname(os.path.abspath(__file__))

    categories = ["Boxing", "Facing", "Handholding", "Handshaking", "Hugging", "Kissing"]
    train_dataset = join(dir_path, 'dataset', 'train')
    test_dataset = join(dir_path, 'dataset', 'test')

    # Key-point feature length
    KP_LEN = 101

    # build the NN model
    # build_and_save_NN_model(isPCAon=True)

    # build SVM classifier
    build_and_save_SVM_Classifier(isPCAon=True)

    # evaluate test set
    # evaluate_SVM_Classifier_On_Test_Set(test_dataset)

    # calculate confusion matrix
    # build_confusion_matrix(isSVM=False,isPCAon=True)

    # classify some input images
    # and print out predicted results in grid
    # test_SVM_Classifier(images_test_path,isPCAon=True)
#endregion
