
# ======================================================= #
# Copyrights @vankhanhdo2021
# Tasks to do with NTU-RGB-D dataset
# Working on skeleton data of mutual actions
# from A50 -> A60
# ======================================================= #
import os
from os.path import join, exists
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
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn import svm, metrics
from joblib import dump, load

from humiact5_kp_feat_modeling import baseline_model

def load_kp_feat_data_from_csv(ds_type="train", setting='cs', skips=2, dist_included=False):
    #
    # load keypoint feature data from csv files, split feature columns from label columns
    # Input:
    #   -dataset -path to the data set (train, or test)
    #   -dist_included -indicate whether or not distance dimension included
    # Ouput:
    #   -(X,y) -dataframe of feature data and label data
    #
    
    # build the dataset path based on arguments of the function
    dataset = "dataset-NTU-RGB-D/extracted_features/frm_mod_skips"
    if setting == "cs":
        dataset = "{}{}/cross-subject/{}/".format(dataset, skips, ds_type)
    elif setting == "cv":
        dataset = "{}{}/cross-view/{}/".format(dataset, skips, ds_type)

    # load dataset
    df_ds = pd.DataFrame()
    for action_code in action_codes:
        df = pd.read_csv(join(dataset, action_code + ".csv"), index_col=0)

        # check if any NaN, remove and save back to .csv file
        if df.isnull().values.any():
            df = df.dropna()
            df.to_csv(join(dataset, action_code + ".csv"))

        labels = [action_code for i in range(len(df.index))]
        df.insert(df.shape[1], "label", labels)
        df_ds = df_ds.append(df)

    # print(df_ds.shape)

    # separate features and labels
    if dist_included:
        X = df_ds.iloc[:, :-1]
    else:
        X = df_ds.iloc[:, :-2]

    y = df_ds.iloc[:, -1]

    # encode the labels
    encoder = LabelEncoder()
    encoder.fit(y)
    y = encoder.transform(y)

    # save encoded classes if not exists
    if not os.path.exists(join(dir_path, 'ntu_saved_models/encoded-classes.joblib')):
        encoded_classes = list(encoder.classes_)
        dump(encoded_classes, 'ntu_saved_models/encoded-classes.joblib')

    return (X, y)    
    
def show_confusion_matrix_and_save(y_pred, y_actual,title, save=False):
    #
    # show confusion matrix for mis-classification on validation set
    #
    confusion = confusion_matrix(y_actual, y_pred, normalize='pred');
    # print(confusion)

    df_cm = DataFrame(confusion, index=action_codes, columns=action_codes)

    fig, ax = pyplot.subplots(figsize=(9, 9))
    ax = sn.heatmap(df_cm, cmap='Oranges', annot=True, ax=ax)
    ax.set_ylabel('True')
    ax.set_xlabel('Predicted')
    ax.set_title(title)
    pyplot.show()

    if save:
      fig.savefig("experiments/SVM_ntu_rgb_d_kp_feat_output/{}.jpg".format(title))

# build SVM classifier
def build_and_save_SVM_Classifier(setting='cs',skips=8):
    #
    # build the SVM classifier for 11 mutual actions (A050-A060)
    # on ntu_rgb_d dataset
    # Input:
    #   -setting -cross-subject 'cs' or cross-view 'cv' setting
    #   -skips -indicate a specific feature set corresponding with
    #   a specific value of 'skips' parameter when doing feature engineering
    #   Example: skips=8, the distance between the next selected frame and
    #                     current selected frame is equal to 8

    (X,y) = load_kp_feat_data_from_csv(ds_type="train", setting=setting, skips=skips,dist_included=True)

    # train test split
    # split train and test set
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33, shuffle=True)

    # create SVM classifier
    clf = svm.SVC(kernel='rbf')

    # cross validation
    # cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    # plot_learning_curve(clf, 'Learning curve for humiact model', X, y, ylim=(0.4, 1.01),
    #                     cv=cv, n_jobs=4)

    # create appropriate folder name for saving classifier beforehand
    clf_folder = "ntu_saved_models/frm_mod_skips{}/{}/".format(skips, setting)
    # clf_folder = join(dir_path,clf_folder)
    # # create directory if not exists
    # if not exists(clf_folder):
    #     os.makedirs(clf_folder)

    print ("Model is fitting...")
    clf.fit(X_train,y_train)
    print ("Done!")

    clf_full_path = clf_folder + "SVM-model-with-keypfea.joblib"
    # dump classifier to file
    dump(clf, clf_full_path)

    # predict the response
    print ("Predicting train/val set...")
    y_pred_train = clf.predict(X_train)
    y_pred = clf.predict(X_val)
    print ("Done!")

    # show confusion matrix for mis-classification on training set
    show_confusion_matrix_and_save(y_pred_train, y_train, "SVM- Training set ({},skips{})".format(setting,skips), save=True)
    # show confusion matrix for mis-classification on validation set
    show_confusion_matrix_and_save(y_pred, y_val,"SVM- Validation set ({},skips{})".format(setting,skips), save=True)

    train_acc = metrics.accuracy_score(y_train, y_pred_train)
    val_acc = metrics.accuracy_score(y_val, y_pred)
    # save train/val accuracy to txt file
    file_name = "experiments/SVM_ntu_rgb_d_kp_feat_output/train_val_accuracy.txt"

    with open(file_name, "a") as f:
      f.write("setting={},skips={}: train_acc= {}, val_acc= {}\n".format(setting,skips,train_acc,val_acc))

    # Model Accuracy: how often does the classifier predict correctly?
    print("Training accuracy:", train_acc)
    print("Validation accuracy:", val_acc)

def evaluate_SVM_Classifier_On_Test_Set(setting='cs', skips=2):
    #
    # evaluate the accuracy of the model in test set
    #

    # load actual test set
    (X,y) = load_kp_feat_data_from_csv(ds_type="test", setting=setting, skips=skips,dist_included=True)

    # # Demo: test on small data (use skips=64, and on the train set)
    # (X, y) = load_kp_feat_data_from_csv(ds_type="test", setting=setting, skips=128, dist_included=True)

    # # apply scaling to entire dataset
    # trans = StandardScaler()
    # trans = MinMaxScaler()
    # X = trans.fit_transform(X)

    clf_folder = "ntu_saved_models/frm_mod_skips{}/{}/".format(skips, setting)
    clf_full_path = clf_folder + "SVM-model-with-keypfea.joblib"

    # set_trace()

    # load SVM classifier
    clf = load(clf_full_path)

    # predict
    print("Model is predicting...")
    y_pred = clf.predict(X)
    print("Done!")

    # diagnose misclassified samples
    # 0:Boxing, 1:Facing, 2:HHold, 3:HShake, 4:XOXO
    # print_out_misclassified_samples(3, 1, y, y_pred)

    # show confusion matrix for mis-classification on validation set
    show_confusion_matrix_and_save(y_pred, y, "SVM- Test set ({},skips{})".format(setting,skips), save=True)

    test_acc = metrics.accuracy_score(y, y_pred)

    # Model Accuracy: how often is the classifier correct?
    print("Test accuracy:", test_acc)

    # save test accuracy to txt file
    file_name = "experiments/SVM_ntu_rgb_d_kp_feat_output/test_accuracy.txt"

    with open(file_name, "a") as f:
        f.write("setting={},skips={}: test_acc= {}\n".format(setting, skips, test_acc))

def build_and_save_SVM_Classifier_withPCA(setting='cs', skips=8):
    #
    # build the SVM classifier for 11 mutual actions (A050-A060)
    # on ntu_rgb_d dataset
    # Input:
    #   -setting -cross-subject 'cs' or cross-view 'cv' setting
    #   -skips -indicate a specific feature set corresponding with
    #   a specific value of 'skips' parameter when doing feature engineering
    #   Example: skips=8, the distance between the next selected frame and
    #                     current selected frame is equal to 8

    (X,y) = load_kp_feat_data_from_csv(setting=setting, skips=skips,dist_included=True)

    # train test split
    # split train and test set
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33, shuffle=True)

    # set_trace()
    # apply PCA
    # initialize pca instance
    pca = PCA(0.95)

    # fit PCA on training set
    pca.fit(X_train)

    # apply mapping (transform) to both training and test set
    X_train = pca.transform(X_train)
    X_val = pca.transform(X_val)

    dump(pca, 'ntu_saved_models/SVM-PCA-transform.joblib')
        
    # create SVM classifier
    clf = svm.SVC(kernel='rbf')

    # cross validation
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    # plot_learning_curve(clf, 'Learning curve for humiact model', X, y, ylim=(0.4, 1.01),
    #                     cv=cv, n_jobs=4)

    # create appropriate folder name for saving classifier beforehand
    clf_folder = "ntu_saved_models/frm_mod_skips{}/{}/".format(skips, setting)
    clf_folder = join(dir_path,clf_folder)
    # create directory if not exists
    if not exists(clf_folder):
        os.makedirs(clf_folder)

    clf.fit(X_train,y_train)

    clf_full_path = clf_folder + "SVM-model-with-keypfea-withPCA.joblib"
    # dump classifier to file
    dump(clf, clf_full_path)

    # predict the response
    y_pred_train = clf.predict(X_train)
    y_pred = clf.predict(X_val)

    # show confusion matrix for mis-classification on training set
    show_confusion_matrix_and_save(y_pred_train, y_train, "SVM- Training set (PCA)")
    # show confusion matrix for mis-classification on validation set
    show_confusion_matrix_and_save(y_pred, y_val,"SVM- Validation set (PCA)")

    # Model Accuracy: how often is the classifier correct?
    print("Training accuracy:", metrics.accuracy_score(y_train, y_pred_train))
    print("Validation accuracy:", metrics.accuracy_score(y_val, y_pred))

def build_and_save_NN_Classifier(setting='cs', skips=2,
                                 epochs=200,batch_size=64, 
                                 neurons=(700,), af=('relu',),
                                 lr=0.001):
    # load dataset
    (X,y) = load_kp_feat_data_from_csv(ds_type="train", setting=setting, skips=skips,dist_included=True)

    # apply scaler to each type of feature
    # trans = StandardScaler()
    # trans = MinMaxScaler()
    # X = trans.fit_transform(X)

    # split train and test set
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33, shuffle=True)
    
    # get input shape and number of output classes
    num_feats = X_train.shape[1]
    num_cls = len(set(y))
    
    # create baseline model
    model = baseline_model(num_feats, num_cls, 
                           neurons=neurons, af=af, 
                           lr=lr)

    # fit the model
    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=epochs,
                        batch_size=batch_size,
                        verbose=1
                        )
    # show confusion matrix for mis-classification on validation set
    y_pred_train = model.predict_classes(X_train)
    y_pred = model.predict_classes(X_val)

    show_confusion_matrix_and_save(y_pred_train, y_train, "NN network- Training set")
    show_confusion_matrix_and_save(y_pred, y_val, "NN network- Validation set")

    # print last epoch accuracy
    # for training set and validation set
    print("Best training accuracy: ")
    print(max(history.history['accuracy']))
    print("Best validation accuracy: ")
    print(max(history.history['val_accuracy']))

    # save the model
    model.save("ntu_saved_models/NN-model({},skips={},neus={},af={},lr={:.4f})".format(setting,skips,neurons, af, lr))

    # plot loss during training
    pyplot.title('Training- Validation Loss')
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='val')
    pyplot.legend()
    pyplot.grid()
    pyplot.show()

    # plot accuracy during training
    pyplot.title('Training- Validation Accuracy')
    pyplot.plot(history.history['accuracy'], label='train')
    pyplot.plot(history.history['val_accuracy'], label='val')
    pyplot.legend()
    pyplot.grid()
    pyplot.show()

def evaluate_NN_Classifier_On_Test_Set(setting='cs', skips=2,
                                       neurons=(700,), af=('relu',),
                                       lr=0.001):
    #
    # evaluate the accuracy of the model in test set
    #

    # load dataset    
    (X,y) = load_kp_feat_data_from_csv(ds_type="test", setting=setting, skips=skips,dist_included=True)

    # # apply scaling to entire dataset
    # trans = StandardScaler()
    # trans = MinMaxScaler()
    # X = trans.fit_transform(X)

    # load NN classifier
    model = keras.models.load_model("ntu_saved_models/NN-model({},skips={},neus={},af={},lr={:.4f})"\
                                    .format(setting, skips, neurons, af, lr))

    # predict
    y_pred = model.predict_classes(X)

    # diagnose misclassified samples
    # 0:Boxing, 1:Facing, 2:HHold, 3:HShake, 4:XOXO
    # print_out_misclassified_samples(3, 1, y, y_pred)

    # show confusion matrix for mis-classification on validation set
    show_confusion_matrix_and_save(y_pred, y, "Misclassification on Test set, Neural Network")

    # Model Accuracy: how often is the classifier correct?
    print("Test accuracy:", metrics.accuracy_score(y, y_pred))


if __name__ == "__main__":
    print("Starting __main__....")
    dir_path = os.path.dirname(os.path.abspath(__file__))

    action_codes = ["A050","A051","A052","A053","A054","A055","A056","A057","A058","A059","A060"]

    # build SVM classifier
    # build_and_save_SVM_Classifier(setting='cs', skips=8)

    # evaluate the SVM model
    evaluate_SVM_Classifier_On_Test_Set(setting='cs', skips=8)