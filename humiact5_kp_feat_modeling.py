import os
from os import listdir
from os.path import join, isfile
import sys
from sys import platform
import cv2
import argparse
import datetime

import pandas as pd
from pandas import DataFrame
import numpy as np
from matplotlib import pyplot
import seaborn as sn
from mpl_toolkits.axes_grid1 import ImageGrid

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from sklearn import __version__
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn import svm, metrics
from joblib import dump, load

from humiact5_plot_learning_curve import plot_learning_curve
from humiact5_preprocessing import get_all_images_recursively
from humiact5_feature_engineering import extract_ROI_and_HOG_feature, \
     keypoints_sets_merging, \
     engineer_keypoint_based_feature_vector,\
     draw_combined_bounding_box,\
     draw_separated_bounding_boxes

#region train and test modules
def baseline_model(input_dim, num_classes,
                   neurons=(600,), af=('relu',), 
                   lr=0.001):
    #
    # baseline model for classifying
    #
    
    # ===================================================================================================== #
    # ===== Tuning hyper-parameters by kerastuner =========
    # [optimal hyper-parameters obtained]:[model performance obtained using the optimal hyper-parameters]
    # ===================================================================================================== #
    #
    
    # STRUCTURE ATTEMPTS:
    # 1. NN model with 1 densly hidden layer and 1 hidden dropout
    # [units1=,lr=,af=]:[train-accuracy,val-accuracy]
    # - [units1=600, lr=0.001,af='relu']: [,]

    assert len(af) == len(neurons)
    assert len(neurons) > 0
    
    model = Sequential()
    # ============== Visible dropout case =============== #
    # model.add(Dropout(0.3, input_shape=(input_dim,)))
    # =================================================== #
    
    # === start 1st structure ===
    # 1st densely hidden layer #
    model.add(Dense(neurons[0], input_shape=(input_dim,), activation= af[0]))

    # consecutive densely hidden layers if any #
    for l in range(1, len(neurons)):
        model.add(Dense(neurons[l], activation= af[l]))
    
    # === output layer with softmax ====
    model.add(Dense(num_classes, activation="softmax"))
    
    # === learning rate ===
    opt = Adam(learning_rate= lr)

    # compile model
    model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    
    return model

# build 1 hidden NN classifier
def build_and_save_NN_Classifier(epochs=250,batch_size=16, 
                                 neurons=(600,),
                                 af=('relu',),
                                 lr=0.001):
    # load dataset
    (X,y) = load_kp_feat_data_from_csv(train_dataset,dist_included=True)

    # apply scaler to each type of feature
    # trans = StandardScaler()
    # trans = MinMaxScaler()
    # X = trans.fit_transform(X)

    # split train and test set
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33, shuffle=True)
    
    # get input shape and number of output classes
    num_feats = X_train.shape[1]
    num_cls = len(set(y))
    
    # create baseline model without PCA
    model = baseline_model(num_feats, num_cls,
                           neurons=neurons, 
                           af=af,
                           lr=lr)

    # fit the model
    print ("Model is fitting...")
    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=epochs,
                        batch_size=batch_size,
                        verbose=0
                        )
    print ("Done!")

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
    model.save("humiact5_saved_models/NN-model(af={},lr={:.4f},neus={})".format(af,lr,first_neurons))

    # plot loss during training
    pyplot.title('Training / Validation Loss')
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='val')
    pyplot.legend()
    pyplot.grid()
    pyplot.show()

    # plot accuracy during training
    pyplot.title('Training / Validation Accuracy')
    pyplot.plot(history.history['accuracy'], label='train')
    pyplot.plot(history.history['val_accuracy'], label='val')
    pyplot.legend()
    pyplot.grid()
    pyplot.show()
    


# build SVM classifier
def build_and_save_SVM_Classifier(attempt= 10,save_stats=False):
    #
    # build SVM classifier and save the model to disk
    # Input:
    #   -attempt - number of times to train the model
    #   (this is for statistical purpose)
    #
    # load dataset    
    (X,y) = load_kp_feat_data_from_csv(train_dataset,dist_included=True)

    # # apply scaling to entire dataset
    # trans = StandardScaler()
    # trans = MinMaxScaler()
    # X = trans.fit_transform(X)

    # try "attempt" times training the model by re-splitting train/val set
    # This is for statistical purpose

    # declare the ndarray to save all performance details
    # row = attempt
    # col = 1st-col(train_acc)-2nd-col(val_acc)-3th-8th-col(full_train_prec)-9th-12th-col(full_val_prec)
    performance_stats = np.array(np.zeros(shape=(attempt,12)))

    for att in range(attempt):
        # split train and test set
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33, shuffle=True)

        # create SVM classifier
        clf = svm.SVC(kernel='rbf')

        print ("Sklearn vesion: {}".format(__version__))

        # cross validation
        cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
        # plot_learning_curve(clf, 'Learning curve for humiact model', X, y, ylim=(0.4, 1.01),
        #                     cv=cv, n_jobs=4)

        clf.fit(X_train,y_train)

        # dump classifier to file
        dump(clf, 'humiact5_saved_models/SVM-model-with-keypfea(attempt={}).joblib'.format(att))

        # predict the response
        y_pred_train = clf.predict(X_train)
        y_pred_val = clf.predict(X_val)

        # ================================================================== #
        # ============= Save trained classifier performance ================ #
        # overall accuracy on training set and test set
        train_acc = metrics.accuracy_score(y_train, y_pred_train)
        val_acc = metrics.accuracy_score(y_val, y_pred_val)

        # accuracy of the classifier on each class of action
        # 5 classes in total: 0:Boxing,1:Facing,2:HHold,3:HShake,4:XOXO
        # on training set, precision calculation
        full_train_prec = [sum(np.logical_and(y_train == cls, y_pred_train == cls)) / sum(y_pred_train == cls) for cls in range(5)]
        # on validation set, precision calculation
        full_val_prec = [sum(np.logical_and(y_val == cls, y_pred_val == cls)) / sum(y_pred_val == cls) for cls in
                        range(5)]

        # accumulate the result to array and save to file (for statistical purpose)
        performance_stats[att,:] = np.concatenate([[train_acc,val_acc], full_train_prec, full_val_prec])

        # print out details
        # model overall accuracy: how often does the classifier predict correctly?
        print("Training accuracy:", train_acc)
        print("Validation accuracy:", val_acc)

        # details for each class of action
        print ("Precision details on training set ({}):\n{}".format(att+1, full_train_prec))
        print ("Precision details on validation set ({}):\n{}".format(att+1, full_val_prec))

        # ================================================================== #

        # show confusion matrix for mis-classification on training set
        show_confusion_matrix_and_save(y_pred_train, y_train, "SVM- Training set ({})".format(att+1),save=save_stats)
        # show confusion matrix for mis-classification on validation set
        show_confusion_matrix_and_save(y_pred_val, y_val, "SVM- Validation set ({})".format(att+1),save=save_stats)

    if save_stats:
        # save stats data to excel file
        cols = ["Train", "Val", "Train_Boxing", "Train_Facing", "Train_HHold", "Train_HShake", "Train_XOXO",
                "Val_Boxing", "Val_Facing", "Val_HHold", "Val_HShake", "Val_XOXO"]
        df = pd.DataFrame(performance_stats, columns=cols)

        filename = "experiments/SVM_humiact5_kp_feat_output/performance_stats_{}.xlsx".format(
            datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
        df.to_excel(filename)

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

def evaluate_NN_Classifier_On_Test_Set(neurons=(600,), 
                                       af=('relu',),
                                       lr=0.001):
    #
    # evaluate the accuracy of the model in test set
    #

    # load dataset    
    (X,y) = load_kp_feat_data_from_csv(test_dataset,dist_included=True)

    # # apply scaling to entire dataset
    # trans = StandardScaler()
    # trans = MinMaxScaler()
    # X = trans.fit_transform(X)

    # load NN classifier
    model = keras.models.load_model("humiact5_saved_models/NN-model(neus={},af={},lr={:.4f})"\
                                    .format(neurons, af, lr))

    # predict
    y_pred = model.predict_classes(X)

    # diagnose misclassified samples
    # 0:Boxing, 1:Facing, 2:HHold, 3:HShake, 4:XOXO
    # print_out_misclassified_samples(3, 1, y, y_pred)

    # show confusion matrix for mis-classification on validation set
    show_confusion_matrix_and_save(y_pred, y, "Misclassification on Test set, Neural Network")

    # Model Accuracy: how often is the classifier correct?
    print("Test accuracy:", metrics.accuracy_score(y, y_pred))
    

def evaluate_SVM_Classifier_On_Test_Set(att=0, save_stats=True):
    #
    # evaluate the accuracy of the model in test set
    #

    # load dataset    
    (X,y) = load_kp_feat_data_from_csv(test_dataset,dist_included=True)

    # # apply scaling to entire dataset
    # trans = StandardScaler()
    # trans = MinMaxScaler()
    # X = trans.fit_transform(X)

    # load SVM classifier without PCA
    clf = load('humiact5_saved_models/SVM-model-with-keypfea(attempt={}).joblib'.format(att+1))

    # predict
    y_pred = clf.predict(X)

    # diagnose misclassified samples
    # 0:Boxing, 1:Facing, 2:HHold, 3:HShake, 4:XOXO
    print_out_misclassified_samples(3, 1, y, y_pred)

    # show confusion matrix for mis-classification on validation set
    
    show_confusion_matrix_and_save(y_pred, y, "SVM- Test set ({})".format(att+1), save= save_stats)

    test_acc = metrics.accuracy_score(y, y_pred)

    # Model Accuracy: how often is the classifier correct?
    print("Test accuracy:", test_acc)

    # save test accuracy to txt file
    file_name = "experiments/SVM_humiact5_kp_feat_output/test_accuracy.txt"

    with open(file_name, "a") as f:
        f.write("attempt={}: test_acc= {}\n".format(att, test_acc))

def show_confusion_matrix_and_save(y_pred, y_actual, title, save=False):
    #
    # show confusion matrix for mis-classification on validation set
    #
    confusion = confusion_matrix(y_actual, y_pred, normalize='pred');
    # print(confusion)

    df_cm = DataFrame(confusion, index=categories, columns=categories)

    fig, ax = pyplot.subplots(figsize=(7, 6))
    ax = sn.heatmap(df_cm, cmap='Oranges', annot=True, ax=ax)
    ax.set_ylabel('True')
    ax.set_xlabel('Predicted')
    ax.set_title(title)
    pyplot.show()

    if save:
        fig.savefig("experiments/SVM_humiact5_kp_feat_output/{}.jpg".format(title))


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


def load_kp_feat_data_from_csv(dataset, dist_included=False):
    #
    # load keypoint feature data from csv files, split feature columns from label columns
    # Input:
    #   -dataset -path to the data set (train, or test)
    #   -dist_included -indicate whether or not distance dimension included
    # Ouput:
    #   -(X,y) -dataframe of feature data and label data 
    #
    df_ds = pd.DataFrame()
    for cat in categories:
        df = pd.read_csv(join(dataset, cat, cat + "_keypoint_feature.csv"), index_col=0)
        labels = [cat for i in range(len(df.index))]
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
    if not os.path.exists(join(dir_path, 'humiact5_saved_models/encoded-classes.joblib')):
        encoded_classes = list(encoder.classes_)
        dump(encoded_classes, 'humiact5_saved_models/encoded-classes.joblib')
    
    return (X,y)

#endregion

#region main function
if __name__ == "__main__":
    print("Starting __main__....")
    dir_path = os.path.dirname(os.path.abspath(__file__))

    categories = ["Boxing", "Facing", "HHold", "HShake", "XOXO"]
    train_dataset = join(dir_path, 'dataset-humiact5', 'train')
    test_dataset = join(dir_path, 'dataset-humiact5', 'test')
    
    # ================================================================================ #
    # ========================= SVM-based model ====================================== #
    # ================================================================================ #
    
    # build SVM classifier
    # 10 attempts to train the model (for statistical purpose)
    # build_and_save_SVM_Classifier(attempt=10, save_stats=False)
    
    # evaluate SVM classifier accuracy in test set
    for att in range(10):
        evaluate_SVM_Classifier_On_Test_Set(att=att,save_stats=True)
    
    # ================================================================================ #
    
    # ================================================================================ #
    # ========================== NN-based model ====================================== #
    # ================================================================================ #

    # build NN classifier
    # build_and_save_NN_Classifier()
    
    # evaluate NN classifier accuracy in test set
    # evaluate_NN_Classifier_On_Test_Set()
    
    # ================================================================================ #

    # ================================================================================ #
    # =========================== Visualization ====================================== #
    # ================================================================================ #
    # visualize_SVM_Classifier("dataset-humiact5/test/HShake/")
    # ================================================================================ #

#endregion
