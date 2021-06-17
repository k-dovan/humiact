import os
from os.path import join

from joblib import dump, load

import numpy as np
import pandas as pd
from matplotlib import pyplot

import seaborn as sn
import keras
from keras import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import kerastuner as kt
from kerastuner import HyperModel
from pandas import DataFrame
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_kp_feat_data_from_csv(setting='cs', skips=2, dist_included=False):
    #
    # load keypoint feature data from csv files, split feature columns from label columns
    # Input:
    #   -dataset -path to the data set (train, or test)
    #   -dist_included -indicate whether or not distance dimension included
    # Ouput:
    #   -(X,y) -dataframe of feature data and label data
    #
    
    # build the dataset path based on arguments of the function
    train_dataset = "dataset-NTU-RGB-D/extracted_features/frm_mod_skips"
    if setting == "cs":
        train_dataset = "{}{}/cross-subject/train/".format(train_dataset, skips)
    elif setting == "cv":
        train_dataset = "{}{}/cross-view/train/".format(train_dataset, skips)

    # load dataset
    df_ds = pd.DataFrame()
    for action_code in action_codes:
        df = pd.read_csv(join(train_dataset, action_code + ".csv"), index_col=0)

        # check if any NaN, remove and save back to .csv file
        if df.isnull().values.any():
            df = df.dropna()
            df.to_csv(join(train_dataset, action_code + ".csv"))

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

def show_confusion_matrix(y_pred, y_actual, title):
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

# ======================================================================================== #
# ============================= HYPER-PARAMETER TUNING =================================== #
# ======================================================================================== #

class NNHyperModel(HyperModel):
    #
    # build a hyper model that inherits from HyperModel class
    #

    def __init__(self, input_dim, num_classes):
        self.input_dim = input_dim
        self.num_classes = num_classes

    def build(self, hp):
        model = keras.Sequential()

        # add one or more densely hidden layers (with or without a consecutive Dropout) before the output layer
        for i in range(hp.Int('layers', 1, 3, default=1)):
            # ====== add 1st densely hidden layer with input_dim specified ======
            if i == 0:
                model.add(Dense(hp.Int('units_' + str(i), min_value=100, max_value=1000, step=100, default=200),
                                input_shape=(self.input_dim,),
                                activation=hp.Choice('activations_' + str(i), ['relu', 'sigmoid'], default='relu'))
                          )
                # model.add(Dropout(hp.Float('dropouts_'+str(i), min_value=0.0, max_value=0.3, step=0.05, default=0.0)))

            # ====== add the rest densely hidden layers if current 'layers' >= 2 ======
            else:
                model.add(Dense(hp.Int('units_' + str(i), min_value=100, max_value=1000, step=100, default=200),
                                activation=hp.Choice('activations_' + str(i), ['relu', 'sigmoid'], default='relu'))
                          )
                # model.add(Dropout(hp.Float('dropouts_' + str(i), min_value=0.0, max_value=0.3, step=0.05, default=0.0)))

        # ====== add the output layer with softmax ======
        model.add(Dense(self.num_classes, activation='softmax'))

        opt = Adam(learning_rate=hp.Float('learning_rate', min_value=1e-5, max_value=1e-2, default=1e-3,
                                          sampling='LOG')
                   )

        model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

        return model

def do_Hyperband_and_get_optimal_model(setting='cs',skips=2):
    #
    # build a tuner, search for optimal hyper-parameters
    # Output:
    #   - the optimal model obtained after the tuning process 
    #
    
    # load the key-point feature data from csv files
    (X,y) = load_kp_feat_data_from_csv(setting=setting, skips=skips,dist_included=True)
    
    # get feature dimension
    num_feats = X.shape[1]
    num_cls = len(set(y))
    
    # initialize a NNHyperModel instance
    hypermodel = NNHyperModel(input_dim=num_feats,num_classes=num_cls)
    
    # init a Hyperband tuner
    tuner = kt.Hyperband(hypermodel, 
                         objective= 'val_accuracy',
                         max_epochs = 5,
                         factor= 3,
                         directory= 'hyperband-tuner',
                         project_name= 'ntu_rgb_d',
                         overwrite= True
                         )
    
    stop_early = keras.callbacks.EarlyStopping(monitor='val_loss', patience= 5)

    tuner.search_space_summary()
    
    tuner.search(X, y, epochs=3, validation_split=0.2, callbacks=[stop_early])

    tuner.results_summary()
    
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    
    # build the model with the optimal hyperparameters 
    model = tuner.hypermodel.build(best_hps)
    
    # return the optimal model
    return model


def do_RandomSearch_and_get_optimal_model(setting='cs',skips=2):
    #
    # build a tuner, search for optimal hyper-parameters
    # Output:
    #   - the optimal model obtained after the tuning process 
    #
    
    # load the key-point feature data from csv files
    (X,y) = load_kp_feat_data_from_csv(setting=setting, skips=skips,dist_included=True)
    
    # get feature dimension
    num_feats = X.shape[1]
    num_cls = len(set(y))
    
    # initialize a NNHyperModel instance
    hypermodel = NNHyperModel(input_dim=num_feats,num_classes=num_cls)
    
    # init a RandomSearch tuner
    tuner = kt.RandomSearch(hypermodel,
                            objective='val_accuracy',
                            max_trials=5,
                            executions_per_trial=3,
                            directory='randomsearch-tuner',
                            project_name= 'ntu_rgb_d',
                            overwrite=True)

    tuner.search_space_summary()

    tuner.search(x=X,
                 y=y,
                 epochs=3,
                 validation_split=0.2)

    tuner.results_summary()
    
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    
    # build the model with the optimal hyperparameters 
    model = tuner.hypermodel.build(best_hps)
    
    # return the optimal model
    return model

def train_NN_Classifier_with_specific_hyperparams(model, setting='cs', skips=2):

    # load the key-point feature data from csv files
    (X,y) = load_kp_feat_data_from_csv(setting=setting, skips=skips,dist_included=True)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33, shuffle=True)

    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=500,
                        batch_size=16,
                        verbose=1
                        )
    # show confusion matrix for mis-classification on validation set
    y_pred_train = model.predict_classes(X_train)
    y_pred = model.predict_classes(X_val)

    show_confusion_matrix(y_pred_train, y_train, "NN network- Training set")
    show_confusion_matrix(y_pred, y_val, "NN network- Validation set")

    # print last epoch accuracy
    # for training set and validation set
    print("Best training accuracy: ")
    print(max(history.history['accuracy']))
    print("Best validation accuracy: ")
    print(max(history.history['val_accuracy']))

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

def create_one_densely_hidden_layer_nn_model(input_dim, num_classes,
                                              activation_fn= 'relu',
                                              learning_rate= 1e-3,
                                              first_neurs= 100
                                              ):
    #
    # create a neural network model with only one densely hidden layer
    #
    # Input:
    #   -input_dim -length of feature vector
    #   -num_classes -number of classes
    #   -activation_fn -activation function used
    #   -learning_rate -learning rate chosen to train the model
    #   -first_neurs -number of neurons in the first densely hidden layer
    # Output:
    #   -model -the built model ready for training
    #

    model = Sequential()

    # === start 1st structure ===
    # 1st densely hidden layer #
    model.add(Dense(first_neurs, input_shape=(input_dim,), activation=activation_fn))

    # === output layer with softmax ====
    model.add(Dense(num_classes, activation="softmax"))

    # === learning rate ===
    opt = Adam(learning_rate=learning_rate)

    # compile model
    model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    return model

def do_hyperparam_tuning_in_small_search_space(setting='cs',
                                               skips= 2,
                                               activation_fn= ('relu', 'sigmoid'),
                                               learning_rate= (-5, -2, 4),
                                               first_neurons= (100, 1000, 100),
                                               attempt= 5,
                                               epochs= 200
                                               ):
    #
    # This function is to conduct experiments of hyper-parameter tuning for small, specific search space
    # Three types of hyper-parameter are involved: activation function, learning rate, and the number of
    # neurons in the first densely hidden layer (for only the NN with 1 densely hidden layer).
    # For each specific hyper-parameter configuration, to reduce variance in performance, we will repeatedly
    # train each model number of times ('attempt') and consider its average result.
    # Input:
    #   -activation_fn -activation function involved in search space
    #   -learning_rate -learning rate range obeys log space (e.g. from 1e-5 to 1e-2 for 4 samples)
    #   -first_neurons -the range of the number of neurons in the first densely hidden layer
    #   Format: (start, stop, step)
    #   -attempt -number of of times to train each model in order to reduce variance in the obtained result
    #   The average values will be taken into consideration
    #   -epochs -number of epochs each neural network model will be trained
    #

    # load data features for training
    (X,y) = load_kp_feat_data_from_csv(setting=setting, skips=skips,dist_included=True)
    
    # get input shape and number of output classes
    num_feats = X.shape[1]
    num_cls = len(set(y))

    # parse parameters
    # activation function
    n_af = len(activation_fn)
    # learning rate
    learning_rate = np.logspace(learning_rate[0],learning_rate[1],learning_rate[2])
    n_lr = len(learning_rate)
    # first neurons
    first_neurons = list(range(first_neurons[0],first_neurons[1]+1,first_neurons[2]))
    n_neus = len(first_neurons)

    # create a ndarray to record all experiments data
    # metrics involved for measurement
    metrics = ('loss', 'accuracy', 'val_loss', 'val_accuracy')
    n_mts = len(metrics)

    # detailed data array
    detailed_tuning_data = np.array(np.zeros(shape=(n_af,n_lr,n_neus,attempt,epochs,n_mts)))

    # summary data array
    summary_tuning_data = np.array(np.zeros(shape=(n_af,n_lr,n_neus,n_mts)))

    # search space
    for af in range(n_af):
        for lr in range(n_lr):
            for neur in range(n_neus):
                # for each configuration, calculate average of the best acc/loss of train/val set
                avg_best_values = np.zeros(shape=(1,n_mts)) # loss,acc,val_loss,val_acc

                # number of attempts
                for att in range(attempt):

                    # re-split train/val set each time
                    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33, shuffle=True)

                    # init a neural network
                    # create a NN model with current hyper-parameters
                    model = create_one_densely_hidden_layer_nn_model(num_feats, num_cls,
                                                                     activation_fn[af],
                                                                     learning_rate[lr],
                                                                     first_neurons[neur]
                                                                    )
                    # fit the model
                    print ("\n=============================================================================")
                    print ("NN model (af={},lr={},neurons={},attempt={}) is fitting...".format(activation_fn[af],
                                                                    learning_rate[lr],
                                                                    first_neurons[neur],
                                                                    att+1))
                    history = model.fit(X_train, y_train,
                                        validation_data=(X_val, y_val),
                                        epochs=epochs,
                                        batch_size=16,
                                        verbose=1
                                        )

                    # save the history data
                    # loss
                    detailed_tuning_data[af,lr,neur,att,:,0] = history.history['loss']
                    # accuracy
                    detailed_tuning_data[af,lr,neur,att,:,1] = history.history['accuracy']
                    # val_loss
                    detailed_tuning_data[af,lr,neur,att,:,2] = history.history['val_loss']
                    # val_accuracy
                    detailed_tuning_data[af,lr,neur,att,:,3] = history.history['val_accuracy']

                    # get the index where validation accuracy is highest
                    best_idx = history.history['val_accuracy'].index(max(history.history['val_accuracy']))

                    # accumulate all values of the 4 metrics at this index
                    avg_best_values = np.add(avg_best_values, [history.history['loss'][best_idx],
                                                               history.history['accuracy'][best_idx],
                                                               history.history['val_loss'][best_idx],
                                                               history.history['val_accuracy'][best_idx]])
                # average the values
                avg_best_values /= attempt

                # add average data to summary tuning array
                summary_tuning_data[af,lr,neur,:] = avg_best_values

     # save all ndarray data to files
    # save detailed tuning data
    np.save("hyperparameter-tuning/ntu_rgb_d_tuning_history_data/detailed_tuning_data(setting={},skips={}).npy".format(setting,skips),
            detailed_tuning_data)
    # save summary tuning data
    np.save("hyperparameter-tuning/ntu_rgb_d_tuning_history_data/summary_tuning_data(setting={},skips={}).npy".format(setting,skips),
            summary_tuning_data)


    pass

def plot_hyperparam_configuration_summary_performance(setting='cs',
                                                      skips= 2,
                                                      activation_fn= ('relu', 'sigmoid'),
                                                      learning_rate= (-5, -2, 4),
                                                      first_neurons= (100, 1000, 100),
                                                      metric= 'val_accuarcy'):
    #
    # build performance chart based on summary tuning data
    # Type: A line chart with Y-axis is accuracy or loss, Y-axis is number of first-neurons;
    #       Each line in the chart is corresponding with a specific combination of
    #       activation function and learning rate values
    # Input:
    #   -metric -a metric on which the performance is measured (loss,accuracy,val_loss,val_accuracy)
    # Output:
    #   -void() -show a chart which is appropriate with input params
    #

    # load the summary data from file
    summary_tuning_data = np.load("experiments/NN_hyperparameter-tuning/ntu_rgb_d_tuning_history_data/summary_tuning_data(setting={},skips={}).npy".format(setting,skips))

    # parse parameters
    # activation function
    n_af = len(activation_fn)
    # learning rate
    learning_rate = np.logspace(learning_rate[0], learning_rate[1], learning_rate[2])
    n_lr = len(learning_rate)
    # first neurons
    first_neurons = list(range(first_neurons[0], first_neurons[1] + 1, first_neurons[2]))
    n_neus = len(first_neurons)

    # check validity
    assert n_af == summary_tuning_data.shape[0]
    assert n_lr == summary_tuning_data.shape[1]
    assert n_neus == summary_tuning_data.shape[2]

    metric_idx = -1
    if metric == 'loss':
        metric_idx = 0
    elif metric == 'accuracy':
        metric_idx = 1
    elif metric == 'val_loss':
        metric_idx = 2
    elif metric == 'val_accuracy':
        metric_idx = 3

    assert metric_idx >= 0

    # plot the chart
    pyplot.figure(figsize=(10,7))
    pyplot.title("Hyper-parameter tuning- {}".format(metric))

    for af in range(n_af):
        for lr in range(n_lr):
            # build legend of each line [activation-func_learning-rate]
            legend = "{}, lr={}".format(activation_fn[af],learning_rate[lr])

            pyplot.plot(first_neurons, summary_tuning_data[af,lr,:,metric_idx], label= legend)

    pyplot.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=n_lr)
    pyplot.subplots_adjust(bottom=0.2)
    pyplot.grid()
    pyplot.show()

    pass


if __name__ == "__main__":
    print("Starting ntu-rgb-d hyperparam-tuning as entry point....")
    dir_path = os.path.dirname(os.path.abspath(__file__))

    action_codes = ["A050","A051","A052","A053","A054","A055","A056","A057","A058","A059","A060"]

    train_dataset = join(dir_path, 'dataset-NTU-RGB-D', 'train')
    test_dataset = join(dir_path, 'dataset-NTU-RGB-D', 'test')

    # =========================================================== #
    # ============ hyperparameter tuning ======================== #
    # =========================================================== #

    # ============ Hyperband method =============
    # optimal_model = do_Hyperband_and_get_optimal_model()

    # ============ Random Search method =============
    # optimal_model = do_RandomSearch_and_get_optimal_model()

    # ============ do tuning in a small search space ==============
    # should be fixed the search space for plotting chart properly
    activation_fn = ('relu','sigmoid')
    learning_rate = (-5, -2, 4)
    first_neurons = (100, 500, 100)

    attempt = 2
    epochs = 5

    # do_hyperparam_tuning_in_small_search_space(activation_fn= activation_fn,
    #                                            learning_rate= learning_rate,
    #                                            first_neurons= first_neurons,
    #                                            attempt= attempt,
    #                                            epochs= epochs
    #                                            )

    # plot statistical chart
    plot_hyperparam_configuration_summary_performance(activation_fn,
                                                      learning_rate,
                                                      first_neurons,
                                                      metric='val_accuracy')

    # =========================================================== #

    # =========================================================== #
    # ======= train model with optimal configuration  =========== #
    # =========================================================== #
    # train_NN_Classifier_with_specific_hyperparams(optimal_model)


    # =========================================================== #