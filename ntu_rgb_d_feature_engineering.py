
# ======================================================= #
#
# Feature engineering for NTU-RGB-D dataset
#   - Compute feature vectors for each .npy file
#   - Split data into train/test sets (for cross-object and cross-view settings)
#   - Save features of each action in a separated file
#   - Using K-means to generate Codebook with key-pose vectors
#   - Compute histogram of key poses for each every sequence, and temporal pyramid
#   - Save accumulative feature data of each action to .csv file
#
# ======================================================= #
import numpy as np
import os
import re
import pandas as pd
import matplotlib.pyplot as plt

# path of .npy files
npy_path = "dataset-NTU-RGB-D/nturgb+d_npy/"
# get .npy files
cleaned_file_names = [file for file in os.listdir(npy_path) if file.endswith('clean.npy')]

# 11 classes of mutual actions (A050-A060)


def translate_skeleton(skeleton_data):
    #
    # Translate skeleton so that the middle of the spine
    # is the origin of the coordinates
    # Input:
    #   - skeleton_data -a (25,2)-shaped ndarray
    #   ID of middle of spine is 2, which is corresponding
    #   to index 1 of the array
    # Output:
    #   - result array after the translation is done
    #

    # get middle spine's coordinate
    middle_spine = skeleton_data[1]

    # apply element-wise subtraction, and return
    return np.subtract(skeleton_data, middle_spine)

    pass

def rotate_skeleton(skeleton_data):
    #
    # Rotate a skeleton so that the vector (spine->spine_base)
    # has the same direction with Y axis
    # Input:
    #   - skeleton_data -a (25,2)-shaped ndarray
    #   IDs of spine_base and spine are 1, 21 respectively
    #   that are corresponding with index 0 and 20 in the array
    # Output:
    #   - result array after the rotation is done
    #

    # calculate angle between vector (spine->spine_base) and Y axis
    coord_diff = skeleton_data[0] - skeleton_data[20]
    alpha = np.arctan2(coord_diff[0], coord_diff[1])

    # rotation matrix
    rot_mat = np.array([[np.cos(alpha), -np.sin(alpha)],
                       [np.sin(alpha), np.cos(alpha)]])

    # return rotated data
    return skeleton_data.dot(rot_mat.transpose())

    pass

def scale_skeleton(skeleton_data):
    #
    # Scale a skeleton by dividing skeleton points by box size
    # Input:
    #   - skeleton_data -a (25,2)-shaped ndarray
    # Output:
    #   - result array after the scaling is done
    #

    # compute bounding box size
    width = max(skeleton_data[:,0]) - min(skeleton_data[:,0])
    height = max(skeleton_data[:,1]) - min(skeleton_data[:,1])

    xs = skeleton_data[:, 0]
    ys = skeleton_data[:, 1]
    if width > 0:
        xs = skeleton_data[:,0]/width
    if height > 0:
        ys = skeleton_data[:,1]/height

    return np.append([xs],[ys],axis=0).transpose()

    pass

def extract_sequence_mode_rgb_skeleton_features(NoS=20):
    #
    # Extract RGB skeleton features from NTU_RGB_D dataset
    # and build train/test sets for 2 settings (cross-subject, cross-view), then
    # for each setting, save train and test set separately to train and test folder
    # Input:
    #   -NoS -number of frame samples picks from each sequence
    #   to create a feature vector with shape (1, NoS*(2*25*2+1)) = (1,NoS*101)
    #   Explain: the shape length equal to
    #   NoS*((2 skeletons)*(25-number of joints)*(2 dimensions XY)+(1 distance between 2 skeletons))
    # Output:
    #   -void() -save train/test sets for each of the two settings
    #   cross-subject:   cs_seq_train.csv, cs_seq_test.csv
    #   cross-view:      cv_seq_train.csv, cv_seq_test.csv
    #
    #TODO: fix NaN values in frames (use method df.fillna(...))

    # ================================================================================================ #
    # ======== cross-subject setting ==============
    # samples pattern: ____Pxxx___ ,where P stands for performer or subject
    # train subject IDS = [1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38]
    # test subject IDs = [3, 6, 7, 10, 11, 12, 20, 21, 22, 23, 24, 26, 29, 30, 32, 33, 36, 37, 39, 40]

    # ======== cross-view setting ==============
    # samples pattern: ____Cxxx___ ,where C stands for camera
    # train camera IDs = [2,3]
    # test camera ID = [1]
    # ================================================================================================ #

    # create regular expression to catch samples for train set, and for test set
    # cross-subject setting
    regex_cs_train = "(^[CS0-9]{8}P0(0[124589]|1[3456789]|2[578]|3[1458]).+)"
    regex_cs_test = "(^[CS0-9]{8}P0(0[367]|1[012]|2[0123469]|3[023679]|40).+)"

    # cross-view setting
    regex_cv_train = "(^[S0-9]{4}C00[23].+)"
    regex_cv_test = "(^[S0-9]{4}C001.+)"

    # regex to catch action code (A050-A060)
    regex_sample_action = "(A\d{3}).+"

    # declare variables to keep features for each setting and each train/test set and each class
    cs_train, cs_test, cv_train, cv_test = dict(),dict(),dict(),dict()
    # 11 classes in total (A050-A060)
    for i in range(50,61):
        action_code = "A0{}".format(i)
        cs_train[action_code] = []
        cs_test[action_code] = []
        cv_train[action_code] = []
        cv_test[action_code] = []

    # traverse through all samples, compute feature and assign to
    # corresponding sets

    # print message
    print ("The dataset is being processed...")

    for file in cleaned_file_names:

        # catch action code of the file
        matches = re.findall(regex_sample_action, file)
        if len(matches) == 0:
            continue
        action_code = matches[0]

        # full file name
        full_path = npy_path + file
        # load data from file as a dictionary
        bodysdata = np.load(full_path, allow_pickle=True).item()

        # compute features based on two skeletons data
        nFrames = len(bodysdata['rgb_bodys'][0])
        if nFrames < NoS:
            # too small frames included, skip this file
            continue

        # strategy to pick NoS samples from the sequence
        # - divide the sequence by NoS parts
        # - each part, randomly pick one frame
        # - finally, get NoS frames for the calculation
        base_eles = nFrames//NoS
        mod_eles = nFrames%NoS
        picked_idxs = [-1]*NoS
        range_list = [[]]*NoS
        idx = 0
        for i in range(mod_eles):
            range_list[i] = [idx, idx + base_eles]
            idx = idx + base_eles + 1

        for i in range(mod_eles, NoS):
            range_list[i] = [idx, idx + base_eles - 1]
            idx = idx + base_eles

        picked_idxs = np.random.randint([range[0] for range in range_list], [range[1]+1 for range in range_list])

        # frame_feat_len = (2 skeletons)*(25 key joints)*(2 dimensions XY) + (distance between 2 skeletons)
        frame_feat_len = 2*25*2+1
        # skeleton feature vector for the sequence
        cur_skel_feat_vector = [0]*NoS*frame_feat_len
        # variable holds max distance between two skeletons
        max_dist = 0
        for frame in range(NoS):
            # get 2 skeletons from current sample frame
            skeleton1 = bodysdata['rgb_bodys'][0][picked_idxs[frame]]
            skeleton2 = bodysdata['rgb_bodys'][1][picked_idxs[frame]]

            # distance between the two (middle spine 1 to middle spine 2)
            dist = np.sqrt(np.sum(np.square(skeleton2[1]-skeleton1[1])))
            if dist > max_dist:
                max_dist = dist

            # temporally save unnormalized distance
            cur_skel_feat_vector[frame*frame_feat_len + 100] = dist

            # normalize each skeleton by translation, rotation, and scaling
            skeleton1 = scale_skeleton(rotate_skeleton(translate_skeleton(skeleton1)))
            skeleton2 = scale_skeleton(rotate_skeleton(translate_skeleton(skeleton2)))

            # save normalized skeletons to feature vector
            cur_skel_feat_vector[frame*frame_feat_len:frame*frame_feat_len + 50] = skeleton1.reshape(1,-1).squeeze(axis=0)
            cur_skel_feat_vector[frame*frame_feat_len + 50:frame*frame_feat_len + 100] = skeleton2.reshape(
                1,-1).squeeze(axis=0)

        # normalize all distance values in the feature vector
        for frame in range(NoS):
            cur_skel_feat_vector[frame * frame_feat_len + 100] =\
                cur_skel_feat_vector[frame * frame_feat_len + 100]/max_dist

        # save to appropriate sets for each setting
        if re.match(regex_cs_train,file):
            cs_train[action_code].append(cur_skel_feat_vector)
        if re.match(regex_cs_test,file):
            cs_test[action_code].append(cur_skel_feat_vector)
        if re.match(regex_cv_train,file):
            cv_train[action_code].append(cur_skel_feat_vector)
        if re.match(regex_cv_test,file):
            cv_test[action_code].append(cur_skel_feat_vector)

    # save train/test sets to appropriate files
    cs_train_path = "dataset-NTU-RGB-D/cross-subject/train/seq_mod/"
    cs_test_path = "dataset-NTU-RGB-D/cross-subject/test/seq_mod/"
    cv_train_path = "dataset-NTU-RGB-D/cross-view/train/seq_mod/"
    cv_test_path = "dataset-NTU-RGB-D/cross-view/test/seq_mod/"
    for action_code in cs_train.keys():
        data_to_write = pd.DataFrame(cs_train[action_code])
        data_to_write.to_csv("{}{}.csv".format(cs_train_path, action_code))
    for action_code in cs_test.keys():
        data_to_write = pd.DataFrame(cs_test[action_code])
        data_to_write.to_csv("{}{}.csv".format(cs_test_path, action_code))
    for action_code in cv_train.keys():
        data_to_write = pd.DataFrame(cv_train[action_code])
        data_to_write.to_csv("{}{}.csv".format(cv_train_path, action_code))
    for action_code in cv_test.keys():
        data_to_write = pd.DataFrame(cv_test[action_code])
        data_to_write.to_csv("{}{}.csv".format(cv_test_path, action_code))

    # print message
    print("The dataset is processed successfully!")

    pass

def extract_frame_mode_rgb_skeleton_features(space=5):
    #
    # Extract RGB skeleton features from NTU_RGB_D dataset
    # and build train/test sets for 2 settings (cross-subject, cross-view), then
    # for each setting, save train and test set separately to train and test folder
    # Input:
    #   -space -the space (or distance) from the next selected frame to current selected one
    #   to create a feature vector with shape (1, (2*25*2)) = (1,100)
    #   Explain: the shape length equal to
    #   (2 skeletons)*(25-number of joints)*(2 dimensions XY)
    # Output:
    #   -void() -save train/test sets for each of the two settings
    #   cross-subject:   cs_frame_train.csv, cs_frame_test.csv
    #   cross-view:      cv_frame_train.csv, cv_frame_test.csv
    #

    # ================================================================================================ #
    # ======== cross-subject setting ==============
    # samples pattern: ____Pxxx___ ,where P stands for performer or subject
    # train subject IDS = [1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38]
    # test subject IDs = [3, 6, 7, 10, 11, 12, 20, 21, 22, 23, 24, 26, 29, 30, 32, 33, 36, 37, 39, 40]

    # ======== cross-view setting ==============
    # samples pattern: ____Cxxx___ ,where C stands for camera
    # train camera IDs = [2,3]
    # test camera ID = [1]
    # ================================================================================================ #

    # create regular expression to catch samples for train set, and for test set
    # cross-subject setting
    regex_cs_train = "(^[CS0-9]{8}P0(0[124589]|1[3456789]|2[578]|3[1458]).+)"
    regex_cs_test = "(^[CS0-9]{8}P0(0[367]|1[012]|2[0123469]|3[023679]|40).+)"

    # cross-view setting
    regex_cv_train = "(^[S0-9]{4}C00[23].+)"
    regex_cv_test = "(^[S0-9]{4}C001.+)"

    # regex to catch action code (A050-A060)
    regex_sample_action = "(A\d{3}).+"

    # declare variables to keep features for each setting and each train/test set and each class
    cs_train, cs_test, cv_train, cv_test = dict(), dict(), dict(), dict()
    # 11 classes in total (A050-A060)
    for i in range(50, 61):
        action_code = "A0{}".format(i)
        cs_train[action_code] = []
        cs_test[action_code] = []
        cv_train[action_code] = []
        cv_test[action_code] = []

    # traverse through all samples, compute feature and assign to
    # corresponding sets

    # print message
    print ("The dataset is being processed...")

    for file in cleaned_file_names:

        # catch action code of the file
        matches = re.findall(regex_sample_action, file)
        if len(matches) == 0:
            continue
        action_code = matches[0]

        # full file name
        full_path = npy_path + file

        # load data from file as a dictionary
        bodysdata = np.load(full_path, allow_pickle=True).item()

        # compute features based on two skeletons data
        nFrames = len(bodysdata['rgb_bodys'][0])
        if nFrames <= 0:
            continue

        # compute number of selected frames
        picked_size = (nFrames-1)//(space+1) + 1

        picked_idxs = [-1]*picked_size

        for i in range(picked_size):
            picked_idxs[i] = i*(space+1)

        # frame_feat_len = (2 skeletons)*(25 key joints)*(2 dimensions XY) + (1 distance)
        frame_feat_len = 2*25*2+1

        # list of skeleton feature vectors for picked frames
        skel_feat_vectors = []

        for frame in range(picked_size):
            cur_feat_vec = [0]*frame_feat_len

            # get 2 skeletons from current sample frame
            skeleton1 = bodysdata['rgb_bodys'][0][picked_idxs[frame]]
            skeleton2 = bodysdata['rgb_bodys'][1][picked_idxs[frame]]

            # if any skeleton of the frame has NaN values, ignore the frame
            if np.any(np.isnan(skeleton1)) or np.any(np.isnan(skeleton2)):
                continue

            # calculate the distance between the two skeletons
            dist = 0.0
            if (skeleton1.any() and skeleton2.any()):

                # calculate bounding boxes of the two skeletons
                width1, height1 = max(skeleton1[:,0]) - min(skeleton1[:,0]), max(skeleton1[:,1]) - min(skeleton1[:,1])
                width2, height2 = max(skeleton2[:,0]) - min(skeleton2[:,0]), max(skeleton2[:,1]) - min(skeleton2[:,1])
                # consider the distance between the two skeletons as
                # the distance from the two middle spines
                diff = skeleton2[1] - skeleton1[1]
                # normalized the diff
                diff = np.divide(diff,[width1+width2, height1+height2])
                dist = np.sqrt(np.sum(np.square(diff)))

            # normalize each skeleton by translation, rotation, and scaling
            skeleton1 = scale_skeleton(rotate_skeleton(translate_skeleton(skeleton1)))
            skeleton2 = scale_skeleton(rotate_skeleton(translate_skeleton(skeleton2)))

            # save normalized skeletons to feature vector
            cur_feat_vec[:50] = skeleton1.reshape(1,-1).squeeze(axis=0)
            cur_feat_vec[50:100] = skeleton2.reshape(1,-1).squeeze(axis=0)
            # add distance
            cur_feat_vec[100] = dist

            # append to list
            skel_feat_vectors.append(cur_feat_vec)

        # save to appropriate sets for each setting
        if re.match(regex_cs_train,file):
            for item in skel_feat_vectors:
                cs_train[action_code].append(item)
        if re.match(regex_cs_test,file):
            for item in skel_feat_vectors:
                cs_test[action_code].append(item)
        if re.match(regex_cv_train,file):
            for item in skel_feat_vectors:
                cv_train[action_code].append(item)
        if re.match(regex_cv_test,file):
            for item in skel_feat_vectors:
                cv_test[action_code].append(item)

    # save train/test sets to appropriate files
    cs_train_path = "dataset-NTU-RGB-D/extracted_features/frm_mod_space{}/cross-subject/train/".format(space)
    cs_test_path = "dataset-NTU-RGB-D/extracted_features/frm_mod_space{}/cross-subject/test/".format(space)
    cv_train_path = "dataset-NTU-RGB-D/extracted_features/frm_mod_space{}/cross-view/train/".format(space)
    cv_test_path = "dataset-NTU-RGB-D/extracted_features/frm_mod_space{}/cross-view/test/".format(space)
    for action_code in cs_train.keys():
        data_to_write = pd.DataFrame(cs_train[action_code])
        data_to_write.to_csv("{}{}.csv".format(cs_train_path, action_code))
    for action_code in cs_test.keys():
        data_to_write = pd.DataFrame(cs_test[action_code])
        data_to_write.to_csv("{}{}.csv".format(cs_test_path, action_code))
    for action_code in cv_train.keys():
        data_to_write = pd.DataFrame(cv_train[action_code])
        data_to_write.to_csv("{}{}.csv".format(cv_train_path, action_code))
    for action_code in cv_test.keys():
        data_to_write = pd.DataFrame(cv_test[action_code])
        data_to_write.to_csv("{}{}.csv".format(cv_test_path, action_code))

    # print message
    print("The dataset is processed successfully!")
    pass

if __name__ == "__main__":

    # # test first file
    # file1 = npy_path + cleaned_file_names[0]
    #
    # bodysdata = np.load(file1,allow_pickle=True).item()
    #
    # # get first skeleton of the first body
    # skeleton1 = bodysdata['rgb_bodys'][0][0]
    #
    # # translate
    # skeleton11 = translate_skeleton(skeleton1)
    # skeleton12 = rotate_skeleton(skeleton11)
    # skeleton13 = scale_skeleton(skeleton12)
    #
    #
    # plt.scatter(skeleton1[:,0], skeleton1[:,1], s=10, c='g', marker="*", label='org')
    # plt.scatter(skeleton11[:,0], skeleton11[:,1], s=10, c='b', marker="s", label='first')
    # plt.scatter(skeleton12[:,0], skeleton12[:,1], s=10, c='r', marker="o", label='second')
    # plt.legend(loc='upper left');
    # fig2 = plt.figure()
    # plt.scatter(skeleton13[:,0], skeleton13[:,1], s=10, c='c', marker="x", label='third')
    # plt.show()

    # ========================================
    # extract_sequence_mode_rgb_skeleton_features()
    # ========================================

    # ========================================
    extract_frame_mode_rgb_skeleton_features(space=8)
    # ========================================


    pass