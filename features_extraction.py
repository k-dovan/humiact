import sys
from sys import platform
import os
from os.path import join
import cv2
import argparse

import pandas as pd
import numpy as np
from preprocessing import normalize_keypoint_by_its_bounding_box

def estimate_bounding_box(poseKeypoints,
                          img_width,
                          img_height,
                          ratio_x,
                          ratio_y):
    #
    # estimate bounding box based on poseKeypoints data of an image
    # input:
    #   poseKeypoints - tuple of key points from openpose
    #   img_width - width of input image
    #   img_height - height of input image
    #   ratio_x - percentage of bounding box's width increase
    #   ratio_y - percentage of bounding box's height increase
    # output:
    #   combined bounding box estimated for all keypoints sets
    #   => actually maximum is 2 sets - two people
    #

    # each bounding box - x1,y1,x2,y2
    #   where (x1,y1) is top-left corner point coordinate
    #         (x2,y2) is right-bottom corner point coordinate

    # check validity
    if poseKeypoints.size < 2:
        return ()

    # concatenate two keypoints sets in order to estimate
    # bounding box for the two sets
    poseKeypoints = poseKeypoints.reshape(-1,3)
    MIN_BOX_SIZE = (16, 16)

    Xs = [item for item in poseKeypoints[:, 0] if item > 0]
    Ys = [item for item in poseKeypoints[:, 1] if item > 0]

    x1 = np.min(Xs)
    x2 = np.max(Xs)
    y1 = np.min(Ys)
    y2 = np.max(Ys)

    delta_width = (x2 - x1) * ratio_x / 2
    delta_height = (y2 - y1) * ratio_y / 2

    x1_new = (x1 - delta_width) if (x1 - delta_width) > 0 else 0
    y1_new = (y1 - delta_height) if (y1 - delta_height) > 0 else 0
    x2_new = (x2 + delta_width) if (x2 + delta_width) < img_width else img_width
    y2_new = (y2 + delta_height) if (y2 + delta_height) < img_height else img_height

    if (x2_new - x1_new) > MIN_BOX_SIZE[0] and (y2_new - y1_new) > MIN_BOX_SIZE[1]:
        return (int(x1_new), int(y1_new), int(x2_new), int(y2_new))
    else:
        return ()

def draw_bounding_box(datum):
    #
    # draw bounding box for testing classifier performance
    #

    # region estimate merged bounding boxes
    img_height, img_width, _ = datum.cvInputData.shape
    box = estimate_bounding_box(datum.poseKeypoints,
                                img_width,
                                img_height,
                                0.10,
                                0.10)
    # endregion

    # draw bounding boxes on output image
    bnb_img = datum.cvInputData

    cv2.rectangle(bnb_img,
                  (box[0], box[1]),
                  (box[2], box[3]),
                  color=(0, 0, 255),
                  thickness=3)

    return bnb_img


def extract_ROI_and_HOG_feature(datum,
                                image_name,
                                roi_and_hog_path ="",
                                isDisplayImageWithROIs= False,
                                isExtractRoiHogBitmap= False):
    #
    # extract regions of interest and appropriate HOG features of an image
    # input:
    #   - danum - the output data of the image processed by OpenPose
    #   - image_name - image name used to build roi and hog images
    #   - roi_and_hog_path - path for saving roi and hog data
    #   - isDisplayImageWithROIs - whether display ROIs
    #   - isExtractRoiHogBitmap - whether save ROIs and HOGs as bitmaps
    # output:
    #   list of HOG features of all bounding boxes of current image
    #

    # region estimate merged bounding boxes
    img_height, img_width, _ = datum.cvOutputData.shape
    box = estimate_bounding_box(datum.poseKeypoints,
                                img_width,
                                img_height,
                                0.10,
                                0.10)
    #endregion

    #region draw merged bounding boxes
    if isDisplayImageWithROIs:
        # draw bounding boxes on output image
        bnb_img = datum.cvOutputData

        if len(box) != 0:
            cv2.rectangle(bnb_img,
                          (box[0], box[1]),
                          (box[2], box[3]),
                          color=(0, 0, 255),
                          thickness=2)

        # display image with bounding boxes
        resized_bnb_img = cv2.resize(bnb_img, (960, 540))
        cv2.imshow("Merged bounding box is drawn.", resized_bnb_img)
        cv2.waitKey(0)
    # endregion

    #region extract hog features

    # check if estimated bounding box tuple is empty (too small -> ignored)
    # if so, then return empty HOG array feature
    if len(box) == 0:
        # return empty array of HOG feature of size of 1764 (42*42)
        return [0.0] * 1764

    roi_img = datum.cvInputData[box[1]:box[3], box[0]:box[2]]

    if isExtractRoiHogBitmap:
        # save roi_img to file
        roi_img_path = join(roi_and_hog_path, image_name + "_roi.png")
        cv2.imwrite(roi_img_path, cv2.resize(roi_img, (192,192)))

    # resize roi image to 64x64
    winSize = (64, 64) # might be 32x32 pixel
    resized_roi_img = cv2.resize(roi_img, winSize)

    # initialize HOG descriptor
    blockSize = (16, 16)
    blockStride = (8, 8)
    cellSize = (8, 8)
    nbins = 9
    derivAperture = 1
    winSigma = 4.
    histogramNormType = 0
    L2HysThreshold = 0.2
    gammaCorrection = 0
    nlevels = 64
    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma,
                            histogramNormType, L2HysThreshold, gammaCorrection, nlevels)

    # compute HOG features for resized ROI image
    hog_desc = hog.compute(resized_roi_img)

    if isExtractRoiHogBitmap:
        # reshape HOG feature as gray-scale image
        hog_desc_image = np.array(hog_desc.reshape(42, 42)) * 255
        # save hog descriptor to file as a gray-scale image
        hog_descriptor_path = join(roi_and_hog_path, image_name + "_hog.png")
        cv2.imwrite(hog_descriptor_path, hog_desc_image)

    # return list of HOG features of all bounding boxes of current image
    return hog_desc
    #endregion

def calc_means_stds_of_keypoints_set(set_of_keypoints):
    # calculate mean and std values of a keypoints set

    Xs = [item for item in set_of_keypoints[:, 0] if item > 0]
    Ys = [item for item in set_of_keypoints[:, 1] if item > 0]
    # mean and std of Xs
    meanX = np.mean(Xs)
    stdX = np.std(Xs)
    # mean and std of Ys
    meanY = np.mean(Ys)
    stdY = np.std(Ys)

    return ((meanX, meanY),(stdX, stdY))

# TODO: 1. merging job: takes mean, std, prob of Openpose keyoints into account
def keypoints_sets_merging(poseKeypoints,
                           ratio_of_intersec_thresh,
                           ratio_of_distance_thresh
                           ):
    # inputs:
    #   - poseKeypoints - list of sets of pose key points of an image
    #   - ratio_of_intersec_thresh - no. intersection points/ min no. 2 set of indexed keypoints
    #     we will merge the 2 sets if this ratio less than appropriate input param
    #   - ratio_of_distance_thresh - (delta mean)/(sum of std of 2 sets)
    #     we will merge the 2 sets if this ratio less than appropriate input param
    # outputs:
    #   - array of merged sets of keypoints

    # print("Bounding box merging module.")
    list_of_sets_of_keypoints_output = list()

    # calculate mean and std values of each keypoints set
    # check validity
    if poseKeypoints.size < 2:
        return np.array([])

    num_objs = len(poseKeypoints)
    list_of_sets_of_keypoints = list()
    # list of tuple of (mean,std) of each keypoints set
    list_of_means_stds = list()
    for i in range(0, num_objs):
        obj = poseKeypoints[i]
        list_of_sets_of_keypoints.append(obj)

        # calculate means and stds
        obj_stat = calc_means_stds_of_keypoints_set(obj)
        list_of_means_stds.append(obj_stat)

        # print("keyjoints set:" + str(obj) + "\n")
        # print("mean and std: " + str(obj_stat) + "\n")

    # do analysis for merging
    while len(list_of_sets_of_keypoints) > 0:
        # denote each set of poseKeypoints as a candidate
        # thus, list of sets of keypoints as list of candidates

        # if there is only one candidate, finalize the job by
        # adding this candidate to list of output
        if len(list_of_sets_of_keypoints) == 1:
            list_of_sets_of_keypoints_output.append(list_of_sets_of_keypoints.pop(0))
            list_of_means_stds.pop(0)
            break

        # if there are more than one candidate
        # get the first candidate
        fst_candidate = list_of_sets_of_keypoints.pop(0)
        ((fst_meanX, fst_meanY), (fst_stdX, fst_stdY)) = list_of_means_stds.pop(0)

        merged_idxs = []
        for cdt_idx in range(0,len(list_of_sets_of_keypoints)):
            # pick second candidate
            snd_candidate = list_of_sets_of_keypoints[cdt_idx]
            # appropriate means and stds
            ((snd_meanX, snd_meanY), (snd_stdX, snd_stdY)) = list_of_means_stds[cdt_idx]

            # try to merge the first and the second candidate
            fst_keypoints_indexes = [i for i in range(0, len(fst_candidate)) if (fst_candidate[i][0] > 0 or fst_candidate[i][1] > 0)]
            snd_keypoints_indexes = [i for i in range(0, len(snd_candidate)) if (snd_candidate[i][0] > 0 or snd_candidate[i][1] > 0)]

            # calculate ratio of intersection
            intersection_idxs = list(set(fst_keypoints_indexes).intersection(snd_keypoints_indexes))
            ratio_of_intersec = (float)(len(intersection_idxs))/min(len(fst_keypoints_indexes),len(snd_keypoints_indexes))
            if ratio_of_intersec < ratio_of_intersec_thresh:
                # calculate ratio of distance
                ratio_of_dist = (float)(np.abs(snd_meanX-fst_meanX))/(2*(fst_stdX+snd_stdX)) + (np.abs(snd_meanY-fst_meanY))/(2*(fst_stdY+snd_stdY))
                if ratio_of_dist < ratio_of_distance_thresh:
                    # both intersec ratio and distance ratio are satisfied
                    # then we merge the second candidate to the first one
                    for idx in snd_keypoints_indexes:
                        # if idx is intersection index
                        if idx in intersection_idxs:
                            # check and take data points with higher probability
                            if fst_candidate[idx][2] < snd_candidate[idx][2]:
                                fst_candidate[idx] = snd_candidate[idx]
                        else:
                            fst_candidate[idx] = snd_candidate[idx]

                    # re-calculate means and stds of the first candidate
                    ((fst_meanX, fst_meanY), (fst_stdX, fst_stdY)) = calc_means_stds_of_keypoints_set(fst_candidate)
                    # record merged index
                    merged_idxs.append(cdt_idx)

        # add the first candidate to result list
        list_of_sets_of_keypoints_output.append(fst_candidate)
        # update current sets of keypoints by removing merged candidates
        list_of_sets_of_keypoints = [list_of_sets_of_keypoints[i] for i in range(0,len(list_of_sets_of_keypoints)) if i not in merged_idxs]
        # update current sets of means and stds appropriately
        list_of_means_stds = [list_of_means_stds[i] for i in range(0,len(list_of_means_stds)) if i not in merged_idxs]

    # ending the merging
    # convert list to ndarray
    arr_of_keypoints_output = np.array(list_of_sets_of_keypoints_output)

    # replace unconfident points from big sets
    arr_of_keypoints_output = replace_unconfident_points_in_big_set_by_more_confident_small_sets(
                                                             poseKeypoints= arr_of_keypoints_output,
                                                             small_set_points_thresh= 4,
                                                             lse_thresh= 1.5,
                                                             confidence_thresh= 0.5,
                                                             unconfidence_thresh= 0.2,
                                                             removed_set_points_thresh= 3
                                                        )

    # print out the result
    # print ("The list of %d merged sets of keypoints:\n" % len(arr_of_keypoints_output))
    # print(arr_of_keypoints_output)

    return get_two_most_confident_sets(arr_of_keypoints_output)

def get_two_most_confident_sets(keypoints_arrs):
    #
    # This function is to get two most confident sets from the input arrays
    # In this case, the more non-zero keypoints in a set, the more confident it is
    #
    # Input:
    #   -keypoints_arrs - the refined arrays after mergring and eliminating
    # Output:
    #   -an array with maximum 2 sets of key poinnts (corresponding with two people)
    #

    n = len(keypoints_arrs)

    if n < 3:
        return keypoints_arrs

    # calculate a tupple ((index1,number_of_non-zero_keypoints),...) from the arrays
    index_value_pairs = ()
    for i in range(n):
        index_value_pairs += (i, len([idx for idx in range(len(keypoints_arrs[i])) if (keypoints_arrs[i][idx][0] > 0 or keypoints_arrs[i][idx][1] > 0)])),

    # sort this tupple
    sorted_index_value_pairs = sorted(index_value_pairs,key=lambda x:x[1],reverse=True)

    return np.append([keypoints_arrs[sorted_index_value_pairs[0][0]]], [keypoints_arrs[sorted_index_value_pairs[1][0]]],axis=0)

# TODO: 2. replace unconfident points by more confident ones in remaining small sets after merging
def replace_unconfident_points_in_big_set_by_more_confident_small_sets(poseKeypoints,
                                              small_set_points_thresh,
                                              lse_thresh,
                                              confidence_thresh,
                                              unconfidence_thresh,
                                              removed_set_points_thresh
                                              ):
    #
    # remove duplicated small sets of pose key points
    #
    # inputs:
    #   - poseKeypoints - array of pose keypoints after merging
    #   - small_set_points_thresh - a threshold determine whether a set is small or not
    #   - lse_thresh - least squared error threshold determine whether we consider
    #     a small set (A) as replacement of a subset of points (B) in a given  big set
    #     let's say, A and B are not too far from each other
    #   - confidence_thresh - the minimum avg of prob of small set we will consider
    #   - unconfidence_thresh - the maximum avg of prob of big set we will consider
    #   - removed_set_points_thresh - a threshold determine whether a set will be removed
    #     after all replacement jobs are done
    # output:
    #   - array of pose keypoints after removing duplicates
    #

    # check validity
    if poseKeypoints.size < 2:
        return np.array([])

    # build small sets indexes and the rest indexes
    small_sets_idxs = [idx for idx in range(0,len(poseKeypoints)) if np.sum(((poseKeypoints[idx][:,0]>0) | (poseKeypoints[idx][:,1]>0)) == True) <= small_set_points_thresh]
    big_sets_idxs = [idx for idx in range(0,len(poseKeypoints)) if idx not in small_sets_idxs]

    # the indexes of small sets are used for replacement
    replaced_small_set_idxs = []
    for small_set_idx in small_sets_idxs:
        # get small set instance
        sml_inst = poseKeypoints[small_set_idx]
        sml_keypoints_idxs = [keyp_idx for keyp_idx in range(0,len(sml_inst)) if (sml_inst[keyp_idx,0]>0 or sml_inst[keyp_idx,1]>0)]

        for big_set_idx in big_sets_idxs:
            # get big set instance
            big_inst = poseKeypoints[big_set_idx]

            # calculate least squared error of small set compared to appropriate elements in big set
            # ignore the calc if an element of the big one doesn't exist data at this index
            lse = 0.0
            count = 0
            for sml_kp_idx in sml_keypoints_idxs:
                big_kp_X = big_inst[sml_kp_idx, 0]
                big_kp_Y = big_inst[sml_kp_idx, 1]
                if big_kp_X > 0 or big_kp_Y > 0:
                    sml_kp_X = sml_inst[sml_kp_idx, 0]
                    sml_kp_Y = sml_inst[sml_kp_idx, 1]

                    # normalize the value by dividing by the std of the small set
                    ((_,_), (std_sml_X, std_sml_Y)) =  calc_means_stds_of_keypoints_set(sml_inst)
                    lse = lse + np.sqrt((sml_kp_X-big_kp_X)**2 + (sml_kp_Y-big_kp_Y)**2)/(std_sml_X+std_sml_Y)
                    count = count + 1

            if lse > 0:
                lse = lse/count
                # check if we can replace by the current small set
                if lse < lse_thresh:
                    # these two sets close enough to consider
                    # calculate the average probability of small set and big set
                    sml_set_prob_avg = np.average([pct for pct in sml_inst[sml_keypoints_idxs, 2]])
                    big_set_prob_avg = np.average([pct for pct in big_inst[sml_keypoints_idxs, 2]])

                    # check if they can be replaced
                    if sml_set_prob_avg > confidence_thresh and big_set_prob_avg < unconfidence_thresh:
                        # do the replacement
                        for sml_kp_idx in sml_keypoints_idxs:
                            big_inst[sml_kp_idx] = sml_inst[sml_kp_idx]
                        poseKeypoints[big_set_idx] = big_inst

                        # record the replaced index
                        replaced_small_set_idxs.append(small_set_idx)
                        break

    # indexes of sets of keypoints we can remove
    removed_set_points_idxs = [idx for idx in range(0,len(poseKeypoints)) if np.sum(((poseKeypoints[idx][:,0]>0) | (poseKeypoints[idx][:,1]>0)) == True) <= removed_set_points_thresh]

    # union removed sets indexes with replaced sets indexes
    combined_set_points_idxs = list(set(removed_set_points_idxs).union(set(replaced_small_set_idxs)))

    # rebuild the array of keypoints after removing duplicated small sets
    arr_of_keypoints_set_output = np.delete(poseKeypoints, combined_set_points_idxs, axis=0)

    return arr_of_keypoints_set_output

def extract_dataset_features(dataset,
                             isWithExtraFeature=True):
    #
    # extract keypoints features and HOG features from dataset
    # input:
    #   dataset - the path of dataset for features extracting
    # output:
    #   void - all features data are saved to disk
    #

    try:
        # region initialization for OpenPose to run
        # code from OpenPose library tutorial

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

        # get all training poses as sub-folders of train dataset
        categories = [f.name for f in os.scandir(join(dir_path, dataset)) if f.is_dir()]

        if len(categories) == 0:
            parent, category = os.path.split(dataset)
            dataset = parent
            categories = [category]

        for category in categories:
            # Get key-points features and image name and size from train dataset
            # declare variables hold features and labels
            mixed_record_details = [] # for mixed features
            kp_record_details = [] # for only keypoint features

            # build category path
            cat_path = join(dir_path, dataset, category)

            # create ROI and HOG directory
            roi_and_hog_path = ""
            if isWithExtraFeature:
                roi_and_hog_path = join(cat_path, "RoI-HOG")
                if not os.path.exists(roi_and_hog_path):
                    os.mkdir(roi_and_hog_path)

            # get all images from the category
            image_paths = op.get_images_on_directory(cat_path)

            # process all images
            for image_path in image_paths:
                print("\nCurrent image: " + image_path + "\n")

                # get image full name (with file extension)
                _, image_full_name = os.path.split(image_path)
                # image name without extension
                image_name = image_full_name[0:image_full_name.rindex('.')]

                # processed by OpenPose
                datum = op.Datum()
                imageToProcess = cv2.imread(image_path)
                height, width, channels = imageToProcess.shape
                datum.cvInputData = imageToProcess
                opWrapper.emplaceAndPop([datum])

                # print("Body keypoints: \n" + str(datum.poseKeypoints))

                # check if exists pose key points data
                if not (datum.poseKeypoints.size < 2):
                    # merging keypoints sets if applicable
                    merged_poseKeypoints = keypoints_sets_merging(datum.poseKeypoints,
                                                                  ratio_of_intersec_thresh= 0.36,
                                                                  ratio_of_distance_thresh= 2)

                    # assign merged_poseKeypoints to datum.poseKeypoints array
                    datum.poseKeypoints = merged_poseKeypoints

                    # an array to save keypoint feature
                    keyPoint_feats_arrs = []
                    if datum.poseKeypoints.size > 1 and datum.poseKeypoints.ndim == 3:
                        # region extract keypoints features
                        keypoints_coordinates = datum.poseKeypoints[:, :, :2]

                        # TODO: 3. after merging with new keypoints sets, these coordinates are translated to their center and scaled by their box size -> added in keypoints features
                        # translate and scale keypoints by its center and box size,
                        # and flattened it to 1D array
                        keyPoint_feats_arrs = normalize_keypoint_by_its_bounding_box(keypoints_coordinates)

                        # check if len of the list is only 50 corresponding with only one keypoint set
                        # append a zero array of size of 50 to this list
                        ONE_KEYPOINTS_SET_FEATURE_LENGTH = 50
                        if len(keyPoint_feats_arrs) == ONE_KEYPOINTS_SET_FEATURE_LENGTH:
                            keyPoint_feats_arrs += [0.0]*ONE_KEYPOINTS_SET_FEATURE_LENGTH

                        # add to list of keypoint-only features
                        kp_record_details.append((keyPoint_feats_arrs))

                        # endregion

                        if isWithExtraFeature:
                            # TODO: 4. RoI and HOG features are extracted -> added in HOG features
                            #region extract HOG features
                            extracted_hog_feats_arrs = \
                                extract_ROI_and_HOG_feature(datum,
                                                            image_name,
                                                            roi_and_hog_path,
                                                            isDisplayImageWithROIs=False,
                                                            isExtractRoiHogBitmap=True)

                            # append HOG feature to a combined feature array
                            keyPoint_HOG_feats_arrs = np.append(keyPoint_feats_arrs,
                                                                extracted_hog_feats_arrs)

                            # add mixed features data to accumulate array
                            mixed_record_details.append(keyPoint_HOG_feats_arrs)

                            #endregion

            # save merged, normalized features to disk
            if isWithExtraFeature:
                if len(mixed_record_details) > 0:
                    data1_to_write = pd.DataFrame(kp_record_details)
                    data2_to_write = pd.DataFrame(mixed_record_details)

                    data1_to_write.to_csv(join(cat_path, category + '_keypoint_feature.csv'))
                    data2_to_write.to_csv(join(cat_path, category + '_extra_feature.csv'))
            else:
                if len(kp_record_details) > 0:
                    data_to_write = pd.DataFrame(mixed_record_details)

                    data_to_write.to_csv(join(cat_path, category + '_keypoint_feature.csv'))

    except Exception as e:
        print(e)
        sys.exit(-1)

if __name__ == "__main__":
    print("Starting features_extraction.py as entry point....")
    dir_path = os.path.dirname(os.path.abspath(__file__))

    dataset_train = join(dir_path, 'dataset', 'train')
    dataset_test = join(dir_path, 'dataset', 'test')

    # extract keypoints features from train set
    extract_dataset_features(dataset_test,
                             isWithExtraFeature=True)

    # extract_dataset_features("dataset/train/Kissing",
    #                          isWithExtraFeature=True)

    # # TODO: 5. test error cases with merging, ROI images
    # extract_dataset_features("media/test",
    #                          isWithExtraFeature=True)