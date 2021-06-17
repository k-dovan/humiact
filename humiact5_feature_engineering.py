import sys
from sys import platform
import os
from os.path import join
import cv2
import argparse

import pandas as pd
import numpy as np

def estimate_combined_bounding_box(poseKeypoints,
                                   img_width,
                                   img_height,
                                   ratio_x = 0.10,
                                   ratio_y = 0.15):
    #
    # estimate bounding box based on poseKeypoints data of an image
    # input:
    #   poseKeypoints - tuple of key points from openpose
    #   img_width - width of input image
    #   img_height - height of input image
    #   ratio_x - percentage of bounding box's width increase
    #   ratio_y - percentage of bounding box's height increase
    # output:
    #   combined bounding box estimated for keypoints sets
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

    # expand more at the top and less at the bottom
    weight_top = 0.8
    delta_height_top = delta_height*weight_top
    delta_height_bottom = delta_height*(1-weight_top)

    x1_new = (x1 - delta_width) if (x1 - delta_width) > 0 else 0
    y1_new = (y1 - delta_height_top) if (y1 - delta_height_top) > 0 else 0
    x2_new = (x2 + delta_width) if (x2 + delta_width) < img_width else img_width
    y2_new = (y2 + delta_height_bottom) if (y2 + delta_height_bottom) < img_height else img_height

    if (x2_new - x1_new) > MIN_BOX_SIZE[0] and (y2_new - y1_new) > MIN_BOX_SIZE[1]:
        return (int(x1_new), int(y1_new), int(x2_new), int(y2_new))
    else:
        return ()

def estimate_separated_bounding_boxes(poseKeypoints,
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
    #   list of bounding boxes estimated
    #

    # each bounding box - x1,y1,x2,y2
    #   where (x1,y1) is top-left corner point coordinate
    #         (x2,y2) is right-bottom corner point coordinate
    bxs = ()

    # check validity
    if poseKeypoints.size < 2:
        return bxs

    num_objs = len(poseKeypoints)
    MIN_BOX_SIZE = (16,16)

    for i in range(0, num_objs):
        obj = poseKeypoints[i]
        Xs = [item for item in obj[:,0] if item > 0]
        Ys = [item for item in obj[:,1] if item > 0]

        x1 = np.min(Xs)
        x2 = np.max(Xs)
        y1 = np.min(Ys)
        y2 = np.max(Ys)

        delta_width = (x2-x1)*ratio_x/2
        delta_height = (y2-y1)*ratio_y/2

        # expand more at the top and less at the bottom
        weight_top = 0.8
        delta_height_top = delta_height * weight_top
        delta_height_bottom = delta_height * (1 - weight_top)

        x1_new = (x1 - delta_width) if (x1 - delta_width) > 0 else 0
        y1_new = (y1 - delta_height_top) if (y1 - delta_height_top) > 0 else 0
        x2_new = (x2 + delta_width) if (x2 + delta_width) < img_width else img_width
        y2_new = (y2 + delta_height_bottom) if (y2 + delta_height_bottom) < img_height else img_height

        if (x2_new-x1_new) > MIN_BOX_SIZE[0] and (y2_new-y1_new) > MIN_BOX_SIZE[1]:
            bx = (int(x1_new), int(y1_new), int(x2_new), int(y2_new)),
            bxs += bx

    return bxs

def draw_combined_bounding_box(datum):
    #
    # draw bounding box for testing classifier performance
    #

    # region estimate merged bounding boxes
    img_height, img_width, _ = datum.cvInputData.shape
    box = estimate_combined_bounding_box(datum.poseKeypoints,
                                         img_width,
                                         img_height,
                                         0.10,
                                         0.10)
    # endregion

    # draw bounding boxes on output image
    bnb_img = datum.cvOutputData

    cv2.rectangle(bnb_img,
                  (box[0], box[1]),
                  (box[2], box[3]),
                  color=(0, 0, 255),
                  thickness=2)

    return bnb_img

def draw_separated_bounding_boxes(datum):
    #
    # draw bounding box for testing classifier performance
    #

    # region estimate merged bounding boxes
    img_height, img_width, _ = datum.cvInputData.shape
    bxs = estimate_separated_bounding_boxes(datum.poseKeypoints,
                                img_width,
                                img_height,
                                0.10,
                                0.10)
    # endregion

    # draw bounding boxes on output image
    bnb_img = datum.cvOutputData
    i = 1
    for bx in bxs:
        cv2.rectangle(bnb_img,
                      (bx[0], bx[1]),
                      (bx[2], bx[3]),
                      color=(0, 0, 255),
                      thickness=2)
        i = i + 1

    return bnb_img

def show_and_save_images_with_drawn_bounding_boxes(datum,image_name, saved_path, show=False, save=False):
    # draw bounding boxes on output image
    bnb_img = draw_separated_bounding_boxes(datum)

    # display image with bounding boxes
    # show resized size (if exist high resolution images in dataset)
    # resized_bnb_img = cv2.resize(bnb_img, (960, 540))
    # cv2.imshow("Merged bounding box is drawn.", resized_bnb_img)

    if show:
        # show original size
        cv2.imshow("Image with bounding boxes drawn.", bnb_img)
        cv2.waitKey(0)

    if save:
        # save the performance of the merging and elimination algorithm
        # just enable when checking performance of the algorithm
        new_image_name = image_name + "_boxes.jpg"

        # There might be more than one file with the same name but extension in the dataset
        # if it was the case, these files must save with different names
        # first try
        if not os.path.exists(join(saved_path,new_image_name)):
            cv2.imwrite(join(saved_path,new_image_name), bnb_img, [cv2.IMWRITE_JPEG_QUALITY, 50])
        else: # consecutive tries
            # attempt another name
            attempt = 1
            while True:
                extra_symbol = attempt + 1
                new_image_name = "{}_{}_boxes.jpg".format(image_name, extra_symbol)
                if not os.path.exists(join(saved_path, new_image_name)):
                    cv2.imwrite(join(saved_path, new_image_name), bnb_img, [cv2.IMWRITE_JPEG_QUALITY, 50])
                    break
                attempt += 1

def extract_ROI_and_HOG_feature(datum,
                                image_name,
                                roi_and_hog_path ="",
                                winSize = (64,64),
                                isExtractRoiHogBitmap= False):
    #
    # extract regions of interest and appropriate HOG features of an image
    #
    # input:
    #   - danum - the output data of the image processed by OpenPose
    #   - image_name - image name used to build roi and hog images
    #   - roi_and_hog_path - path for saving roi and hog data
    #   - winSize - resize roi image to winSize
    #   - isDisplayImageWithROIs - whether display ROIs
    #   - isExtractRoiHogBitmap - whether save ROIs and HOGs as bitmaps
    # output:
    #   list of HOG features of a bounding box of current image
    #

    # region Estimate merged bounding boxes for HOG computation
    img_height, img_width, _ = datum.cvOutputData.shape
    box = estimate_combined_bounding_box(datum.poseKeypoints,
                                         img_width,
                                         img_height,
                                         0.10,
                                         0.15)
    #endregion

    #region extract hog features

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

    # manually calculate size of HOG descriptor (feature vector size)
    block_steps_horizontal = (winSize[0] - blockSize[0]) / blockStride[0] + 1
    block_steps_vertical = (winSize[1] - blockSize[1]) / blockStride[1] + 1
    block_steps =  block_steps_horizontal*block_steps_vertical
    cell_steps_horizontal = blockSize[0] / cellSize[0]
    cell_steps_vertical = blockSize[1] / cellSize[1]
    cell_steps =  cell_steps_horizontal* cell_steps_vertical
    hog_desc_size = int(block_steps * cell_steps * nbins)

    # check if estimated bounding box tuple is empty (too small -> ignored)
    # if so, then return empty HOG array feature
    if len(box) == 0:
        # return empty array of HOG feature
        return [0.0] * hog_desc_size


    roi_img = datum.cvInputData[box[1]:box[3], box[0]:box[2]]

    if isExtractRoiHogBitmap:
        # save roi_img to file
        roi_image_name = image_name + "_roi.jpg"
        # There might be more than one file with the same name but extension in the dataset
        # if it was the case, these files must save with different names
        # first try
        if not os.path.exists(join(roi_and_hog_path, roi_image_name)):
            cv2.imwrite(join(roi_and_hog_path, roi_image_name), cv2.resize(roi_img, (192,192)), [cv2.IMWRITE_JPEG_QUALITY, 50])
        else:  # consecutive tries
            # attempt another name
            attempt = 1
            while True:
                extra_symbol = attempt + 1
                roi_image_name = "{}_{}_roi.jpg".format(image_name, extra_symbol)
                if not os.path.exists(join(roi_and_hog_path, roi_image_name)):
                    cv2.imwrite(join(roi_and_hog_path, roi_image_name), cv2.resize(roi_img, (192,192)), [cv2.IMWRITE_JPEG_QUALITY, 50])
                    break
                attempt += 1

    # resize roi image to winSize (64x64 or 32x32)
    resized_roi_img = cv2.resize(roi_img, winSize)

    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma,
                            histogramNormType, L2HysThreshold, gammaCorrection, nlevels)

    # compute HOG features for resized ROI image
    hog_desc = hog.compute(resized_roi_img)

    if isExtractRoiHogBitmap:
        # reshape HOG feature as gray-scale image
        # assume nbins is divisible by 3
        assert nbins%3 == 0
        hog_desc_image_width = int(3*block_steps_horizontal*cell_steps_horizontal)
        hog_desc_image_height = int((nbins/3)*block_steps_vertical*cell_steps_vertical)

        hog_desc_image = np.array(hog_desc.reshape(hog_desc_image_width, hog_desc_image_height)) * 255

        # save hog descriptor to file as an image
        hog_image_name = image_name + "_hog_" + str(winSize[0])\
                                    + "x" + str(winSize[1]) + ".jpg"
        # There might be more than one file with the same name but extension in the dataset
        # if it was the case, these files must save with different names
        # first try
        if not os.path.exists(join(roi_and_hog_path, hog_image_name)):
            cv2.imwrite(join(roi_and_hog_path, hog_image_name), hog_desc_image, [cv2.IMWRITE_JPEG_QUALITY, 50])
        else:  # consecutive tries
            # attempt another name
            attempt = 1
            while True:
                extra_symbol = attempt + 1
                hog_image_name = "{}_{}_hog_{}x{}.jpg".format(image_name, extra_symbol,winSize[0],winSize[1])
                if not os.path.exists(join(roi_and_hog_path, hog_image_name)):
                    cv2.imwrite(join(roi_and_hog_path, hog_image_name), hog_desc_image, [cv2.IMWRITE_JPEG_QUALITY, 50])
                    break
                attempt += 1

    # return list of HOG features of all bounding boxes of current image
    return hog_desc
    #endregion

def calc_means_stds_of_keypoints_set(one_set_of_keypoints):
    # calculate mean and std values of a keypoints set

    Xs = [item for item in one_set_of_keypoints[:, 0] if item > 0]
    Ys = [item for item in one_set_of_keypoints[:, 1] if item > 0]
    # mean and std of Xs
    meanX = np.mean(Xs)
    stdX = np.std(Xs)
    # mean and std of Ys
    meanY = np.mean(Ys)
    stdY = np.std(Ys)

    return ((meanX, meanY),(stdX, stdY))

# merging job: takes mean, std, prob of Openpose keyoints into account
def keypoints_sets_merging(poseKeypoints,
                           ratio_of_intersec_thresh=0.36,
                           ratio_of_distance_thresh=2
                           ):
    # inputs:
    #   - poseKeypoints - list of sets of pose key points of an image
    #   - ratio_of_intersec_thresh - no. intersection points/ min no. 2 set of indexed keypoints
    #     we will merge the 2 sets if this ratio less than appropriate input param
    #   - ratio_of_distance_thresh - (delta mean)/(sum of std of 2 sets)
    #     we will merge the 2 sets if this ratio less than appropriate input param
    # outputs:
    #   - array of at most 2 merged sets of keypoints

    # print("Keypoint sets merging module.")
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
    # arr_of_keypoints_output = replace_unconfident_points_in_big_set_by_more_confident_small_sets(
                                                             # poseKeypoints= arr_of_keypoints_output,
                                                             # small_set_points_thresh= 4,
                                                             # lse_thresh= 1.5,
                                                             # confidence_thresh= 0.5,
                                                             # unconfidence_thresh= 0.2,
                                                             # removed_set_points_thresh= 3
                                                        # )

    # print out the result
    # print ("The list of %d merged sets of keypoints:\n" % len(arr_of_keypoints_output))
    # print(arr_of_keypoints_output)

    return get_two_sets_with_biggest_variance(arr_of_keypoints_output)

# replace unconfident points by more confident ones in remaining small sets after merging
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

    # union indexes of removed sets with indexes of replaced sets
    combined_set_points_idxs = list(set(removed_set_points_idxs).union(set(replaced_small_set_idxs)))

    # rebuild the array of keypoints after removing duplicated small sets
    arr_of_keypoints_set_output = np.delete(poseKeypoints, combined_set_points_idxs, axis=0)

    return arr_of_keypoints_set_output

#TODO: adjust the function to take variace of a set into account. Prioritize sets with bigger variance.
def get_two_sets_with_biggest_variance(keypoints_arrs):
    #
    # This function is to get two sets with the biggest variance from the input arrays and with
    # the main actor at index 0 of the output array
    # Note: bigger variance means the skeleton is closer from the camera (two actors of the scene)
    # One main actor would be defined as following:
    # If the 2 sets have the same number of key-points, take the lower variance one
    # If not, take the one which has bigger number of key-points
    #
    # Input:
    #   -keypoints_arrs - the refined arrays after merging and eliminating (at least 5 non-zero joints)
    # Output:
    #   -an array of two skeletons with main actor at index 0
    #

    n = len(keypoints_arrs)

    if n < 3:
        return keypoints_arrs

    # calculate a tupple ((index1,variance of the set calculated by F1_score formula),...) from the arrays
    index_value_pairs = ()
    for i in range(n):
        # calculate the std of current set
        (_, stdXY) = calc_means_stds_of_keypoints_set(keypoints_arrs[i])
        # std = np.sqrt(stdXY[0]**2 + stdXY[1]**2)

        # calculate scalar std using F1_score formula
        std_F1 = stdXY[0]*stdXY[1]/(stdXY[0]+stdXY[1])
        index_value_pairs += (i, std_F1),

    # sort this tuple by variance
    sorted_index_value_pairs = sorted(index_value_pairs,key=lambda x:x[1],reverse=True)

    # get two first sets (two main actors of the scene)
    actor1 = keypoints_arrs[sorted_index_value_pairs[0][0]]
    actor2 = keypoints_arrs[sorted_index_value_pairs[1][0]]

    # calculate number of key-points for each actor
    noKP_actor1 = len([i for i in range(len(actor1)) if (actor1[i][0] > 0 or actor1[i][1] > 0)])
    noKP_actor2 = len([i for i in range(len(actor2)) if (actor2[i][0] > 0 or actor2[i][1] > 0)])

    if noKP_actor1 <= noKP_actor2:
        return np.stack((actor2,actor1))
    else:
        return np.stack((actor1, actor2))

def extract_relative_dist_orient_between_two_sets(keypoints_coordinates):

    # raise error if number of sets > 2 people
    assert keypoints_coordinates.shape[0] < 3

    extracted_data = [0] * (2*keypoints_coordinates.shape[1] ** 2)
    # if only one person
    if keypoints_coordinates.shape[0] == 1:
        return extracted_data

    # if two people
    set1 = keypoints_coordinates[0]
    set2 = keypoints_coordinates[1]
    no_points = keypoints_coordinates.shape[1]
    set2_col0_zero_mask = set2[:,0] == 0
    set2_col1_zero_mask = set2[:,1] == 0

    for i in range(no_points):
        cur_point = set1[i]
        if cur_point[0] == 0 and cur_point[1] == 0:
            extracted_data[i*2*no_points:(i+1)*2*no_points] = [0]*2*no_points
        else:
            set2 = np.copy(keypoints_coordinates[1])
            set2[set2_col0_zero_mask&set2_col1_zero_mask,0] = cur_point[0]
            set2[set2_col0_zero_mask&set2_col1_zero_mask,1] = cur_point[1]

            # subtract current point from set2
            subtracted_set = set2 - cur_point

            # calculate distances and orientations for subtracted_set
            # the distances
            dist_set = np.sqrt(np.sum(np.square(subtracted_set), axis=1))
            # normalize distances
            non_zero_set = dist_set[dist_set!=0]
            mean_val = np.mean(non_zero_set)
            max_val = np.max(non_zero_set)
            min_val = np.min(non_zero_set)
            dist_set[dist_set!=0] = (dist_set[dist_set!=0] - mean_val)/(max_val-min_val)

            # assign distances data to extracted_data
            extracted_data[i*2*no_points:(i*2*no_points + no_points)] = dist_set

            # the orientations
            orient_set = np.arctan2(subtracted_set[:,1],subtracted_set[:,0])
            non_zero_set = orient_set[orient_set!=0]
            mean_val = np.mean(non_zero_set)
            max_val = np.max(non_zero_set)
            min_val = np.min(non_zero_set)
            orient_set[orient_set!=0] = (orient_set[orient_set!=0] - mean_val) / (max_val - min_val)

            # assign orientations data to extracted_data
            extracted_data[(i*2*no_points + no_points):(i+1)*2*no_points] = orient_set

    # return extracted data (normalized distances and orientations)
    return extracted_data

def engineer_keypoint_based_feature_vector(keypoints_coordinates, spine_based=True):
    #
    # Each keypoint set will be translated to its centers and scaled by bounding box sizes.
    # Before doing the above translation and scaling, the distance between the two set is calculated,
    # and then the distance will be appended to the flatted feature vector.
    # Thus the length of the feature vector is (100+1)=101
    #
    # Input:
    #   -keypoints_coordinates - original pose keypoint coordinates
    #   with the shape= (2,25,2) or (1,25,2)
    #   -spine_mode - if True, data will be translated to middle of spine,
    #   and the distance dimension between two skeletons will be calculated based
    #   on the two middle-of-spine points. If False, data will be translated to
    #   mean of all points and all key-points will be involved for the calculation
    # Output:
    #   -keypoints_out - list of translated, scaled and flattened keypoint coordinates +
    #   the a distance between the two sets at the last index of the vector
    #

    # check validity
    if keypoints_coordinates.size < 2:
        return ()

    list_of_translated_scaled_keypoints = list()

    # two mean points of the two sets (skeletons)
    two_mean_points = []
    # two bounding box sizes of the two sets
    two_bnb_sizes = []
    
    # two middle of spine points that is the middle point of
    # two points (Neck-1 and MidHip-8)
    two_middle_of_spine_points = []
    # two lengths of spine of the two skeletons
    two_spine_lengths = []
    
    for keypoint_set in keypoints_coordinates:
        # calculate its center and box size
        Xs = [item for item in keypoint_set[:, 0] if item > 0]
        Ys = [item for item in keypoint_set[:, 1] if item > 0]

        if len(Xs)==0 or len(Ys)==0:
            continue

        xMin = np.min(Xs)
        xMax = np.max(Xs)
        yMin = np.min(Ys)
        yMax = np.max(Ys)

        center = ((xMax + xMin)/2, (yMax + yMin)/2)
        box_size = (xMax - xMin, yMax - yMin)

        # if the keyjoint set's points are on a line, or center of box=(0,0), ignore the set
        if (box_size[0] == 0 or box_size[1] ==0 or center[0] == 0 and center[1] ==0):
            continue
    
        # add non-zero mean point
        two_mean_points.append(center)
        # add non-zero bounding boxe
        two_bnb_sizes.append(box_size)
        
        # calculate middle point and the lenght of spine
        neck_point = keypoint_set[1,:]
        midhip_point = keypoint_set[8,:]

        middle_of_spine = (0.0, 0.0)
        length_of_spine = 0.0
        if np.all(neck_point) and np.all(midhip_point):
            middle_of_spine = ((neck_point[0]+midhip_point[0])/2, (neck_point[1]+midhip_point[1])/2)
            diff_neck_hip = (neck_point[0]-midhip_point[0], neck_point[1]-midhip_point[1])
            length_of_spine = np.sqrt(diff_neck_hip[0]**2 + diff_neck_hip[1]**2)

        # save middle of spine coordinate and the length
        # if not exists, all values = 0.0
        two_middle_of_spine_points.append(middle_of_spine)
        two_spine_lengths.append(length_of_spine)

        # zip normalized (x,y) coordinate
        # translate to middle of spine if it has non-zero coordinate
        if spine_based:
            if middle_of_spine[0] != 0 and middle_of_spine[1] != 0:
                normalized_coordinates = np.array([(item[0],item[1]) if (item[0]==0 or item[1]==0) else ((item[0]-middle_of_spine[0])/box_size[0],(item[1]-middle_of_spine[1])/box_size[1]) for item in keypoint_set])
            else:
                normalized_coordinates = np.array([(item[0],item[1]) if (item[0]==0 or item[1]==0) else ((item[0]-center[0])/box_size[0],(item[1]-center[1])/box_size[1]) for item in keypoint_set])
        else:
            normalized_coordinates = np.array([(item[0], item[1]) if (item[0] == 0 or item[1] == 0) else (
            (item[0] - center[0]) / box_size[0], (item[1] - center[1]) / box_size[1]) for item in keypoint_set])

        # flatten the normalized coordinates to feature vector
        list_of_translated_scaled_keypoints.append(normalized_coordinates.reshape(-1))

    # reshape the list to array
    list_of_translated_scaled_keypoints = [item for sublist in list_of_translated_scaled_keypoints \
                                           for item in sublist]

    # check if len of the list is only 50 corresponding with only one keypoint set
    # append a zero array of size of 50 to this list
    ONE_KEYPOINTS_SET_FEATURE_LENGTH = 50
    if len(list_of_translated_scaled_keypoints) == ONE_KEYPOINTS_SET_FEATURE_LENGTH:
        list_of_translated_scaled_keypoints += [0.0] * ONE_KEYPOINTS_SET_FEATURE_LENGTH
        # add zero distance to the end of this feature vector
        list_of_translated_scaled_keypoints += [0.0]
    else: # the vector feature is of length 100
        # calculate the distance and add to the end
        # Formula to calculate the distance:
        # dist = sqrt(((x2-x1)/(w1+w2))**2 + ((y2-y1)/(h1+h2))**2)
        dist = 0.0
        spine_based_factor = 0.1
        allpoint_factor = 0.33
        if spine_based:
            # check if two middle of spine points and the lengths have only non-zero values
            # if it is the case, the distance = segment(midP1-midP2)/(length of spine of main actor)
            # the first key-point set should be the main actor of the scene after merging process
            if np.all(two_middle_of_spine_points) and np.all(two_spine_lengths):
                diff_2midP = (two_middle_of_spine_points[0][0]-two_middle_of_spine_points[1][0],\
                              two_middle_of_spine_points[0][1]-two_middle_of_spine_points[1][1])
                dist_2midP = np.sqrt(diff_2midP[0]**2 + diff_2midP[1]**2)

                dist = spine_based_factor*dist_2midP/two_spine_lengths[0]

            # if middle of spines and the lengths are all zero, then
            # calculate the distance as torso_mod=False, that means using mean points and bounding
            # boxes of all key-points to calculate
            elif not np.any(two_middle_of_spine_points) and not np.any(two_spine_lengths):
                # the distance will be calculated as the following:
                # dist = (distance between two mean points)/(the biggest value from all box size values)
                #      = segment(meanP1-meanP2)/ max(w1,h1,w2,h2)
                diff_2meanP = (
                two_mean_points[0][0] - two_mean_points[1][0], two_mean_points[0][1] - two_mean_points[1][1])
                dist_2meanP = np.sqrt(diff_2meanP[0] ** 2 + diff_2meanP[1] ** 2)
                max_box_size_value = np.max(two_bnb_sizes)

                dist = allpoint_factor * dist_2meanP / max_box_size_value

            # if only one middle-of-spine point and one length are non-zero
            # then, the distance will be calculated as the distance from the non-zero
            # middle-of-spine point to the mean point of the remaining set, and scaled by
            # the non-zero length of spine
            else:
                # find index of non-zero middle-of-spine point and length values
                # init default value
                spine_based_index = 0
                mean_based_index = 1
                if two_spine_lengths[0] == 0:
                    spine_based_index = 1
                    mean_based_index = 0

                # distance from middle-of-spine point to the mean point
                diff_mid2mean = (two_middle_of_spine_points[spine_based_index][0]- two_mean_points[mean_based_index][0], two_middle_of_spine_points[spine_based_index][1]- two_mean_points[mean_based_index][1])
                dist_mid2mean = np.sqrt(diff_mid2mean[0]**2 + diff_mid2mean[1]**2)

                dist = spine_based_factor*dist_mid2mean/two_spine_lengths[spine_based_index]

        else:
            # the distance will be calculated as the following:
            # dist = (distance between two mean points)/(the biggest value from all box size values)
            #      = segment(meanP1-meanP2)/ max(w1,h1,w2,h2)
            diff_2meanP = (two_mean_points[0][0]-two_mean_points[1][0], two_mean_points[0][1]-two_mean_points[1][1])
            dist_2meanP = np.sqrt(diff_2meanP[0]**2 + diff_2meanP[1]**2)
            max_box_size_value = np.max(two_bnb_sizes)
            dist = allpoint_factor*dist_2meanP/max_box_size_value
            
        list_of_translated_scaled_keypoints += [dist]

    return list_of_translated_scaled_keypoints

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

        # RoI-HOG folder name if generated by the function already in this image folder
        roi_hog_folder_name = "RoI-HOG"
        # get all training poses as sub-folders of train dataset
        categories = [f.name for f in os.scandir(join(dir_path, dataset)) if f.is_dir() and f.name != roi_hog_folder_name]

        if len(categories) == 0:
            parent, category = os.path.split(dataset.rstrip('/\\'))
            dataset = parent
            categories = [category]

        for category in categories:
            # Get key-points features and image name and size from train dataset
            # declare variables hold features and labels
            kp_record_details = [] # for only keypoint features
            kp_do_record_details = [] # for keypoint+distance+orientation features
            mixed_record_details = [] # for mixed features
            winSize = (32,32) # winSize in the case of using HOG feature

            # build category path
            cat_path = join(dir_path, dataset, category)

            # create ROI and HOG directory
            roi_and_hog_path = ""
            roi_and_hog_path = join(cat_path, roi_hog_folder_name)
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
                    keypoint_based_feats_vector = []
                    if datum.poseKeypoints.size > 1 and datum.poseKeypoints.ndim == 3:

                        # ================================================================================= #
                        # ============== show and save images with drawn bounding boxes =================== #
                        show_and_save_images_with_drawn_bounding_boxes(datum, image_name, roi_and_hog_path,show=False, save=True)
                        # ================================================================================= #

                        # region Extract key-point features
                        keypoints_coordinates = datum.poseKeypoints[:, :, :2]

                        # calculate distances and orientations (if exist) between
                        # each every pair of points (each from each set)
                        # NOTE: consider each set corresponding with one person
                        # extracted_dists_orients = \
                        #     extract_relative_dist_orient_between_two_sets(keypoints_coordinates)

                        # after merging with new keypoints sets, (these coordinates are translated to their centers
                        # and scaled by their box sizes) + (distance between two skeletons) -> added in feature vectors
                        keypoint_based_feats_vector = engineer_keypoint_based_feature_vector(keypoints_coordinates)

                        # add to list of keypoint-only features
                        kp_record_details.append(keypoint_based_feats_vector)

                        # ===================== distances+orientations features ===================== #
                        # save keypoints+distances+orientations features
                        # kp_do_record_details.append(keypoint_based_feats_vector + extracted_dists_orients)
                        # =========================================================================== #

                        # endregion

                        if isWithExtraFeature:
                            # RoI and HOG features are extracted -> added in HOG features
                            #region extract HOG features
                            extracted_hog_feats_arrs = \
                                extract_ROI_and_HOG_feature(datum,
                                                            image_name,
                                                            roi_and_hog_path,
                                                            winSize= winSize,
                                                            isExtractRoiHogBitmap=True)

                            # append HOG feature to a combined feature array
                            keyPoint_HOG_feats_arrs = np.append(keypoint_based_feats_vector,
                                                                extracted_hog_feats_arrs)

                            # add mixed features data to accumulate array
                            mixed_record_details.append(keyPoint_HOG_feats_arrs)

                            #endregion

            # save merged, normalized features to disk
            if isWithExtraFeature:
                if len(mixed_record_details) > 0:
                    data1_to_write = pd.DataFrame(kp_record_details)
                    # data2_to_write = pd.DataFrame(kp_do_record_details)
                    data3_to_write = pd.DataFrame(mixed_record_details)

                    data1_to_write.to_csv(join(cat_path, category + '_keypoint_feature.csv'))
                    # data2_to_write.to_csv(join(cat_path, category + '_keypoint_do_feature.csv'))
                    data3_to_write.to_csv(join(cat_path, category + '_extra_feature_' +
                                                         str(winSize[0])+'x'+str(winSize[1]) +'.csv'))
            else:
                if len(kp_record_details) > 0:
                    data1_to_write = pd.DataFrame(kp_record_details)
                    # data2_to_write = pd.DataFrame(kp_do_record_details)

                    data1_to_write.to_csv(join(cat_path, category + '_keypoint_feature.csv'))
                    # data2_to_write.to_csv(join(cat_path, category + '_keypoint_do_feature.csv'))

    except Exception as e:
        print(e)
        sys.exit(-1)

if __name__ == "__main__":
    print("Starting features_extraction.py as entry point....")
    dir_path = os.path.dirname(os.path.abspath(__file__))

    extract_dataset_features("dataset-humiact5/test/",
                            isWithExtraFeature=False)

    # extract_dataset_features("experiments/testing/",
                              # isWithExtraFeature=False)