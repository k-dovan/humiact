
from os import listdir
from os.path import isfile, isdir, join
import cv2
import pandas as pd
import numpy as np

def get_all_images_recursively(path):
    #
    # get all images recursively from a given directory
    # input:
    #   path - the directory to get images
    # output:
    #   images - list of all image paths in the directory
    #

    search_dirs = [path]
    images = []
    while len(search_dirs) > 0:
        # pop first candidate for searching
        cur_dir = search_dirs.pop(0)

        images += [join(cur_dir, f) for f in listdir(cur_dir) if isfile(join(cur_dir, f)) and f.lower().endswith((
            '.png', '.jpg', '.jpeg', '.bmp'))]
        search_dirs += [join(cur_dir,f) for f in listdir(cur_dir) if isdir(join(cur_dir,f))]

    return images

def reduce_large_size_images(path):

    # Reduce every large size images in a directory recursively
    # input:
    #   path - dataset directory
    # output:
    #   void - all resized images are saved on the disk

    # set resolution threshold
    MAX_WIDTH = 1920
    MAX_HEIGHT = 900

    # get all images from dataset path
    images = get_all_images_recursively(path)

    # reduce large-size images to get CUDNN
    # running well on old model graphic card
    for image in images:
        img = cv2.imread(image)
        if img is not None:
            scaleW, scaleH = 1,1
            height, width, _ = img.shape
            if width > MAX_WIDTH:
                scaleW = width/MAX_WIDTH + 1
            if height > MAX_HEIGHT:
                scaleH = height / MAX_HEIGHT + 1

            scale = max(scaleW, scaleH)
            if scale > 1:
                width = int(width/scale)
                height = int(height/scale)
                dim = (width, height)

                # reduce image to the smaller size
                resized_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
                cv2.imwrite(image, resized_img)

def normalize_keypoint_by_its_bounding_box(keypoints_coordinates):
    #
    # Each keypoint set will be translated its coordinate
    # by its center and scaled by bounding box size.
    # Before doing the above translation and scaling,
    # the distance between the two set is calculated,
    # and then append to the flatted feature vector;
    # thus the length of the feature vector is (100+1)=101
    #
    # input:
    #   -keypoints_coordinates - original pose keypoint coordinates
    #   with the shape= (2,25,2) or (1,25,2)
    # output:
    #   -keypoints_out - list of translated, scaled and flattened keypoint coordinates +
    #   the a distance between the two sets at the last index of the vector
    #

    # check validity
    if keypoints_coordinates.size < 2:
        return ()

    list_of_translated_scaled_keypoints = list()

    # two mean points of the two sets
    two_mean_points = []
    # two bounding box sizes of the two sets
    two_bnb_sizes = []
    for keypoint_set in keypoints_coordinates:
        # calculate its center and box size
        Xs = [item for item in keypoint_set[:, 0] if item > 0]
        Ys = [item for item in keypoint_set[:, 1] if item > 0]

        xMin = np.min(Xs)
        xMax = np.max(Xs)
        yMin = np.min(Ys)
        yMax = np.max(Ys)

        center = ((xMax + xMin)/2, (yMax + yMin)/2)
        box_size = (xMax - xMin, yMax - yMin)

        # if the keyjoint set's points are on a line, or center of box=(0,0), ignore the set
        if (box_size[0] == 0 or box_size[1] ==0 or center[0] == 0 and center[1] ==0):
            continue
        else:
            # add non-zero mean points
            two_mean_points.append(center)
            # add non-zero bounding boxes
            two_bnb_sizes.append(box_size)

        # zip normalized X Y coordinate
        normalized_coordinates = np.array([(item[0],item[1]) if (item[0]==0 or item[1]==0) else ((item[0]-center[0])/box_size[0],(item[1]-center[1])/box_size[1]) for item in keypoint_set])

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
        normalized_delta_X = (two_mean_points[0][0] - two_mean_points[1][0])/(two_bnb_sizes[0][0] +two_bnb_sizes[1][0])
        normalized_delta_Y = (two_mean_points[0][1] - two_mean_points[1][1])/(two_bnb_sizes[0][1] +two_bnb_sizes[1][1])
        dist = np.sqrt(normalized_delta_X**2 + normalized_delta_Y**2)
        list_of_translated_scaled_keypoints += [dist]

    return list_of_translated_scaled_keypoints

if __name__ == "__main__":

    print("Starting preprocessing.py as entry point....")

    ##test: search all images in a directory
    # images = get_all_images_recursively("yoga-pose-dataset")
    # print(images)

    #test: reduce size of an image
    reduce_large_size_images("dataset-humiact5")