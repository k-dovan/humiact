
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

        images += [join(cur_dir, f) for f in listdir(cur_dir) if isfile(join(cur_dir, f))]
        search_dirs += [join(cur_dir,f) for f in listdir(cur_dir) if isdir(join(cur_dir,f))]

    return images

def reduce_large_size_images(path):

    # Reduce every large size images in a directory recursively
    # input:
    #   path - dataset directory
    # output:
    #   void - all resized images are saved on the disk

    # set resolution threshold
    MAX_WIDTH = 3840
    MAX_HEIGHT = 3840

    # get all images from dataset path
    images = get_all_images_recursively(path)

    # reduce large-size images to get CUDNN
    # running well on old model graphic card
    for image in images:
        img = cv2.imread(image)
        if img is not None:
            scale = 1
            height, width, _ = img.shape
            if width > MAX_WIDTH:
                scale = width/MAX_WIDTH + 1
            else:
                if height > MAX_HEIGHT:
                    scale = height/MAX_HEIGHT + 1
            if scale > 1:
                width = int(width/scale)
                height = int(height/scale)
                dim = (width, height)

                # reduce image to the smaller size
                resized_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
                cv2.imwrite(image, resized_img)

def normalize_keypoint_by_its_bounding_box(keypoints_coordinates):
    #
    # keypoint vectors will be translated its coordinate
    # by its center and scaled by bounding box size
    #
    # input:
    #   -keypoints_coordinates - original pose keypoint coordinates
    # output:
    #   -keypoints_out - list of translated, scaled and flattened keypoint coordinates
    #

    # check validity
    if keypoints_coordinates.size < 2:
        return ()

    list_of_translated_scaled_keypoints = list()
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

        if box_size == 0: return ()

        # zip normalized X Y coordinate
        normalized_coordinates = np.array([(item[0],item[1]) if (item[0]==0 or item[1]==0) else ((item[0]-center[0])/box_size[0],(item[1]-center[1])/box_size[1]) for item in keypoint_set])

        # flatten the normalized coordinates to feature vector
        list_of_translated_scaled_keypoints.append(normalized_coordinates.reshape(-1))

    # reshape the list to array
    list_of_translated_scaled_keypoints = [item for sublist in list_of_translated_scaled_keypoints \
                                           for item in sublist]

    return list_of_translated_scaled_keypoints

if __name__ == "__main__":

    print("Starting preprocessing.py as entry point....")

    ##test: search all images in a directory
    # images = get_all_images_recursively("yoga-pose-dataset")
    # print(images)

    #test: reduce size of an image
    reduce_large_size_images("dataset/")