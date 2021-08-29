# ============================================================================ #
# Copyrights @vankhanhdo 2021
#
# This file is to pre-process humiact5 dataset:
#   - reducing high resolution images
#   - cleaning the dataset which is to remove images whose key-joints
#   detected by OpenPose are incorrect or not detected
#   - split train/test sets after cleaning step
# ============================================================================ #
import math
import sys
import cv2
import os
import argparse
from random import shuffle
from sys import platform
from os import listdir
from os.path import isfile, isdir, join

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


def interactively_build_a_list_of_error_samples(sub_dir):
    #
    # build a list of error samples in terms of incorrectly detecting key-joints by OpenPose
    # after validating the list built correctly, run remove function to remove error samples
    #

    try:
        # Import Openpose (Windows/Ubuntu/OSX)
        dir_path = os.path.dirname(os.path.realpath(__file__))
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

        # Flags
        parser = argparse.ArgumentParser()
        parser.add_argument("--image_dir", default="dataset-humiact5/{}/".format(sub_dir),
                            help="Process a directory of "
                                 "images. Read "
                                 "all standard formats (jpg, png, bmp, etc.).")
        parser.add_argument("--no_display", default=False, help="Enable to disable the visual display.")
        args = parser.parse_known_args()

        # Custom Params (refer to include/openpose/flags.hpp for more parameters)
        params = dict()
        params["model_folder"] = "./dependencies/openpose/models/"
        params["net_resolution"] = "320x176"

        # Add others in path?
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

        # Construct it from system arguments
        # op.init_argv(args[1])
        # oppython = op.OpenposePython()

        # Starting OpenPose
        opWrapper = op.WrapperPython()
        opWrapper.configure(params)
        opWrapper.start()

        # Read frames on directory
        imagePaths = get_all_images_recursively(args[0].image_dir);
        # start = time.time()

        # save list of image names whose keypoints detected by OpenPose were incorrect
        incorrect_keyjoints_samples = []

        # Process and display images
        for imagePath in imagePaths:
            datum = op.Datum()
            imageToProcess = cv2.imread(imagePath)
            datum.cvInputData = imageToProcess
            opWrapper.emplaceAndPop([datum])

            # print("Body keypoints: \n" + str(datum.poseKeypoints))

            if not args[0].no_display:
                resized_img = cv2.resize(datum.cvOutputData, (960, 540))
                cv2.imshow(imagePath, resized_img)
                key = cv2.waitKey(0)
                if key == 27:
                    break
                elif key == 32:  # SPACE
                    # add to incorrect_keyjoints_samples
                    incorrect_keyjoints_samples.append(imagePath)

                # close current window
                cv2.destroyWindow(imagePath)

        # save the list of incorrect samples to file
        textfile = open("dataset-humiact5/{}_error_samples.txt".format(sub_dir), "w")
        for image_path in incorrect_keyjoints_samples:
            textfile.write(image_path + "\n")
        textfile.close()

        # end = time.time()
        # print("OpenPose demo successfully finished. Total time: " + str(end - start) + " seconds")
    except Exception as e:
        print(e)
        sys.exit(-1)


def remove_error_samples(error_list_file):
    #
    # remove error samples from the built list
    #

    # read the text file
    textfile = open(error_list_file, "r")

    err_samples = [err_file.rstrip("\n") for err_file in textfile.readlines()]

    for err_sampl in err_samples:
        os.remove(err_sampl)

    pass

def split_train_test_set(dataset_path="dataset-humiact5/", train_per_test_ratio = 4):
    #
    # This function is to split the dataset into train/test set
    # Input:
    #   - dataset_path -relative path to dataset
    #   - train_per_test_ratio -the ratio between train set and test set size
    #   The ratio = 4 means train=0.8/test=0.2
    # Output:
    #   - void() -dataset is split into train/ Boxing|Facing|HHold|HShake|XOXO
    #                                   test/  Boxing|Facing|HHold|HShake|XOXO
    #

    # absolute path of dataset
    abs_dataset_path = join(cur_dir, dataset_path)

    # create train/test folder if not exists yet
    abs_train_path = join(abs_dataset_path,"train")
    abs_test_path = join(abs_dataset_path,"test")
    if not os.path.exists(abs_train_path):
        os.mkdir(abs_train_path)
    if not os.path.exists(abs_test_path):
        os.mkdir(abs_test_path)

    # get list of interactions
    categories = [f for f in os.listdir(abs_dataset_path) if isdir(join(abs_dataset_path,f))]
    # remove train/test folder from category list
    categories.remove("train")
    categories.remove("test")

    # process each category in turn
    for cat in categories:
        src_path = join(abs_dataset_path, cat)
        dst_train_path = join(abs_train_path, cat)
        dst_test_path = join(abs_test_path, cat)

        # create folder for current category in train/test folder if not exists yet
        if not os.path.exists(dst_train_path):
            os.mkdir(dst_train_path)
        if not os.path.exists(dst_test_path):
            os.mkdir(dst_test_path)

        # get all samples of this category
        samples = [f for f in listdir(src_path) if isfile(join(src_path,f)) and f.lower().endswith((
            '.png', '.jpg', '.jpeg', '.bmp'))]

        # get number of samples
        N = len(samples)

        # get number of test samples
        N_train = (N*train_per_test_ratio)//(train_per_test_ratio+1)

        # get N_test random samples
        # permutation of N integers (0->N-1)
        indexes = list(range(N))
        shuffle(indexes)
        # get train/test indexes
        train_idxs = indexes[:N_train]
        test_idxs = indexes[N_train:]

        # move train/test samples to corresponding directory
        # move train samples
        for s in train_idxs:
            src_full_path = join(src_path,samples[s])
            dst_train_full_path = join(dst_train_path,samples[s])

            os.rename(src_full_path,dst_train_full_path)
        for s in test_idxs:
            src_full_path = join(src_path, samples[s])
            dst_test_full_path = join(dst_test_path, samples[s])

            os.rename(src_full_path, dst_test_full_path)

        # check if current source folder is empty, then remove the folder
        if len(listdir(src_path)) == 0:
            os.rmdir(src_path)

    pass

def convert_image_format(image_dir, target_format = "jpg"):
    #
    # convert all images to target image format
    # Input:
    #   -image_dir -images' directory
    #   -target_format -target image format
    # Output:
    #   -void() -replace original images by converted images
    #

    # absolute path of image directory
    abs_image_dir = join(cur_dir, image_dir)

    # get all images in the directory
    images = [f for f in listdir(abs_image_dir) if isfile(join(abs_image_dir,f))]

    for image in images:
        image_full_path = join(abs_image_dir,image)
        without_ext, ext = os.path.splitext(image_full_path)
        
        if ext[1:] == target_format:
            continue
            
        # open the image
        data = cv2.imread(image_full_path)

        # save back the image
        dst_image_full_path = "{}.{}".format(without_ext,target_format)
        
        if not os.path.exists(dst_image_full_path):
            cv2.imwrite(dst_image_full_path,data, [cv2.IMWRITE_JPEG_QUALITY, 95])
            os.remove(image_full_path)
        else:
            # attempt another name
            attempt = 1
            while True:
                extra_symbol = attempt + 1
                new_dst_image_full_path = "{}_{}.{}".format(without_ext, extra_symbol, target_format)
                if not os.path.exists(new_dst_image_full_path):
                    cv2.imwrite(new_dst_image_full_path, data, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    os.remove(image_full_path)
                    break
                attempt += 1   

    pass

if __name__ == "__main__":

    print("Starting preprocessing.py as entry point....")
    # get current absolute path
    cur_dir = os.path.dirname(__file__)

    # ============================================================ #
    # test: search all images in a directory
    # images = get_all_images_recursively("yoga-pose-dataset")
    # print(images)

    # ============================================================ #

    # ============================================================ #
    # test: reduce size of an image
    # reduce_large_size_images("dataset-humiact5")

    # ============================================================ #

    # ============================================================ #
    # interactively_build_a_list_of_error_samples("XOXO")

    # ============================================================ #

    # ============================================================ #
    # the file of list of error samples
    # error_list_file = "dataset-humiact5/XOXO_error_samples.txt"
    # remove_error_samples(error_list_file)

    # ============================================================ #

    # ============================================================ #
    # split dataset into train/test sets
    split_train_test_set(dataset_path="dataset-humiact5/", train_per_test_ratio = 4)
    # ============================================================ #

    # ============================================================ #
    # convert image format
    # convert_image_format("dataset-humiact5/XOXO/")
    # ============================================================ #

    pass