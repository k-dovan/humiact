
# ============================================================================ #
# This file is to clean self-made dataset which intends to remove
# images whose keyjoints detected by OpenPose are incorrect or not detected
# ============================================================================ #

# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
from sys import platform
import argparse
import time

from humiact5_preprocessing import get_all_images_recursively

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
        parser.add_argument("--image_dir", default="dataset-humiact5/{}/".format(sub_dir), help="Process a directory of "
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
    textfile = open(error_list_file,"r")

    err_samples = [err_file.rstrip("\n") for err_file in textfile.readlines()]

    for err_sampl in err_samples:
        os.remove(err_sampl)

    pass

if __name__ == "__main__":
    # interactively_build_a_list_of_error_samples("XOXO")

    # the file of list of error samples
    error_list_file = "dataset-humiact5/XOXO_error_samples.txt"

    remove_error_samples(error_list_file)
    pass