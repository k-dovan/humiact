
# ===================================================== #
#   OpenPose Library Tutorial 01
# ===================================================== #

# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
from sys import platform
import argparse
import numpy as np
from humiact5_feature_engineering import estimate_combined_bounding_box

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
        print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
        raise e

    # Flags
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", default="media/test/kising (340).jpg",
                        help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
    args = parser.parse_known_args()

    # Custom Params (refer to include/openpose/flags.hpp for more parameters)
    params = dict()
    params["model_folder"] = "./dependencies/openpose/models/"
    params["net_resolution"] = "320x176"

    # Add others in path?
    for i in range(0, len(args[1])):
        curr_item = args[1][i]
        if i != len(args[1])-1: next_item = args[1][i+1]
        else: next_item = "1"
        if "--" in curr_item and "--" in next_item:
            key = curr_item.replace('-','')
            if key not in params:  params[key] = "1"
        elif "--" in curr_item and "--" not in next_item:
            key = curr_item.replace('-','')
            if key not in params: params[key] = next_item

    # Construct it from system arguments
    # op.init_argv(args[1])
    # oppython = op.OpenposePython()

    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    # Process Image
    datum = op.Datum()
    imageToProcess = cv2.imread(args[0].image_path)
    datum.cvInputData = imageToProcess
    opWrapper.emplaceAndPop([datum])

    # estimate boundinig boxes
    img_height, img_width, _ = datum.cvOutputData.shape
    bxs = estimate_combined_bounding_box(datum.poseKeypoints,
                                         img_width,
                                         img_height,
                                         0.10,
                                         0.10)

    # draw bounding boxes on output image
    bnb_img = datum.cvOutputData
    for bx in bxs:
        cv2.rectangle(bnb_img,
                      (bx[0], bx[1]),
                      (bx[2], bx[3]),
                      color=(0, 0, 255),
                      thickness=2)

    # display image with bounding boxes
    resized_bnb_img = cv2.resize(bnb_img, (960, 540))
    cv2.imshow("Bounding box cropped", resized_bnb_img)
    cv2.waitKey(0)

    # # Display Image
    # print("Body keypoints: \n" + str(datum.poseKeypoints))
    # resized_img = cv2.resize(datum.cvOutputData, (960,540))
    # cv2.imshow("OpenPose 1.6.0 - Tutorial Python API", resized_img)
    # cv2.waitKey(0)
except Exception as e:
    print(e)
    sys.exit(-1)
